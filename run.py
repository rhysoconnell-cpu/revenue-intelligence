import os
import ssl
import smtplib
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dateutil import tz
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# =========================
# SETTINGS (SAFE DEFAULTS)
# =========================

NY_TZ = tz.gettz("America/New_York")

# Use NAME-ONLY (plus Brooklyn, NY) — no street addresses
YOUR_HOTEL = "The Bay Ridge Hotel, Brooklyn, NY"

HOTELS = [
    "The Bay Ridge Hotel, Brooklyn, NY",
    "avid hotel Brooklyn - Dyker Heights by IHG, Brooklyn, NY",
    "Insignia Hotel, Brooklyn, NY",
    "Umbrella Hotel, Brooklyn, NY",
    "Best Western Gregory Hotel, Brooklyn, NY",
]

# Forward-looking windows
OFFSETS = [0, 7, 14]

# Recommendation guardrails
MIN_RATE = 99
MAX_RATE = 399

# Simple explainable rules (tune later)
RAISE_IF_BELOW_AVG_PCT = 0.08
DROP_IF_ABOVE_AVG_PCT = 0.10
BASE_STEP = 10
WEEKEND_BONUS = 10
EVENT_BONUS = 10


# =========================
# ENV VARS (GitHub Secrets)
# =========================

SERP_API_KEY = os.getenv("SERP_API_KEY", "").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()

EMAIL_TO  = os.getenv("EMAIL_TO", "Rhys.oconnell@yourplacehotel.com").strip()
DEBUG = os.getenv("DEBUG", "0") == "1"

# Standardize the search
CURRENCY = "USD"
ADULTS = 2
ROOMS = 1

# IMPORTANT: geo hints so Google Hotels returns results consistently
GOOGLE_GL = "us"
GOOGLE_HL = "en"
GOOGLE_LOCATION = "Brooklyn, New York, United States"


# =========================
# DATA MODEL
# =========================

@dataclass
class Rate:
    hotel_query: str
    check_in: dt.date
    check_out: dt.date

    # What we report:
    entry_rate: Optional[float]     # nightly price
    source: str = "Unknown"         # e.g. Booking.com or "Google Hotels (base)"
    source_url: str = ""            # link if provided

    # For verification (optional)
    matched_name: str = ""
    matched_address: str = ""

    @property
    def available(self) -> bool:
        return self.entry_rate is not None


# =========================
# HELPERS
# =========================

def ny_today() -> dt.date:
    return dt.datetime.now(tz=NY_TZ).date()

def money(x: Optional[float]) -> str:
    return "N/A" if x is None else f"${x:,.0f}"

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def is_weekend(d: dt.date) -> bool:
    return d.weekday() in (4, 5)  # Fri/Sat

def avg(vals: List[float]) -> Optional[float]:
    return None if not vals else sum(vals) / len(vals)


# =========================
# SERPAPI — GOOGLE HOTELS
# Strategy:
# 1) Prefer lowest OTA row from chosen["prices"] (Featured options)
# 2) If prices[] missing, FALL BACK to chosen["rate_per_night"] base
# =========================

def fetch_google_hotels_rate(hotel_query: str, check_in: dt.date, check_out: dt.date) -> Rate:
    if not SERP_API_KEY:
        raise RuntimeError("Missing SERP_API_KEY (add it in GitHub Secrets).")

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_hotels",
        "q": hotel_query,
        "check_in_date": check_in.isoformat(),
        "check_out_date": check_out.isoformat(),
        "currency": CURRENCY,
        "adults": ADULTS,
        "rooms": ROOMS,
        "gl": GOOGLE_GL,
        "hl": GOOGLE_HL,
        "location": GOOGLE_LOCATION,
        "api_key": SERP_API_KEY,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    props = data.get("properties") or []
    chosen = None

    # Pick the best-matching property result
    if isinstance(props, list) and props:
        target = hotel_query.lower()
        chosen = props[0]
        for p in props:
            nm = (p.get("name") or "").lower()
            if nm and (nm in target or target in nm):
                chosen = p
                break

    entry_rate = None
    source = "Unknown"
    source_url = ""
    matched_name = ""
    matched_address = ""

    if isinstance(chosen, dict):
        matched_name = str(chosen.get("name") or "")
        matched_address = str(chosen.get("address") or "")

        # 1) Preferred: lowest OTA row from prices[]
        prices = chosen.get("prices")
        if isinstance(prices, list) and prices:
            best = None
            for p in prices:
                if not isinstance(p, dict):
                    continue

                rate = p.get("extracted_rate")
                if rate is None:
                    rate = p.get("rate")
                if not isinstance(rate, (int, float)) or rate <= 0:
                    continue

                name = p.get("name") or p.get("source") or "Unknown"
                link = p.get("link") or p.get("url") or ""

                if best is None or float(rate) < best["rate"]:
                    best = {"rate": float(rate), "name": str(name), "link": str(link)}

            if best:
                entry_rate = best["rate"]
                source = best["name"]
                source_url = best["link"]

        # 2) Fallback: Google Hotels base rate_per_night (if OTA rows missing)
        if entry_rate is None:
            rpn = chosen.get("rate_per_night") if isinstance(chosen.get("rate_per_night"), dict) else None
            if rpn:
                for k in ("extracted_lowest", "lowest"):
                    v = rpn.get(k)
                    if isinstance(v, (int, float)) and v > 0:
                        entry_rate = float(v)
                        source = "Google Hotels (base)"
                        source_url = chosen.get("link") or ""
                        break

        if DEBUG:
            print("\n==== DEBUG RATE PULL ====")
            print("QUERY:", hotel_query)
            print("DATE:", check_in.isoformat(), "→", check_out.isoformat())
            print("MATCHED_NAME:", matched_name)
            print("MATCHED_ADDRESS:", matched_address)
            print("SELECTED_RATE:", entry_rate)
            print("SELECTED_SOURCE:", source)
            print("SELECTED_SOURCE_URL:", source_url)
            print("HAS_PRICES_LIST:", isinstance(chosen.get("prices"), list))
            print("RATE_PER_NIGHT:", chosen.get("rate_per_night"))
            print("=========================\n")

    return Rate(
        hotel_query=hotel_query,
        check_in=check_in,
        check_out=check_out,
        entry_rate=entry_rate,
        source=source,
        source_url=source_url,
        matched_name=matched_name,
        matched_address=matched_address,
    )


# =========================
# EVENTS (Forward-looking)
# =========================

def fetch_barclays_events(next_days: int = 14) -> List[Tuple[dt.date, str]]:
    try:
        r = requests.get("https://www.barclayscenter.com/events", timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]

    today = ny_today()
    end = today + dt.timedelta(days=next_days)

    month_map = {m: i for i, m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"], start=1
    )}

    out: List[Tuple[dt.date, str]] = []
    i = 0
    while i < len(lines) - 1:
        parts = lines[i].split()
        if len(parts) == 2 and parts[0] in month_map and parts[1].isdigit():
            month = month_map[parts[0]]
            day = int(parts[1])
            year = today.year
            if today.month == 12 and month == 1:
                year += 1
            try:
                d = dt.date(year, month, day)
            except ValueError:
                i += 1
                continue

            if today <= d <= end:
                title = lines[i + 1]
                if title and len(title) > 3 and "Buy" not in title:
                    out.append((d, title))
        i += 1

    seen = set()
    uniq = []
    for d, t in out:
        key = (d.isoformat(), t)
        if key not in seen:
            uniq.append((d, t))
            seen.add(key)
    return uniq

def build_event_map() -> Dict[dt.date, List[str]]:
    ev = fetch_barclays_events(next_days=14)
    m: Dict[dt.date, List[str]] = {}
    for d, t in ev:
        m.setdefault(d, []).append(t)
    return m


# =========================
# ANALYSIS (Forward-looking)
# =========================

def recommend(bucket_rates: List[Rate], check_in: dt.date, events: Dict[dt.date, List[str]]) -> Tuple[str, str]:
    your = next((r for r in bucket_rates if r.hotel_query == YOUR_HOTEL), None)
    comps = [r for r in bucket_rates if r.hotel_query != YOUR_HOTEL]

    comp_vals = [r.entry_rate for r in comps if r.entry_rate is not None]
    comp_avg = avg([float(v) for v in comp_vals]) if comp_vals else None

    has_event = bool(events.get(check_in))
    weekend = is_weekend(check_in)

    if not your or your.entry_rate is None or comp_avg is None:
        return ("HOLD (insufficient data)", "Some rates were not found — try slightly different hotel names.")

    your_rate = float(your.entry_rate)
    diff_pct = (comp_avg - your_rate) / comp_avg  # positive => below market

    action = "HOLD"
    step = 0

    if diff_pct >= RAISE_IF_BELOW_AVG_PCT:
        action = "RAISE"
        step = BASE_STEP + (WEEKEND_BONUS if weekend else 0) + (EVENT_BONUS if has_event else 0)
    elif (your_rate - comp_avg) / comp_avg >= DROP_IF_ABOVE_AVG_PCT:
        action = "DROP"
        step = BASE_STEP

    target = int(round(your_rate))
    if action == "RAISE":
        target += step
    elif action == "DROP":
        target -= step

    target = clamp(target, MIN_RATE, MAX_RATE)

    opp = []
    if diff_pct > 0:
        opp.append(f"Bay Ridge is BELOW comp avg by {diff_pct*100:.1f}%.")
    else:
        opp.append(f"Bay Ridge is ABOVE comp avg by {abs(diff_pct)*100:.1f}%.")

    if weekend:
        opp.append("Weekend demand bias (Fri/Sat).")
    if has_event:
        opp.append("Event demand driver detected (Brooklyn).")

    rec_line = "HOLD"
    if action == "RAISE":
        rec_line = f"🔼 RAISE to {money(target)}"
    elif action == "DROP":
        rec_line = f"🔽 DROP to {money(target)}"

    return (rec_line, " ".join(opp))


# =========================
# EMAIL
# =========================

def send_email(subject: str, body: str) -> None:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        raise RuntimeError("Missing SMTP credentials. Check GitHub Secrets.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(body, "plain"))

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
        server.starttls(context=context)
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [EMAIL_TO], msg.as_string())


def build_email(windows: Dict[int, List[Rate]], events: Dict[dt.date, List[str]]) -> Tuple[str, str]:
    today = ny_today()
    now = dt.datetime.now(tz=NY_TZ)
    subject = f"Bay Ridge — Forward Revenue Intelligence (Tonight / 7 / 14) | {today.isoformat()}"

    lines: List[str] = []
    lines.append("BAY RIDGE HOTEL — FORWARD REVENUE INTELLIGENCE")
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %I:%M %p')} ET")
    lines.append("Source: SerpApi → Google Hotels (lowest OTA offer if available, else base rate)")
    lines.append("")

    # Executive summary
    lines.append("EXECUTIVE SUMMARY")
    for off in OFFSETS:
        label = "Tonight (D0)" if off == 0 else f"D+{off}"
        date = today + dt.timedelta(days=off)
        rec, _ = recommend(windows[off], date, events)
        lines.append(f"- {label} ({date.isoformat()}): {rec}")
    lines.append("")

    # Detail sections
    for off in OFFSETS:
        date = today + dt.timedelta(days=off)
        label = "Tonight (D0)" if off == 0 else f"D+{off}"

        lines.append("=" * 88)
        lines.append(f"{label} — Check-in {date.isoformat()} (1 night)")
        lines.append("=" * 88)

        bucket = windows[off]
        avail = [r for r in bucket if r.entry_rate is not None]
        unavail = [r for r in bucket if r.entry_rate is None]
        avail.sort(key=lambda r: r.entry_rate)

        lines.append("ENTRY RATE RANKING (price + OTA source)")
        i = 1
        for r in avail:
            marker = "  <-- YOUR HOTEL" if r.hotel_query == YOUR_HOTEL else ""
            src = f"{r.source}" if r.source else "Unknown"
            link = f" | {r.source_url}" if r.source_url else ""
            # Include matched listing info when DEBUG is ON (helps QA)
            if DEBUG and (r.matched_name or r.matched_address):
                lines.append(f"{i}. {r.hotel_query} — {money(r.entry_rate)} ({src}){link}{marker}")
                lines.append(f"    Matched: {r.matched_name} | {r.matched_address}")
            else:
                lines.append(f"{i}. {r.hotel_query} — {money(r.entry_rate)} ({src}){link}{marker}")
            i += 1

        for r in unavail:
            marker = "  <-- YOUR HOTEL" if r.hotel_query == YOUR_HOTEL else ""
            lines.append(f"{i}. {r.hotel_query} — N/A (not found){marker}")
            i += 1
        lines.append("")

        ev = events.get(date, [])
        lines.append("BROOKLYN EVENTS / DEMAND DRIVERS (forward-looking)")
        if ev:
            for t in ev[:8]:
                lines.append(f"- {t}")
        else:
            lines.append("- None detected (Barclays feed).")
        lines.append("")

        rec, opp = recommend(bucket, date, events)
        lines.append("OPPORTUNITIES")
        lines.append(f"- {opp}")
        lines.append("")
        lines.append("RECOMMENDATION")
        lines.append(f"- {rec}")
        lines.append("")

    lines.append("=" * 88)
    lines.append("NOTES")
    lines.append("- If Google returns OTA rows, we report the lowest OTA offer + OTA name/link (owner-friendly).")
    lines.append("- If OTA rows are missing, we fall back to Google Hotels base rate to avoid N/A.")
    lines.append("- Google may show promo badges (e.g., 'Deal') that differ from standard OTA rows; OTA source clarifies this.")
    lines.append("")

    return subject, "\n".join(lines)


def main():
    today = ny_today()
    events = build_event_map()

    windows: Dict[int, List[Rate]] = {}
    for off in OFFSETS:
        check_in = today + dt.timedelta(days=off)
        check_out = check_in + dt.timedelta(days=1)

        bucket: List[Rate] = []
        for h in HOTELS:
            bucket.append(fetch_google_hotels_rate(h, check_in, check_out))
        windows[off] = bucket

    subject, body = build_email(windows, events)
    send_email(subject, body)
    print("Sent:", subject)


if __name__ == "__main__":
    main()
