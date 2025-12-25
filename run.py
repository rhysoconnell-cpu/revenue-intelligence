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
# SETTINGS (EDIT THESE)
# =========================

NY_TZ = tz.gettz("America/New_York")

YOUR_HOTEL = "The Bay Ridge Hotel, 315 93rd St, Brooklyn, NY 11209",

HOTELS = [
    The Bay Ridge Hotel, 315 93rd St, Brooklyn, NY 11209",
    "Insignia Hotel Dyker Heights",
    "avid hotel Brooklyn - Dyker Heights by IHG",
    "Umbrella Hotel Brooklyn",
    "Best Western Gregory Hotel",
]

# Forward-looking windows:
# Tonight (D0), D+7, D+14
OFFSETS = [0, 7, 14]

# Recommendation guardrails (edit later)
MIN_RATE = 99
MAX_RATE = 399

# How aggressive to move (simple deterministic logic)
RAISE_IF_BELOW_AVG_PCT = 0.08   # if Bay Ridge is 8%+ below comp avg => raise
DROP_IF_ABOVE_AVG_PCT  = 0.10   # if Bay Ridge is 10%+ above comp avg => drop
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

# Google Hotels query standardization
CURRENCY = "USD"
ADULTS = 2
ROOMS = 1


# =========================
# DATA MODEL
# =========================

@dataclass
class Rate:
    hotel: str
    check_in: dt.date
    check_out: dt.date
    entry_rate: Optional[float]  # base rate only
    source: str = "SerpApi / Google Hotels"

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
    # Fri or Sat
    return d.weekday() in (4, 5)

def avg(vals: List[float]) -> Optional[float]:
    return None if not vals else sum(vals) / len(vals)


# =========================
# SERPAPI — GOOGLE HOTELS (BASE RATE ONLY)
# =========================

def fetch_google_hotels_base_rate(hotel: str, check_in: dt.date, check_out: dt.date) -> Rate:
    """
    Pulls Google Hotels via SerpApi, attempting to extract BASE entry rate (rate_per_night),
    avoiding taxes/fees totals.
    """
    if not SERP_API_KEY:
        raise RuntimeError("Missing SERP_API_KEY (add it in GitHub Secrets).")

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_hotels",
        "q": hotel,
        "check_in_date": check_in.isoformat(),
        "check_out_date": check_out.isoformat(),
        "currency": CURRENCY,
        "adults": ADULTS,
        "rooms": ROOMS,
        "api_key": SERP_API_KEY,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # SerpApi often returns a list like: data["properties"]
    props = data.get("properties") or []
    chosen = None

    if isinstance(props, list) and props:
        # Try to match by name loosely; otherwise take first result
        target = hotel.lower()
        chosen = props[0]
        for p in props:
            nm = (p.get("name") or "").lower()
            if nm and (nm in target or target in nm):
                chosen = p
                break

    base_rate = None

    if isinstance(chosen, dict):
        # BASE RATE ONLY: prefer rate_per_night -> lowest/extracted_lowest
        rpn = chosen.get("rate_per_night") if isinstance(chosen.get("rate_per_night"), dict) else None
        if rpn:
            candidates = []
            for k in ("lowest", "extracted_lowest"):
                v = rpn.get(k)
                if isinstance(v, (int, float)) and v > 0:
                    candidates.append(float(v))
            if candidates:
                base_rate = min(candidates)

        # Sometimes base nightly rate appears directly
        if base_rate is None:
            for k in ("price", "extracted_price"):
                v = chosen.get(k)
                if isinstance(v, (int, float)) and v > 0:
                    base_rate = float(v)
                    break

        if DEBUG:
            print("DEBUG HOTEL:", hotel, check_in, "CHOSEN_KEYS:", list(chosen.keys()))
            print("DEBUG rate_per_night:", chosen.get("rate_per_night"))
            print("DEBUG price/extracted_price:", chosen.get("price"), chosen.get("extracted_price"))
            print("DEBUG total_rate (ignored):", chosen.get("total_rate"))

    return Rate(hotel=hotel, check_in=check_in, check_out=check_out, entry_rate=base_rate)


# =========================
# EVENTS (FORWARD LOOKING)
# =========================

def fetch_barclays_events(next_days: int = 14) -> List[Tuple[dt.date, str]]:
    """
    Lightweight scrape of Barclays Center events page for demand flags.
    If it fails, it returns [] and your email still works.
    """
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

    # de-dupe
    seen = set()
    uniq = []
    for d, t in out:
        key = (d.isoformat(), t)
        if key not in seen:
            uniq.append((d, t))
            seen.add(key)
    return uniq

def build_event_map() -> Dict[dt.date, List[str]]:
    """
    Returns dict: {date: [event titles]}
    """
    ev = fetch_barclays_events(next_days=14)
    m: Dict[dt.date, List[str]] = {}
    for d, t in ev:
        m.setdefault(d, []).append(t)
    return m


# =========================
# ANALYSIS (FORWARD ONLY)
# =========================

def recommend(bucket_rates: List[Rate], check_in: dt.date, events: Dict[dt.date, List[str]]) -> Tuple[str, str]:
    """
    Returns: (recommendation_line, opportunities_line)
    """
    your = next((r for r in bucket_rates if r.hotel == YOUR_HOTEL), None)
    comps = [r for r in bucket_rates if r.hotel != YOUR_HOTEL]

    comp_vals = [r.entry_rate for r in comps if r.entry_rate is not None]
    comp_avg = avg([float(v) for v in comp_vals]) if comp_vals else None

    has_event = bool(events.get(check_in))
    weekend = is_weekend(check_in)

    if not your or your.entry_rate is None or comp_avg is None:
        return ("HOLD (insufficient data)", "Some rates were not found — try slightly different hotel names.")

    your_rate = float(your.entry_rate)
    diff_pct = (comp_avg - your_rate) / comp_avg  # positive => you are below market

    # Decide action
    action = "HOLD"
    step = 0

    if diff_pct >= RAISE_IF_BELOW_AVG_PCT:
        action = "RAISE"
        step = BASE_STEP
        if weekend:
            step += WEEKEND_BONUS
        if has_event:
            step += EVENT_BONUS
    elif (your_rate - comp_avg) / comp_avg >= DROP_IF_ABOVE_AVG_PCT:
        action = "DROP"
        step = BASE_STEP

    target = int(round(your_rate))
    if action == "RAISE":
        target += step
    elif action == "DROP":
        target -= step

    target = clamp(target, MIN_RATE, MAX_RATE)

    # Text
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


def rank_lines(bucket_rates: List[Rate]) -> List[str]:
    avail = [r for r in bucket_rates if r.entry_rate is not None]
    unavail = [r for r in bucket_rates if r.entry_rate is None]
    avail.sort(key=lambda r: r.entry_rate)
    lines = []
    i = 1
    for r in avail:
        marker = "  <-- YOUR HOTEL" if r.hotel == YOUR_HOTEL else ""
        lines.append(f"{i}. {r.hotel} — {money(r.entry_rate)}{marker}")
        i += 1
    for r in unavail:
        marker = "  <-- YOUR HOTEL" if r.hotel == YOUR_HOTEL else ""
        lines.append(f"{i}. {r.hotel} — N/A (not found){marker}")
        i += 1
    return lines


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
    lines.append("Source: SerpApi → Google Hotels (base entry rate)")
    lines.append("")

    # Executive summary
    lines.append("EXECUTIVE SUMMARY")
    for off in OFFSETS:
        label = "Tonight (D0)" if off == 0 else (f"D+{off}")
        bucket = windows[off]
        date = today + dt.timedelta(days=off)
        rec, _ = recommend(bucket, date, events)
        lines.append(f"- {label} ({date.isoformat()}): {rec}")
    lines.append("")

    # Detail sections
    for off in OFFSETS:
        date = today + dt.timedelta(days=off)
        label = "Tonight (D0)" if off == 0 else f"D+{off}"
        bucket = windows[off]

        lines.append("=" * 70)
        lines.append(f"{label} — Check-in {date.isoformat()} (1 night)")
        lines.append("=" * 70)
        lines.append("ENTRY RATE RANKING (cheapest available base rate)")
        lines.extend(rank_lines(bucket))
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

    lines.append("=" * 70)
    lines.append("NOTES")
    lines.append("- Rates are live public entry rates at time of run (Google Hotels base rate view).")
    lines.append("- If any hotel shows N/A, adjust the hotel name string (add 'Brooklyn NY').")
    lines.append("")

    return subject, "\n".join(lines)


def main():
    today = ny_today()

    # events map for next 14 days
    events = build_event_map()

    windows: Dict[int, List[Rate]] = {}
    for off in OFFSETS:
        check_in = today + dt.timedelta(days=off)
        check_out = check_in + dt.timedelta(days=1)

        bucket: List[Rate] = []
        for h in HOTELS:
            bucket.append(fetch_google_hotels_base_rate(h, check_in, check_out))
        windows[off] = bucket

    subject, body = build_email(windows, events)
    send_email(subject, body)
    print("Sent:", subject)


if __name__ == "__main__":
    main()
