import os
import json
import math
import smtplib
import ssl
import datetime as dt
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dateutil import tz

# -----------------------------
# CONFIG
# -----------------------------

NY_TZ = tz.gettz("America/New_York")

HOTELS = [
    "Bay Ridge Hotel Brooklyn",
    "Insignia Hotel Dyker Heights",
    "Avid Hotel Dyker Heights",
    "Umbrella Hotel Brooklyn",
    "Best Western Gregory Hotel",
]

YOUR_HOTEL = "Bay Ridge Hotel Brooklyn"

# 1-night stay check-in dates: Tonight, +7, +14
DATE_OFFSETS = [0, 7, 14]

# Pricing logic knobs (tune later)
RAISE_IF_UNDER_COMP_AVG_PCT = 0.08   # if you're >8% below comp avg, raise
DROP_IF_OVER_COMP_AVG_PCT  = 0.10   # if you're >10% above comp avg, drop
BASE_RAISE_STEP = 10
BASE_DROP_STEP  = 10
WEEKEND_RAISE_BONUS = 10            # extra raise on Fri/Sat if underpriced
EVENT_RAISE_BONUS = 10              # extra raise when event flag triggers
MIN_RATE = 99                       # guardrail (edit)
MAX_RATE = 299                      # guardrail (edit)

# Data source: "serpapi" or "searchapi"
PROVIDER = os.getenv("RATES_PROVIDER", "searchapi").strip().lower()

SERP_API_KEY = os.getenv("SERP_API_KEY", "").strip()  # used for either provider

# Email
EMAIL_TO = os.getenv("EMAIL_TO", "Rhys.oconnell@yourplacehotel.com").strip()
EMAIL_FROM = os.getenv("SMTP_USER", "").strip()
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()

# Optional: to reduce false rate variance, fix the "geo" inputs for SERP providers
DEFAULT_CURRENCY = "USD"
ADULTS = 2
CHILDREN = 0
ROOMS = 1


# -----------------------------
# MODELS
# -----------------------------

@dataclass
class RateResult:
    hotel: str
    check_in: dt.date
    check_out: dt.date
    entry_rate: Optional[float]   # None if unavailable / not found
    currency: str
    provider: str
    raw: dict

    @property
    def is_available(self) -> bool:
        return self.entry_rate is not None


# -----------------------------
# UTILITIES
# -----------------------------

def ny_today() -> dt.date:
    return dt.datetime.now(tz=NY_TZ).date()

def fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"${x:,.0f}"

def pct(a: float, b: float) -> float:
    # (a - b) / b
    if b == 0:
        return 0.0
    return (a - b) / b

def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))

def is_weekend(d: dt.date) -> bool:
    # Fri (4) / Sat (5)
    return d.weekday() in (4, 5)

def safe_get(d: dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

# -----------------------------
# PROVIDERS (GOOGLE HOTELS VIA API)
# -----------------------------

def fetch_rate_serpapi(hotel: str, check_in: dt.date, check_out: dt.date) -> RateResult:
    """
    SerpApi: Google Hotels engine.
    Docs vary; we keep it defensive.
    """
    if not SERP_API_KEY:
        raise RuntimeError("Missing SERP_API_KEY")

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_hotels",
        "q": hotel,
        "check_in_date": check_in.isoformat(),
        "check_out_date": check_out.isoformat(),
        "adults": ADULTS,
        "children": CHILDREN,
        "rooms": ROOMS,
        "currency": DEFAULT_CURRENCY,
        "api_key": SERP_API_KEY,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Try several known layouts.
    # Common: data["properties"] list with ["rate_per_night"]["lowest"] or ["prices"].
    entry = None

    properties = data.get("properties") or data.get("hotel_results") or []
    if isinstance(properties, list) and properties:
        # Find the matching hotel result by name similarity; otherwise take first
        chosen = properties[0]
        hl = hotel.lower()
        for p in properties:
            name = (p.get("name") or p.get("hotel_name") or "").lower()
            if name and name in hl or hl in name:
                chosen = p
                break

        # Candidate price fields (defensive)
        candidates = []
        # Example structures:
        candidates.append(safe_get(chosen, ["rate_per_night", "lowest"]))
        candidates.append(safe_get(chosen, ["rate_per_night", "extracted_lowest"]))
        candidates.append(safe_get(chosen, ["total_rate", "lowest"]))
        candidates.append(chosen.get("price") if isinstance(chosen.get("price"), (int, float)) else None)
        candidates.append(chosen.get("extracted_price") if isinstance(chosen.get("extracted_price"), (int, float)) else None)

        # Sometimes there is a list of "prices"
        prices_list = chosen.get("prices")
        if isinstance(prices_list, list):
            for pr in prices_list:
                if isinstance(pr, dict):
                    candidates.append(pr.get("rate") if isinstance(pr.get("rate"), (int, float)) else None)
                    candidates.append(pr.get("extracted_rate") if isinstance(pr.get("extracted_rate"), (int, float)) else None)

        # Choose minimum numeric candidate
        nums = [c for c in candidates if isinstance(c, (int, float)) and c > 0]
        if nums:
            entry = float(min(nums))

    return RateResult(
        hotel=hotel,
        check_in=check_in,
        check_out=check_out,
        entry_rate=entry,
        currency=DEFAULT_CURRENCY,
        provider="serpapi",
        raw=data,
    )

def fetch_rate_searchapi(hotel: str, check_in: dt.date, check_out: dt.date) -> RateResult:
    """
    SearchApi (searchapi.io) also supports Google Hotels-like results.
    API formats can vary by plan/endpoint, so we keep it defensive.
    """
    if not SERP_API_KEY:
        raise RuntimeError("Missing SERP_API_KEY")

    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": "google_hotels",
        "q": hotel,
        "check_in_date": check_in.isoformat(),
        "check_out_date": check_out.isoformat(),
        "adults": ADULTS,
        "children": CHILDREN,
        "rooms": ROOMS,
        "currency": DEFAULT_CURRENCY,
        "api_key": SERP_API_KEY,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    entry = None
    properties = data.get("properties") or data.get("hotel_results") or data.get("results") or []
    if isinstance(properties, list) and properties:
        chosen = properties[0]
        hl = hotel.lower()
        for p in properties:
            name = (p.get("name") or p.get("hotel_name") or "").lower()
            if name and (name in hl or hl in name):
                chosen = p
                break

        candidates = []
        candidates.append(safe_get(chosen, ["rate_per_night", "lowest"]))
        candidates.append(safe_get(chosen, ["rate_per_night", "extracted_lowest"]))
        candidates.append(chosen.get("price") if isinstance(chosen.get("price"), (int, float)) else None)
        candidates.append(chosen.get("extracted_price") if isinstance(chosen.get("extracted_price"), (int, float)) else None)

        prices_list = chosen.get("prices")
        if isinstance(prices_list, list):
            for pr in prices_list:
                if isinstance(pr, dict):
                    candidates.append(pr.get("rate") if isinstance(pr.get("rate"), (int, float)) else None)
                    candidates.append(pr.get("extracted_rate") if isinstance(pr.get("extracted_rate"), (int, float)) else None)

        nums = [c for c in candidates if isinstance(c, (int, float)) and c > 0]
        if nums:
            entry = float(min(nums))

    return RateResult(
        hotel=hotel,
        check_in=check_in,
        check_out=check_out,
        entry_rate=entry,
        currency=DEFAULT_CURRENCY,
        provider="searchapi",
        raw=data,
    )

def fetch_rate(hotel: str, check_in: dt.date, check_out: dt.date) -> RateResult:
    if PROVIDER == "serpapi":
        return fetch_rate_serpapi(hotel, check_in, check_out)
    return fetch_rate_searchapi(hotel, check_in, check_out)


# -----------------------------
# EVENTS (FORWARD-LOOKING)
# -----------------------------

def fetch_barclays_events(next_days: int = 14) -> List[Tuple[dt.date, str]]:
    """
    Lightweight scraper of Barclays Center 'Events' page.
    Used only for forward-looking demand flags.
    Source pages: https://www.barclayscenter.com/events  [oai_citation:1‡Barclays Center](https://www.barclayscenter.com/events?utm_source=chatgpt.com)
    """
    url = "https://www.barclayscenter.com/events"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n")

    # Heuristic parsing: Barclays markup changes; instead of brittle selectors,
    # try to capture "Month DD" + nearby title lines.
    # We'll scan lines and look for patterns like "January 01" etc.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    events: List[Tuple[dt.date, str]] = []

    today = ny_today()
    end = today + dt.timedelta(days=next_days)

    # Try parse lines like: "January 01" then next non-empty line is title
    month_map = {m: i for i, m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"], start=1
    )}

    i = 0
    while i < len(lines) - 1:
        parts = lines[i].split()
        if len(parts) == 2 and parts[0] in month_map and parts[1].isdigit():
            month = month_map[parts[0]]
            day = int(parts[1])
            year = today.year
            # handle year rollover (Dec -> Jan)
            if today.month == 12 and month == 1:
                year += 1
            try:
                d = dt.date(year, month, day)
            except ValueError:
                i += 1
                continue

            if today <= d <= end:
                title = lines[i + 1]
                # filter out noise
                if len(title) > 3 and "Buy Tickets" not in title and "Info" not in title:
                    events.append((d, title))
        i += 1

    # De-dupe
    uniq = {}
    for d, t in events:
        uniq[(d, t)] = True
    return list(uniq.keys())

def load_manual_events() -> List[Tuple[dt.date, str]]:
    """
    Optional: put a file named events.json in repo to add/override:
    [
      {"date":"2025-12-31","title":"NYE Brooklyn demand spike"},
      ...
    ]
    """
    path = "events.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        out = []
        for it in arr:
            d = dt.date.fromisoformat(it["date"])
            t = str(it["title"])
            out.append((d, t))
        return out
    except Exception:
        return []

def build_event_flags() -> Dict[dt.date, List[str]]:
    # Barclays + manual file
    ev = fetch_barclays_events(next_days=14)
    ev += load_manual_events()
    flags: Dict[dt.date, List[str]] = {}
    for d, title in ev:
        flags.setdefault(d, []).append(title)
    return flags


# -----------------------------
# ANALYSIS + RECOMMENDATIONS (FORWARD ONLY)
# -----------------------------

def compute_bucket(rates: List[RateResult]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = [r.entry_rate for r in rates if r.entry_rate is not None]
    if not vals:
        return None, None, None
    avg = sum(vals) / len(vals)
    return min(vals), avg, max(vals)

def rank_table(rates: List[RateResult]) -> List[RateResult]:
    # Available first by price, then unavailable
    avail = [r for r in rates if r.entry_rate is not None]
    unavail = [r for r in rates if r.entry_rate is None]
    avail.sort(key=lambda x: x.entry_rate)
    unavail.sort(key=lambda x: x.hotel)
    return avail + unavail

def recommend_for_date(
    date_label: str,
    check_in: dt.date,
    bucket_rates: List[RateResult],
    event_flags: Dict[dt.date, List[str]]
) -> Dict[str, str]:
    """
    Returns dict with recommendation + opportunities, forward-looking only.
    """
    # Extract your rate and comp rates
    your = next((r for r in bucket_rates if r.hotel == YOUR_HOTEL), None)
    comps = [r for r in bucket_rates if r.hotel != YOUR_HOTEL]

    min_rate, comp_avg, comp_max = compute_bucket(comps)
    event_titles = event_flags.get(check_in, [])
    has_event = len(event_titles) > 0
    weekend = is_weekend(check_in)

    if your is None:
        return {
            "label": date_label,
            "date": check_in.isoformat(),
            "recommendation": "HOLD",
            "target_rate": "N/A",
            "reason": "Your hotel rate was not found in the data pull.",
            "opportunities": "Check mapping/name accuracy for Bay Ridge Hotel query.",
            "events": "; ".join(event_titles) if event_titles else "None detected",
        }

    # If your rate not found, still provide comp context
    if your.entry_rate is None:
        return {
            "label": date_label,
            "date": check_in.isoformat(),
            "recommendation": "HOLD",
            "target_rate": "N/A",
            "reason": "Bay Ridge entry rate not found / not available in current pull.",
            "opportunities": "Verify Bay Ridge listing visibility on Google Hotels for this date.",
            "events": "; ".join(event_titles) if event_titles else "None detected",
        }

    # If comp avg missing, cannot compute parity
    if comp_avg is None:
        return {
            "label": date_label,
            "date": check_in.isoformat(),
            "recommendation": "HOLD",
            "target_rate": fmt_money(your.entry_rate),
            "reason": "Insufficient comp-set data to compute market positioning.",
            "opportunities": "Add more comps or confirm comp-set visibility.",
            "events": "; ".join(event_titles) if event_titles else "None detected",
        }

    # Position vs comp avg
    diff_pct = (comp_avg - your.entry_rate) / comp_avg  # positive means you're below market
    below_market = diff_pct > 0

    # Base action
    action = "HOLD"
    step = 0

    if below_market and diff_pct >= RAISE_IF_UNDER_COMP_AVG_PCT:
        action = "RAISE"
        step = BASE_RAISE_STEP
    elif (not below_market) and (pct(your.entry_rate, comp_avg) >= DROP_IF_OVER_COMP_AVG_PCT):
        action = "DROP"
        step = BASE_DROP_STEP

    # Bonuses (forward looking)
    if action == "RAISE" and weekend:
        step += WEEKEND_RAISE_BONUS
    if action == "RAISE" and has_event:
        step += EVENT_RAISE_BONUS

    # Target
    target = int(round(your.entry_rate))
    if action == "RAISE":
        target += step
    elif action == "DROP":
        target -= step

    target = clamp(target, MIN_RATE, MAX_RATE)

    # Opportunities text
    opp = []
    if below_market:
        opp.append(f"Bay Ridge is positioned BELOW comp average by {diff_pct*100:.1f}% for {date_label}.")
        if weekend:
            opp.append("Weekend bias detected (Fri/Sat).")
        if has_event:
            opp.append("Event-driven demand detected (Brooklyn venue/event).")
    else:
        opp.append(f"Bay Ridge is positioned ABOVE comp average by {pct(your.entry_rate, comp_avg)*100:.1f}% for {date_label}.")

    reason_bits = []
    reason_bits.append(f"Comp avg entry rate: {fmt_money(comp_avg)}; Bay Ridge entry: {fmt_money(your.entry_rate)}.")
    if weekend:
        reason_bits.append("Weekend compression bias.")
    if has_event:
        reason_bits.append("City/venue event demand driver.")

    rec_line = "HOLD"
    if action == "RAISE":
        rec_line = f"🔼 RAISE to {fmt_money(target)}"
    elif action == "DROP":
        rec_line = f"🔽 DROP to {fmt_money(target)}"

    return {
        "label": date_label,
        "date": check_in.isoformat(),
        "recommendation": rec_line,
        "target_rate": fmt_money(target),
        "reason": " ".join(reason_bits),
        "opportunities": " ".join(opp),
        "events": "; ".join(event_titles) if event_titles else "None detected",
    }


# -----------------------------
# EMAIL RENDERING
# -----------------------------

def build_email_body(all_rates: Dict[int, List[RateResult]], event_flags: Dict[dt.date, List[str]]) -> Tuple[str, str]:
    """
    Returns (subject, plain_text_body)
    """
    today = ny_today()
    now = dt.datetime.now(tz=NY_TZ)

    subject = f"Bay Ridge Hotel — Forward Revenue Intelligence (7:00 AM ET) | {today.isoformat()}"

    lines: List[str] = []
    lines.append(f"Bay Ridge Hotel — Forward Revenue Intelligence Brief")
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %I:%M %p')} ET")
    lines.append(f"Provider: {PROVIDER}")
    lines.append("")

    # Executive recommendations per bucket
    recs = []
    for off in DATE_OFFSETS:
        check_in = today + dt.timedelta(days=off)
        label = "Tonight (D0)" if off == 0 else (f"D+{off}")
        bucket = all_rates[off]
        rec = recommend_for_date(label, check_in, bucket, event_flags)
        recs.append(rec)

    lines.append("EXECUTIVE SUMMARY (Forward-looking only)")
    for r in recs:
        lines.append(f"- {r['label']} ({r['date']}): {r['recommendation']}")
    lines.append("")

    # Detail per bucket
    for off in DATE_OFFSETS:
        check_in = today + dt.timedelta(days=off)
        label = "Tonight (D0)" if off == 0 else (f"Check-in D+{off}")
        check_out = check_in + dt.timedelta(days=1)

        lines.append("=" * 72)
        lines.append(f"{label}: {check_in.isoformat()} → {check_out.isoformat()} (1 night)")
        lines.append("=" * 72)

        bucket = all_rates[off]
        ranked = rank_table(bucket)

        # Entry rate ranking table
        lines.append("ENTRY RATE RANKING (Cheapest public entry rate available)")
        rank_num = 1
        for rr in ranked:
            price = fmt_money(rr.entry_rate)
            note = ""
            if rr.hotel == YOUR_HOTEL:
                note = "  <-- YOUR HOTEL"
            if rr.entry_rate is None:
                note = (note + " (not available / not found)").strip()
            lines.append(f"{rank_num}. {rr.hotel} — {price}{note}")
            rank_num += 1
        lines.append("")

        # Market stats
        comps = [r for r in bucket if r.hotel != YOUR_HOTEL]
        _, comp_avg, _ = compute_bucket(comps)
        your = next((r for r in bucket if r.hotel == YOUR_HOTEL), None)

        if comp_avg is not None and your and your.entry_rate is not None:
            diff_pct = (comp_avg - your.entry_rate) / comp_avg
            pos = "BELOW" if diff_pct > 0 else "ABOVE"
            lines.append(f"MARKET POSITIONING")
            lines.append(f"- Comp-set avg entry rate: {fmt_money(comp_avg)}")
            lines.append(f"- Bay Ridge entry rate:     {fmt_money(your.entry_rate)}")
            lines.append(f"- Position: {pos} market by {abs(diff_pct)*100:.1f}%")
            lines.append("")
        else:
            lines.append("MARKET POSITIONING")
            lines.append("- Not enough data to compute positioning (missing rates).")
            lines.append("")

        # Events
        ev = event_flags.get(check_in, [])
        lines.append("BROOKLYN EVENTS / DEMAND DRIVERS (forward-looking)")
        if ev:
            for t in ev[:8]:
                lines.append(f"- {t}")
        else:
            lines.append("- None detected for this date window from Barclays feed + manual events.json.")
        lines.append("")

        # Recommendation + opportunities
        rec = recommend_for_date("Tonight (D0)" if off == 0 else f"D+{off}", check_in, bucket, event_flags)
        lines.append("OPPORTUNITIES")
        lines.append(f"- {rec['opportunities']}")
        lines.append("")
        lines.append("RECOMMENDATION")
        lines.append(f"- {rec['recommendation']}")
        lines.append(f"- Rationale: {rec['reason']}")
        lines.append("")

    # Quick actions (tool-agnostic)
    lines.append("=" * 72)
    lines.append("10-MINUTE EXECUTION CHECKLIST (Tool-agnostic)")
    lines.append("- Confirm Bay Ridge entry rate is positioned intentionally vs comp-set for Tonight / D+7 / D+14.")
    lines.append("- If weekend or event nights are flagged, pre-load increases early (don’t wait for pickup).")
    lines.append("- Apply changes in your RMS/PMS/channel manager within guardrails (min/max).")
    lines.append("")

    return subject, "\n".join(lines)


def send_email(subject: str, body_text: str) -> None:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and EMAIL_FROM):
        raise RuntimeError("Missing SMTP config. Set SMTP_HOST/SMTP_USER/SMTP_PASS.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    part = MIMEText(body_text, "plain")
    msg.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
        server.starttls(context=context)
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())


# -----------------------------
# MAIN
# -----------------------------

def main():
    today = ny_today()

    # Build event flags (forward-looking)
    event_flags = build_event_flags()

    # Pull rates for each date offset
    all_rates: Dict[int, List[RateResult]] = {}
    for off in DATE_OFFSETS:
        check_in = today + dt.timedelta(days=off)
        check_out = check_in + dt.timedelta(days=1)

        bucket: List[RateResult] = []
        for h in HOTELS:
            rr = fetch_rate(h, check_in, check_out)
            bucket.append(rr)

        all_rates[off] = bucket

    subject, body = build_email_body(all_rates, event_flags)
    send_email(subject, body)
    print("Email sent:", subject)


if __name__ == "__main__":
    main()
