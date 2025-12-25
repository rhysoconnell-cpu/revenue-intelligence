import os, json, ssl, smtplib, datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from dateutil import tz
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

NY_TZ = tz.gettz("America/New_York")

# ---------------- ENV ----------------
SERP_API_KEY = os.getenv("SERP_API_KEY", "").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
EMAIL_TO  = os.getenv("EMAIL_TO", "Rhys.oconnell@yourplacehotel.com").strip()

DEBUG = os.getenv("DEBUG", "0") == "1"

CURRENCY = "USD"
ADULTS = 2
ROOMS = 1

# Forward-looking windows
OFFSETS = [0, 7, 14]

# Recommendation behavior (simple + explainable)
MIN_RATE = 99
MAX_RATE = 399
RAISE_IF_BELOW_AVG_PCT = 0.08
DROP_IF_ABOVE_AVG_PCT  = 0.10
BASE_STEP = 10
WEEKEND_BONUS = 10
EVENT_BONUS = 10


# ---------------- MODELS ----------------
@dataclass
class Hotel:
    key: str
    display_name: str
    query: str
    property_token: str

@dataclass
class Quote:
    hotel_key: str
    display_name: str
    check_in: dt.date
    check_out: dt.date

    price: Optional[float]
    ota_name: str = "Unknown"
    ota_url: str = ""
    matched_name: str = ""
    matched_address: str = ""

    @property
    def ok(self) -> bool:
        return self.price is not None


# ---------------- HELPERS ----------------
def ny_now() -> dt.datetime:
    return dt.datetime.now(tz=NY_TZ)

def ny_today() -> dt.date:
    return ny_now().date()

def money(x: Optional[float]) -> str:
    return "N/A" if x is None else f"${x:,.0f}"

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def is_weekend(d: dt.date) -> bool:
    return d.weekday() in (4, 5)  # Fri/Sat

def avg(nums: List[float]) -> Optional[float]:
    return None if not nums else sum(nums) / len(nums)

def short_domain(url: str) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        return host
    except Exception:
        return ""

def serp_get(params: Dict[str, Any]) -> Dict[str, Any]:
    if not SERP_API_KEY:
        raise RuntimeError("Missing SERP_API_KEY (GitHub Secret).")
    params = dict(params)
    params["api_key"] = SERP_API_KEY
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
    r.raise_for_status()
    return r.json()


# ---------------- CONFIG ----------------
def load_config() -> Tuple[str, str, List[Hotel]]:
    with open("hotels.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    portfolio = str(cfg.get("portfolio_name") or "Revenue Intel")
    location = str(cfg.get("location") or "Brooklyn, New York, United States")

    hotels: List[Hotel] = []
    for h in cfg.get("hotels", []):
        hotels.append(Hotel(
            key=str(h.get("key")),
            display_name=str(h.get("display_name")),
            query=str(h.get("query")),
            property_token=str(h.get("property_token") or ""),
        ))

    return portfolio, location, hotels


# ---------------- EVENTS (BROOKLYN SIGNALS) ----------------
def fetch_brooklyn_events() -> List[Dict[str, str]]:
    """
    Forward-looking Brooklyn events via SerpApi google_events.
    """
    data = serp_get({
        "engine": "google_events",
        "q": "Brooklyn events",
        "location": "Brooklyn, New York, United States",
        "hl": "en",
        "gl": "us",
    })
    events = data.get("events_results") or data.get("events") or []
    out: List[Dict[str, str]] = []
    if isinstance(events, list):
        for e in events[:12]:
            if isinstance(e, dict):
                out.append({
                    "title": str(e.get("title") or ""),
                    "date": str(e.get("date") or e.get("start_date") or ""),
                    "venue": str(e.get("venue") or ""),
                    "link": str(e.get("link") or ""),
                })
    return out


# ---------------- RATE FETCH (TOKEN-LOCKED) ----------------
def fetch_property_quote(h: Hotel, location: str, check_in: dt.date, check_out: dt.date) -> Quote:
    """
    Uses google_hotels_property_details (token-locked).
    Picks the lowest OTA offer if present, and includes OTA name + URL.
    """
    q = Quote(
        hotel_key=h.key,
        display_name=h.display_name,
        check_in=check_in,
        check_out=check_out,
        price=None,
    )

    if not h.property_token:
        return q

    data = serp_get({
        "engine": "google_hotels_property_details",
        "property_token": h.property_token,
        "check_in_date": check_in.isoformat(),
        "check_out_date": check_out.isoformat(),
        "currency": CURRENCY,
        "adults": ADULTS,
        "rooms": ROOMS,
        "location": location,
        "hl": "en",
        "gl": "us",
    })

    # identity fields (trust / audit trail)
    q.matched_name = str(data.get("property_name") or data.get("title") or data.get("name") or "")
    q.matched_address = str(data.get("address") or "")

    candidates: List[Tuple[float, str, str]] = []

    def consider(rate: Any, name: Any, url: Any):
        if isinstance(rate, (int, float)) and rate > 0:
            candidates.append((float(rate), str(name or "Unknown"), str(url or "")))

    # Common patterns
    prices = data.get("prices")
    if isinstance(prices, list):
        for p in prices:
            if isinstance(p, dict):
                consider(p.get("extracted_rate") or p.get("rate"), p.get("name") or p.get("source"), p.get("link") or p.get("url"))

    offers = data.get("offers")
    if isinstance(offers, list):
        for o in offers:
            if isinstance(o, dict):
                consider(o.get("extracted_rate") or o.get("rate"), o.get("name") or o.get("source"), o.get("link") or o.get("url"))

    # Some payloads nest "rate_per_night"
    rpn = data.get("rate_per_night")
    if isinstance(rpn, dict):
        consider(rpn.get("extracted_lowest") or rpn.get("lowest"), "Google Hotels (base)", data.get("link"))

    if candidates:
        candidates.sort(key=lambda t: t[0])
        best = candidates[0]
        q.price, q.ota_name, q.ota_url = best[0], best[1], best[2]

    if DEBUG:
        print("\n==== DEBUG QUOTE ====")
        print("Hotel:", h.display_name, "| Token:", h.property_token)
        print("Dates:", check_in.isoformat(), "->", check_out.isoformat())
        print("Matched:", q.matched_name, "|", q.matched_address)
        print("Selected:", q.price, q.ota_name, q.ota_url)
        print("=====================\n")

    return q


# ---------------- RECOMMENDATION ----------------
def recommend(quotes: List[Quote], your_key: str, check_in: dt.date, event_boost: bool) -> Tuple[str, str]:
    your = next((q for q in quotes if q.hotel_key == your_key), None)
    comps = [q for q in quotes if q.hotel_key != your_key and q.price is not None]

    comp_avg = avg([float(q.price) for q in comps if q.price is not None])

    if not your or your.price is None or comp_avg is None:
        return ("HOLD (insufficient data)", "Missing rates or comps — check tokens / availability.")

    your_rate = float(your.price)
    diff_pct = (comp_avg - your_rate) / comp_avg

    weekend = is_weekend(check_in)
    step = BASE_STEP + (WEEKEND_BONUS if weekend else 0) + (EVENT_BONUS if event_boost else 0)

    if diff_pct >= RAISE_IF_BELOW_AVG_PCT:
        target = clamp(int(round(your_rate + step)), MIN_RATE, MAX_RATE)
        return (f"🔼 RAISE to {money(target)}", f"Below comp avg by {diff_pct*100:.1f}% | comp avg {money(comp_avg)} | weekend={weekend} event={event_boost}")

    if (your_rate - comp_avg) / comp_avg >= DROP_IF_ABOVE_AVG_PCT:
        target = clamp(int(round(your_rate - BASE_STEP)), MIN_RATE, MAX_RATE)
        return (f"🔽 DROP to {money(target)}", f"Above comp avg by {((your_rate-comp_avg)/comp_avg)*100:.1f}% | comp avg {money(comp_avg)}")

    return ("HOLD", f"Near comp avg | comp avg {money(comp_avg)} | weekend={weekend} event={event_boost}")


# ---------------- DUETTO-STYLE EMAIL BUILDER ----------------
def build_email(portfolio: str, windows: Dict[int, List[Quote]], hotels: List[Hotel], your_key: str, events: List[Dict[str, str]]) -> Tuple[str, str, str]:
    today = ny_today()
    now = ny_now()

    subject = f"{portfolio} — Revenue Intelligence (D0 / D+7 / D+14) | {today.isoformat()}"

    def row_source(q: Quote) -> str:
        dom = short_domain(q.ota_url)
        if dom:
            return f"{q.ota_name} ({dom})"
        return q.ota_name or "Unknown"

    def open_link(q: Quote) -> str:
        if not q.ota_url:
            return ""
        return f"<a href='{q.ota_url}' style='color:#0b57d0;text-decoration:none;font-size:12px;'>(open)</a>"

    def market_stats(quotes: List[Quote]) -> Dict[str, Optional[float]]:
        vals = [float(x.price) for x in quotes if x.price is not None]
        if not vals:
            return {"min": None, "max": None, "avg": None}
        return {"min": min(vals), "max": max(vals), "avg": sum(vals)/len(vals)}

    def your_quote(quotes: List[Quote]) -> Optional[Quote]:
        return next((q for q in quotes if q.hotel_key == your_key), None)

    def rank_position(quotes: List[Quote]) -> Tuple[Optional[int], int]:
        avail = [q for q in quotes if q.price is not None]
        avail_sorted = sorted(avail, key=lambda x: x.price)
        total = len(avail_sorted)
        for i, q in enumerate(avail_sorted, start=1):
            if q.hotel_key == your_key:
                return i, total
        return None, total

    # Executive summary rows
    exec_rows_html = []
    exec_rows_txt = []

    for off in OFFSETS:
        d = today + dt.timedelta(days=off)
        label = "Tonight (D0)" if off == 0 else f"D+{off}"
        quotes = windows[off]
        stats = market_stats(quotes)
        pos, total = rank_position(quotes)
        yq = your_quote(quotes)

        # simple “event boost” heuristic: if the date string appears in an event date text
        event_boost = any(d.isoformat() in (e.get("date") or "") for e in events)
        rec, why = recommend(quotes, your_key, d, event_boost)

        your_rate = money(yq.price) if (yq and yq.price is not None) else "N/A"
        mkt = money(stats["avg"]) if stats["avg"] is not None else "N/A"
        pos_txt = f"{pos}/{total}" if pos is not None else "N/A"

        exec_rows_txt.append(f"- {label} ({d.isoformat()}): Your {your_rate} | Market avg {mkt} | Position {pos_txt} | {rec}")

        exec_rows_html.append(
            f"<tr>"
            f"<td style='padding:10px;border-bottom:1px solid #eee;'><b>{label}</b><div style='color:#666;font-size:12px;'>{d.isoformat()}</div></td>"
            f"<td style='padding:10px;border-bottom:1px solid #eee;text-align:right;'><b>{your_rate}</b></td>"
            f"<td style='padding:10px;border-bottom:1px solid #eee;text-align:right;color:#333;'>{mkt}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #eee;text-align:center;color:#333;'>{pos_txt}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #eee;'><b>{rec}</b><div style='color:#666;font-size:12px;'>{why}</div></td>"
            f"</tr>"
        )

    # Plain text fallback
    text = []
    text.append(f"{portfolio} — Revenue Intelligence")
    text.append(f"Generated: {now.strftime('%Y-%m-%d %I:%M %p')} ET")
    text.append("Windows: Tonight / D+7 / D+14 (forward-looking only)")
    text.append("")
    text.append("EXECUTIVE SUMMARY")
    text.extend(exec_rows_txt)
    text.append("")
    text.append("Notes: Rates are token-locked (property_token) and include OTA source when available.")
    text = "\n".join(text)

    # HTML (Duetto-style)
    html = []
    html.append("<div style='font-family: Arial, Helvetica, sans-serif; color:#111; max-width:980px;'>")

    # Header
    html.append(
        "<div style='padding:18px 18px 10px 18px; border:1px solid #eee; border-radius:14px;'>"
        f"<div style='font-size:18px;font-weight:700;letter-spacing:0.2px;'>{portfolio}</div>"
        f"<div style='margin-top:6px;color:#666;font-size:12px;'>Generated: {now.strftime('%Y-%m-%d %I:%M %p')} ET • Forward-looking only • Token-locked identity</div>"
        "</div>"
    )

    # Executive Summary card
    html.append("<div style='height:12px;'></div>")
    html.append("<div style='padding:16px 18px; border:1px solid #eee; border-radius:14px;'>")
    html.append("<div style='font-size:14px;font-weight:700;margin-bottom:10px;'>Executive Summary</div>")
    html.append(
        "<table style='border-collapse:collapse;width:100%;font-size:13px;'>"
        "<tr>"
        "<th style='text-align:left;padding:10px;border-bottom:1px solid #ddd;color:#444;font-weight:700;'>Window</th>"
        "<th style='text-align:right;padding:10px;border-bottom:1px solid #ddd;color:#444;font-weight:700;'>Your Rate</th>"
        "<th style='text-align:right;padding:10px;border-bottom:1px solid #ddd;color:#444;font-weight:700;'>Market Avg</th>"
        "<th style='text-align:center;padding:10px;border-bottom:1px solid #ddd;color:#444;font-weight:700;'>Position</th>"
        "<th style='text-align:left;padding:10px;border-bottom:1px solid #ddd;color:#444;font-weight:700;'>Recommendation</th>"
        "</tr>"
        + "".join(exec_rows_html) +
        "</table>"
    )
    html.append("</div>")

    # Demand signals
    html.append("<div style='height:12px;'></div>")
    html.append("<div style='padding:16px 18px; border:1px solid #eee; border-radius:14px;'>")
    html.append("<div style='font-size:14px;font-weight:700;margin-bottom:8px;'>Brooklyn Demand Signals</div>")
    if events:
        html.append("<ul style='margin:0;padding-left:18px;color:#333;font-size:13px;'>")
        for e in events[:8]:
            t = (e.get("title") or "").strip()
            when = (e.get("date") or "").strip()
            v = (e.get("venue") or "").strip()
            html.append(f"<li style='margin:6px 0;'><b>{t}</b> <span style='color:#666;font-size:12px;'>— {when} {v}</span></li>")
        html.append("</ul>")
    else:
        html.append("<div style='color:#666;font-size:12px;'>No events returned.</div>")
    html.append("</div>")

    # Per-window detail cards
    for off in OFFSETS:
        d = today + dt.timedelta(days=off)
        label = "Tonight (D0)" if off == 0 else f"D+{off}"
        quotes = windows[off]

        stats = market_stats(quotes)
        pos, total = rank_position(quotes)
        yq = your_quote(quotes)

        event_boost = any(d.isoformat() in (e.get("date") or "") for e in events)
        rec, why = recommend(quotes, your_key, d, event_boost)

        html.append("<div style='height:12px;'></div>")
        html.append("<div style='padding:16px 18px; border:1px solid #eee; border-radius:14px;'>")

        html.append(
            f"<div style='display:flex;justify-content:space-between;gap:12px;align-items:flex-end;'>"
            f"<div><div style='font-size:14px;font-weight:700;'>{label}</div>"
            f"<div style='color:#666;font-size:12px;margin-top:3px;'>Check-in {d.isoformat()} • 1 night</div></div>"
            f"<div style='text-align:right;'>"
            f"<div style='font-size:12px;color:#666;'>Market Range</div>"
            f"<div style='font-size:13px;font-weight:700;'>{money(stats['min'])} – {money(stats['max'])}</div>"
            f"</div>"
            f"</div>"
        )

        your_rate = money(yq.price) if (yq and yq.price is not None) else "N/A"
        pos_txt = f"{pos}/{total}" if pos is not None else "N/A"

        # Action bar
        html.append(
            "<div style='margin-top:12px;padding:12px 14px;border-radius:12px;background:#111;color:#fff;'>"
            f"<div style='display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;'>"
            f"<div><div style='font-size:11px;opacity:.75;'>YOUR ENTRY RATE</div><div style='font-size:16px;font-weight:800;'>{your_rate}</div></div>"
            f"<div><div style='font-size:11px;opacity:.75;'>MARKET POSITION</div><div style='font-size:16px;font-weight:800;'>{pos_txt}</div></div>"
            f"<div style='flex:1;min-width:260px;'><div style='font-size:11px;opacity:.75;'>RECOMMENDED ACTION</div>"
            f"<div style='font-size:16px;font-weight:800;'>{rec}</div>"
            f"<div style='font-size:12px;opacity:.75;margin-top:4px;'>{why}</div>"
            f"</div>"
            f"</div>"
            "</div>"
        )

        # Comp table
        html.append("<div style='margin-top:14px;'>")
        html.append("<div style='font-size:13px;font-weight:700;margin-bottom:8px;'>Comp Set — Entry Rate (with OTA source)</div>")

        avail = [q for q in quotes if q.price is not None]
        unavail = [q for q in quotes if q.price is None]
        avail_sorted = sorted(avail, key=lambda x: x.price)

        html.append("<table style='border-collapse:collapse;width:100%;font-size:13px;'>")
        html.append("<tr>"
                    "<th style='text-align:left;padding:10px;border-bottom:1px solid #ddd;color:#444;font-weight:700;'>Hotel</th>"
                    "<th style='text-align:right;padding:10px;border-bottom:1px solid #ddd;color:#444;font-weight:700;'>Entry Rate</th>"
                    "<th style='text-align:left;padding:10px;border-bottom:1px solid #ddd;color:#444;font-weight:700;'>Source</th>"
                    "<th style='text-align:left;padding:10px;border-bottom:1px solid #ddd;color:#444;font-weight:700;'>Matched Listing</th>"
                    "</tr>")

        def row(q: Quote, highlight: bool) -> str:
            bg = "background:#f6f6f6;" if highlight else ""
            name = q.display_name
            rate = money(q.price)
            src = row_source(q)
            matched = (q.matched_name or "").strip()
            if q.matched_address:
                matched = f"{matched} — {q.matched_address}".strip(" —")
            you_tag = " <span style='color:#666;font-size:12px;'>(you)</span>" if highlight else ""
            return (
                f"<tr style='{bg}'>"
                f"<td style='padding:10px;border-bottom:1px solid #f0f0f0;'><b>{name}</b>{you_tag}</td>"
                f"<td style='padding:10px;border-bottom:1px solid #f0f0f0;text-align:right;'><b>{rate}</b></td>"
                f"<td style='padding:10px;border-bottom:1px solid #f0f0f0;'>{src} {open_link(q)}</td>"
                f"<td style='padding:10px;border-bottom:1px solid #f0f0f0;color:#333;'>{matched}</td>"
                f"</tr>"
            )

        for q in avail_sorted:
            html.append(row(q, q.hotel_key == your_key))

        for q in unavail:
            html.append(
                "<tr>"
                f"<td style='padding:10px;border-bottom:1px solid #f0f0f0;'><b>{q.display_name}</b></td>"
                f"<td style='padding:10px;border-bottom:1px solid #f0f0f0;text-align:right;'>N/A</td>"
                f"<td style='padding:10px;border-bottom:1px solid #f0f0f0;color:#666;'>not found</td>"
                f"<td style='padding:10px;border-bottom:1px solid #f0f0f0;color:#666;'>{(q.matched_name or '')}</td>"
                "</tr>"
            )

        html.append("</table>")
        html.append("</div>")  # comp wrapper
        html.append("</div>")  # card

    # Footer
    html.append("<div style='height:12px;'></div>")
    html.append("<div style='color:#666;font-size:11px;padding:0 4px 8px 4px;'>"
                "Notes: Prices are pulled using token-locked identity (property_token). "
                "We show OTA source when available; otherwise we fall back to Google Hotels base. "
                "Forward-looking only."
                "</div>")

    html.append("</div>")  # container

    return subject, text, "".join(html)


# ---------------- EMAIL SEND ----------------
def send_email(subject: str, html: str, text: str) -> None:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        raise RuntimeError("Missing SMTP_* secrets (SMTP_HOST/PORT/USER/PASS).")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO

    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=45) as s:
        s.starttls(context=ctx)
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_USER, [EMAIL_TO], msg.as_string())


# ---------------- MAIN ----------------
def main():
    portfolio, location, hotels = load_config()

    your = next((h for h in hotels if h.key == "bay_ridge"), None)
    if not your:
        raise RuntimeError("Missing bay_ridge in hotels.json")

    # Pull events (signals)
    events = fetch_brooklyn_events()

    # Pull windows
    today = ny_today()
    windows: Dict[int, List[Quote]] = {}

    for off in OFFSETS:
        ci = today + dt.timedelta(days=off)
        co = ci + dt.timedelta(days=1)
        bucket: List[Quote] = []
        for h in hotels:
            bucket.append(fetch_property_quote(h, location, ci, co))
        windows[off] = bucket

    subject, text, html = build_email(portfolio, windows, hotels, your.key, events)
    send_email(subject, html, text)
    print("Sent:", subject)


if __name__ == "__main__":
    main()
