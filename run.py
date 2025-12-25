import os
import json
import time
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ----------------------------
# CONFIG (ENV VARS)
# ----------------------------
SERP_API_KEY = os.getenv("SERP_API_KEY", "").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()

EMAIL_TO = os.getenv("EMAIL_TO", "").strip()
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER).strip()

DEBUG = os.getenv("DEBUG", "0").strip() == "1"

DEFAULT_CURRENCY = os.getenv("CURRENCY", "USD").strip()
DEFAULT_ADULTS = int(os.getenv("ADULTS", "2"))
DEFAULT_ROOMS = int(os.getenv("ROOMS", "1"))
HL = os.getenv("HL", "en").strip()
GL = os.getenv("GL", "us").strip()

HOTELS_JSON_PATH = os.getenv("HOTELS_JSON", "hotels.json")

REQUEST_SLEEP_SEC = float(os.getenv("REQUEST_SLEEP_SEC", "0.6"))

# ----------------------------
# DATA MODELS
# ----------------------------
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
    price: Optional[float] = None
    currency: str = DEFAULT_CURRENCY
    source: Optional[str] = None
    link: Optional[str] = None
    raw: Optional[dict] = None
    error: Optional[str] = None

# ----------------------------
# TIME (NY / ET)
# ----------------------------
def ny_now() -> dt.datetime:
    # Simple ET handling without external libs:
    # GitHub runners use UTC; we keep dates by ET offset.
    # Current use-case: you’re on EST (-5). If you want DST precision later, we’ll add zoneinfo.
    return dt.datetime.utcnow() - dt.timedelta(hours=5)

def ny_today() -> dt.date:
    return ny_now().date()

# ----------------------------
# SERPAPI
# ----------------------------
def serp_get(params: Dict[str, Any]) -> Dict[str, Any]:
    if not SERP_API_KEY:
        raise RuntimeError("Missing SERP_API_KEY")

    url = "https://serpapi.com/search.json"
    params = dict(params)
    params["api_key"] = SERP_API_KEY

    if DEBUG:
        safe = dict(params)
        safe["api_key"] = "***"
        print("SERP GET:", safe)

    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        # include body for debugging
        raise RuntimeError(f"SerpApi error {r.status_code}: {r.text[:800]}")
    return r.json()

# ----------------------------
# URL SHORTENER (FREE)
# ----------------------------
def short_url(long_url: str) -> str:
    if not long_url:
        return ""
    try:
        api = "https://tinyurl.com/api-create.php"
        r = requests.get(api, params={"url": long_url}, timeout=20)
        if r.status_code == 200 and r.text.startswith("http"):
            return r.text.strip()
    except Exception:
        pass
    return long_url

# ----------------------------
# LOAD HOTELS
# ----------------------------
def load_hotels() -> Tuple[str, str, List[Hotel]]:
    with open(HOTELS_JSON_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    portfolio = cfg.get("portfolio_name", "Revenue Intel")
    location = cfg.get("location", "Brooklyn, NY")

    hotels = []
    for h in cfg["hotels"]:
        hotels.append(
            Hotel(
                key=h["key"],
                display_name=h["display_name"],
                query=h.get("query", h["display_name"]),
                property_token=h.get("property_token", "").strip(),
            )
        )
    return portfolio, location, hotels

# ----------------------------
# FETCH QUOTE (TOKEN-LOCKED)
# ----------------------------
def pick_lowest_offer(data: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Prefer featured_prices (contains OTA 'source' + link).
    Fallback to rate_per_night.extracted_lowest.
    """
    # 1) Featured prices (best: has OTA name + link)
    featured = data.get("featured_prices") or []
    best = None  # (price, source, link)

    for item in featured:
        # Different responses may call it "extracted_price" or embed in strings.
        # We defensively try common fields.
        price = None
        for k in ("extracted_price", "extracted_lowest", "extracted_rate"):
            if isinstance(item.get(k), (int, float)):
                price = float(item[k])
                break

        # Sometimes it’s nested
        if price is None and isinstance(item.get("rate_per_night"), dict):
            v = item["rate_per_night"].get("extracted_lowest")
            if isinstance(v, (int, float)):
                price = float(v)

        # As last resort, skip if no numeric price
        if price is None:
            continue

        source = item.get("source") or item.get("name")
        link = item.get("link")

        if best is None or price < best[0]:
            best = (price, source, link)

    if best:
        return best[0], best[1], best[2]

    # 2) Fallback: rate_per_night (may not have OTA)
    rpn = data.get("rate_per_night") or {}
    if isinstance(rpn.get("extracted_lowest"), (int, float)):
        return float(rpn["extracted_lowest"]), data.get("name"), data.get("link")

    return None, None, None

def fetch_property_quote(h: Hotel, check_in: dt.date, check_out: dt.date) -> Quote:
    q = Quote(
        hotel_key=h.key,
        display_name=h.display_name,
        check_in=check_in,
        check_out=check_out,
        currency=DEFAULT_CURRENCY,
    )

    if not h.property_token:
        q.error = "Missing property_token"
        return q

    # IMPORTANT:
    # For engine=google_hotels_property_details, DO NOT send `location=` (can cause 400).
    # Use token + q + dates.
    params = {
        "engine": "google_hotels_property_details",
        "q": h.query,
        "property_token": h.property_token,
        "check_in_date": check_in.isoformat(),
        "check_out_date": check_out.isoformat(),
        "adults": DEFAULT_ADULTS,
        "rooms": DEFAULT_ROOMS,
        "currency": DEFAULT_CURRENCY,
        "hl": HL,
        "gl": GL,
        "no_cache": "true",  # force fresh pull (SerpApi caches by default up to ~1h)
    }

    try:
        data = serp_get(params)
        price, source, link = pick_lowest_offer(data)

        q.price = price
        q.source = source
        q.link = short_url(link) if link else None
        q.raw = data if DEBUG else None

        if DEBUG:
            print("MATCHED:", h.display_name)
            print("TOKEN:", h.property_token)
            print("CHECKIN/CHECKOUT:", check_in, check_out)
            print("PRICE:", q.price, q.currency)
            print("SOURCE:", q.source)
            print("LINK:", q.link)

    except Exception as e:
        q.error = str(e)

    time.sleep(REQUEST_SLEEP_SEC)
    return q

# ----------------------------
# EVENTS (LIGHTWEIGHT)
# ----------------------------
def fetch_brooklyn_events() -> List[Dict[str, str]]:
    """
    Simple forward-looking events pull using SerpApi Google Events engine.
    Keep it lightweight (top 5).
    """
    try:
        data = serp_get({
            "engine": "google_events",
            "q": "Brooklyn events",
            "hl": HL,
            "gl": GL,
            "no_cache": "true",
        })
        events = []
        for ev in (data.get("events_results") or [])[:5]:
            title = ev.get("title") or "Event"
            when = ev.get("date", {}).get("when") or ev.get("date", {}).get("start_date") or ""
            link = ev.get("link") or ""
            events.append({"title": title, "when": when, "link": short_url(link) if link else ""})
        return events
    except Exception as e:
        if DEBUG:
            print("EVENTS ERROR:", e)
        return []

# ----------------------------
# DUETTO-STYLE RECOMMENDATIONS
# ----------------------------
def rate_reco(your_price: Optional[float], comps: List[Quote]) -> str:
    """
    Very simple rules (you can refine later):
    - Compare your price vs comp median.
    - If you’re >10% above median → consider drop.
    - If you’re >10% below median → opportunity to raise.
    """
    comp_prices = sorted([c.price for c in comps if c.price is not None])
    if your_price is None or len(comp_prices) < 2:
        return "Insufficient pricing data to compute recommendation (missing prices)."

    median = comp_prices[len(comp_prices)//2]
    if median <= 0:
        return "Invalid comp pricing data."

    delta = (your_price - median) / median

    if delta > 0.10:
        return f"Bay Ridge is ~{delta*100:.0f}% ABOVE comp median (${median:.0f}). Consider testing a reduction or adding value-add fences (cancellation/meal) to defend rate."
    if delta < -0.10:
        return f"Bay Ridge is ~{abs(delta)*100:.0f}% BELOW comp median (${median:.0f}). Opportunity to push rate upward (or tighten discounts) while staying competitive."
    return f"Bay Ridge is in-line with comp median (${median:.0f}). Hold rate; adjust only if pickup/occupancy demands it."

# ----------------------------
# EMAIL BUILD (HTML)
# ----------------------------
def money(x: Optional[float]) -> str:
    return "—" if x is None else f"${x:,.0f}"

def build_email(portfolio: str, windows: Dict[str, List[Quote]], hotels: List[Hotel], your_key: str, events: List[Dict[str, str]]) -> Tuple[str, str]:
    today = ny_today()
    now = ny_now()

    subject = f"{portfolio} — Bay Ridge Forward Rate Intel (Tonight / D+7 / D+14) | {today.isoformat()}"

    # Index by key for display names
    name_map = {h.key: h.display_name for h in hotels}

    html = []
    html.append("<div style='font-family:Arial,Helvetica,sans-serif;color:#111;'>")
    html.append(f"<div style='font-size:18px;font-weight:700;margin-bottom:6px;'>{portfolio}</div>")
    html.append(f"<div style='font-size:12px;color:#555;margin-bottom:16px;'>Generated {now.strftime('%Y-%m-%d %I:%M %p')} ET • Identity locked via <b>property_token</b> • Source shown per rate</div>")

    # Events
    if events:
        html.append("<div style='margin:14px 0 8px;font-weight:700;'>Brooklyn Events (Market Signals)</div>")
        html.append("<ul style='margin-top:6px;'>")
        for ev in events:
            link = ev["link"]
            title = ev["title"]
            when = ev["when"]
            if link:
                html.append(f"<li style='margin-bottom:6px;'><a href='{link}' style='color:#0b57d0;text-decoration:none;'>{title}</a> <span style='color:#666;'>— {when}</span></li>")
            else:
                html.append(f"<li style='margin-bottom:6px;'>{title} <span style='color:#666;'>— {when}</span></li>")
        html.append("</ul>")

    # Each window section
    for label, quotes in windows.items():
        html.append(f"<div style='margin-top:18px;font-weight:800;font-size:14px;'>{label}</div>")
        html.append("<table cellpadding='0' cellspacing='0' style='border-collapse:collapse;width:100%;margin-top:8px;font-size:13px;'>")
        html.append("<tr>"
                    "<th align='left' style='padding:8px;border-bottom:1px solid #ddd;'>Hotel</th>"
                    "<th align='left' style='padding:8px;border-bottom:1px solid #ddd;'>Rate</th>"
                    "<th align='left' style='padding:8px;border-bottom:1px solid #ddd;'>Source</th>"
                    "<th align='left' style='padding:8px;border-bottom:1px solid #ddd;'>Link</th>"
                    "<th align='left' style='padding:8px;border-bottom:1px solid #ddd;'>Notes</th>"
                    "</tr>")

        # Split your hotel vs comps
        your_quote = next((q for q in quotes if q.hotel_key == your_key), None)
        comp_quotes = [q for q in quotes if q.hotel_key != your_key]

        for q in quotes:
            rate = money(q.price)
            src = q.source or "—"
            link = q.link or ""
            notes = q.error or ""

            hotel_name = name_map.get(q.hotel_key, q.display_name)

            if link:
                link_html = f"<a href='{link}' style='color:#0b57d0;text-decoration:none;'>Open</a>"
            else:
                link_html = "—"

            # Highlight Bay Ridge row
            bg = "#f3f6ff" if q.hotel_key == your_key else "#fff"
            fw = "700" if q.hotel_key == your_key else "400"

            html.append(
                "<tr>"
                f"<td style='padding:8px;border-bottom:1px solid #eee;background:{bg};font-weight:{fw};'>{hotel_name}</td>"
                f"<td style='padding:8px;border-bottom:1px solid #eee;background:{bg};font-weight:{fw};'>{rate}</td>"
                f"<td style='padding:8px;border-bottom:1px solid #eee;background:{bg};'>{src}</td>"
                f"<td style='padding:8px;border-bottom:1px solid #eee;background:{bg};'>{link_html}</td>"
                f"<td style='padding:8px;border-bottom:1px solid #eee;background:{bg};color:#a00;'>{notes}</td>"
                "</tr>"
            )
        html.append("</table>")

        # Recommendation block (Duetto-ish)
        reco = rate_reco(your_quote.price if your_quote else None, comp_quotes)
        html.append("<div style='margin-top:10px;padding:10px;border:1px solid #ddd;border-radius:10px;background:#fafafa;'>"
                    f"<div style='font-weight:800;margin-bottom:6px;'>Recommendation</div>"
                    f"<div style='font-size:13px;color:#222;'>{reco}</div>"
                    "<div style='font-size:12px;color:#666;margin-top:6px;'>"
                    "Easy adds to test in Lighthouse/Duetto-style workflow: fenced rates (NRF vs Flex), value-add packages, cancellation tightening, LOS controls on peak nights."
                    "</div>"
                    "</div>")

    html.append("</div>")
    return subject, "\n".join(html)

# ----------------------------
# SEND EMAIL
# ----------------------------
def send_email(subject: str, html_body: str):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and EMAIL_TO):
        raise RuntimeError("Missing SMTP config (SMTP_HOST/SMTP_USER/SMTP_PASS/EMAIL_TO).")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())

# ----------------------------
# MAIN
# ----------------------------
def main():
    portfolio, location, hotels = load_hotels()

    # Windows (Tonight, +7, +14)
    base = ny_today()
    windows_def = [
        ("TONIGHT (D0)", base, base + dt.timedelta(days=1)),
        ("D+7", base + dt.timedelta(days=7), base + dt.timedelta(days=8)),
        ("D+14", base + dt.timedelta(days=14), base + dt.timedelta(days=15)),
    ]

    your_key = "bay_ridge"

    # Fetch events once
    events = fetch_brooklyn_events()

    windows: Dict[str, List[Quote]] = {}

    for label, ci, co in windows_def:
        bucket = []
        for h in hotels:
            bucket.append(fetch_property_quote(h, ci, co))
        windows[label] = bucket

    subject, html = build_email(portfolio, windows, hotels, your_key, events)
    send_email(subject, html)

    if DEBUG:
        print("Email sent to:", EMAIL_TO)

if __name__ == "__main__":
    main()
