import os, json, re, ssl, smtplib, datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from dateutil import tz
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

NY_TZ = tz.gettz("America/New_York")

# ---------- ENV ----------
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

# Windows (forward only)
OFFSETS = [0, 7, 14]

# Recommendation controls (simple + explainable)
MIN_RATE = 99
MAX_RATE = 399
RAISE_IF_BELOW_AVG_PCT = 0.08
DROP_IF_ABOVE_AVG_PCT  = 0.10
BASE_STEP = 10
WEEKEND_BONUS = 10
EVENT_BONUS = 10


# ---------- MODELS ----------
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


# ---------- HELPERS ----------
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
        host = urlparse(url).netloc.lower()
        host = host.replace("www.", "")
        return host
    except Exception:
        return ""

def serp_get(params: Dict[str, Any]) -> Dict[str, Any]:
    if not SERP_API_KEY:
        raise RuntimeError("Missing SERP_API_KEY in GitHub Secrets.")
    params = dict(params)
    params["api_key"] = SERP_API_KEY
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
    r.raise_for_status()
    return r.json()


# ---------- TOKEN BOOTSTRAP ----------
def bootstrap_tokens(hotels: List[Hotel], location: str) -> Dict[str, str]:
    """
    More reliable than autocomplete:
    Use engine=google_hotels and grab properties[0].property_token.
    """
    results: Dict[str, str] = {}

    # pick a safe future date so hotels returns properties
    today = ny_today()
    check_in = today + dt.timedelta(days=7)
    check_out = check_in + dt.timedelta(days=1)

    for h in hotels:
        data = serp_get({
            "engine": "google_hotels",
            "q": h.query,
            "check_in_date": check_in.isoformat(),
            "check_out_date": check_out.isoformat(),
            "currency": CURRENCY,
            "adults": ADULTS,
            "rooms": ROOMS,
            "location": location,
            "hl": "en",
            "gl": "us",
        })

        props = data.get("properties") or []
        token = ""
        picked_name = ""
        picked_addr = ""

        if isinstance(props, list) and props:
            # try best name match first
            target = (h.query or "").lower()
            chosen = props[0]
            for p in props:
                nm = (p.get("name") or "").lower()
                if nm and (nm in target or target in nm):
                    chosen = p
                    break

            token = str(chosen.get("property_token") or "")
            picked_name = str(chosen.get("name") or "")
            picked_addr = str(chosen.get("address") or "")

        results[h.key] = token

        print("\n=== TOKEN LOOKUP (google_hotels) ===")
        print("Hotel:", h.display_name)
        print("Query:", h.query)
        print("Picked:", picked_name, "|", picked_addr)
        print("Token:", token or "NOT FOUND")
        print("===================================\n")

    return results


# ---------- RATE FETCH (TOKEN-LOCKED) ----------
def fetch_property_quote(h: Hotel, location: str, check_in: dt.date, check_out: dt.date) -> Quote:
    """
    Uses google_hotels_property_details (token-locked).
    Prefers lowest OTA offer from prices/offers. Falls back to typical/base when needed.
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

    # basic identity for trust
    q.matched_name = str(data.get("property_name") or data.get("title") or data.get("name") or "")
    q.matched_address = str(data.get("address") or "")

    # Try to find offers/prices structures
    candidates: List[Tuple[float, str, str]] = []

    def consider(rate: Any, name: Any, url: Any):
        if isinstance(rate, (int, float)) and rate > 0:
            candidates.append((float(rate), str(name or "Unknown"), str(url or "")))

    # Common patterns: "prices" list or "offers" list
    prices = data.get("prices")
    if isinstance(prices, list):
        for p in prices:
            if not isinstance(p, dict):
                continue
            consider(p.get("extracted_rate") or p.get("rate"), p.get("name") or p.get("source"), p.get("link") or p.get("url"))

    offers = data.get("offers")
    if isinstance(offers, list):
        for o in offers:
            if not isinstance(o, dict):
                continue
            consider(o.get("extracted_rate") or o.get("rate"), o.get("name") or o.get("source"), o.get("link") or o.get("url"))

    # pick best
    if candidates:
        candidates.sort(key=lambda t: t[0])
        best = candidates[0]
        q.price, q.ota_name, q.ota_url = best[0], best[1], best[2]
    else:
        # fallback: some responses contain "rate_per_night"
        rpn = data.get("rate_per_night")
        if isinstance(rpn, dict):
            val = rpn.get("extracted_lowest") or rpn.get("lowest")
            if isinstance(val, (int, float)) and val > 0:
                q.price = float(val)
                q.ota_name = "Google Hotels (base)"
                q.ota_url = str(data.get("link") or "")

    if DEBUG:
        print("\n==== DEBUG QUOTE ====")
        print("Hotel:", h.display_name, "| Token:", h.property_token)
        print("Dates:", check_in.isoformat(), "->", check_out.isoformat())
        print("Matched:", q.matched_name, "|", q.matched_address)
        print("Selected:", q.price, q.ota_name, q.ota_url)
        print("=====================\n")

    return q


# ---------- EVENTS (Brooklyn) ----------
def fetch_brooklyn_events() -> List[Dict[str, str]]:
    """
    Pulls a small set of forward-looking Brooklyn events using SerpApi google_events.
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


# ---------- RECOMMENDATION ----------
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


# ---------- EMAIL ----------
def send_email(subject: str, html: str, text: str) -> None:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        raise RuntimeError("Missing SMTP_* secrets. Set SMTP_HOST/PORT/USER/PASS in GitHub Secrets.")

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


def build_email(portfolio: str, windows: Dict[int, List[Quote]], hotels: List[Hotel], your_key: str, events: List[Dict[str, str]]) -> Tuple[str, str, str]:
    today = ny_today()
    now = ny_now()

    subject = f"{portfolio} — Bay Ridge Forward Rate Intel (D0 / D+7 / D+14) | {today.isoformat()}"

    # --- plain text ---
    text_lines = []
    text_lines.append(f"{portfolio}")
    text_lines.append(f"Generated: {now.strftime('%Y-%m-%d %I:%M %p')} ET")
    text_lines.append("Locked identity: property_token (stable hotel match)")
    text_lines.append("")

    # --- html ---
    html = []
    html.append("<div style='font-family: Arial, Helvetica, sans-serif; color:#111;'>")
    html.append(f"<h2 style='margin:0 0 6px 0;'>{portfolio}</h2>")
    html.append(f"<div style='font-size:12px;color:#444;margin-bottom:14px;'>Generated: {now.strftime('%Y-%m-%d %I:%M %p')} ET</div>")
    html.append("<div style='font-size:12px;color:#444;margin-bottom:18px;'>")
    html.append("Identity is locked via <b>property_token</b> (prevents wrong-property drift).")
    html.append("</div>")

    # Executive summary
    html.append("<h3 style='margin:0 0 8px 0;'>Executive Summary</h3>")
    html.append("<ul style='margin-top:0;'>")

    for off in OFFSETS:
        d = today + dt.timedelta(days=off)
        event_boost = any(d.isoformat() in (e.get("date") or "") for e in events)  # light heuristic
        rec, why = recommend(windows[off], your_key, d, event_boost)
        label = "Tonight (D0)" if off == 0 else f"D+{off}"
        html.append(f"<li><b>{label}</b> ({d.isoformat()}): {rec} <span style='color:#666;font-size:12px;'>— {why}</span></li>")
        text_lines.append(f"- {label} {d.isoformat()}: {rec} — {why}")

    html.append("</ul>")

    # Sections
    for off in OFFSETS:
        d = today + dt.timedelta(days=off)
        label = "Tonight (D0)" if off == 0 else f"D+{off}"

        quotes = windows[off]
        quotes_sorted = sorted(quotes, key=lambda q: (q.price is None, q.price if q.price is not None else 10**9))

        html.append(f"<h3 style='margin:18px 0 8px 0;'>{label} — Check-in {d.isoformat()} (1 night)</h3>")
        html.append("<table style='border-collapse:collapse;width:100%;font-size:13px;'>")
        html.append("<tr>"
                    "<th style='text-align:left;border-bottom:1px solid #ddd;padding:8px;'>Hotel</th>"
                    "<th style='text-align:right;border-bottom:1px solid #ddd;padding:8px;'>Entry Rate</th>"
                    "<th style='text-align:left;border-bottom:1px solid #ddd;padding:8px;'>Source</th>"
                    "<th style='text-align:left;border-bottom:1px solid #ddd;padding:8px;'>Matched Listing</th>"
                    "</tr>")

        for q in quotes_sorted:
            is_you = (q.hotel_key == your_key)
            rate = money(q.price)
            dom = short_domain(q.ota_url)
            src = q.ota_name if q.ota_name else "Unknown"
            src_display = f"{src} ({dom})" if dom else src

            matched = (q.matched_name or "").strip()
            if q.matched_address:
                matched = f"{matched} — {q.matched_address}".strip(" —")

            # short “open” link
            open_link = ""
            if q.ota_url:
                open_link = f" <a href='{q.ota_url}' style='color:#0b57d0;text-decoration:none;font-size:12px;'>(open)</a>"

            row_style = "background:#111;color:#fff;" if is_you else ""
            html.append("<tr>")
            html.append(f"<td style='padding:8px;border-bottom:1px solid #f0f0f0;{row_style}'><b>{q.display_name}</b></td>")
            html.append(f"<td style='padding:8px;border-bottom:1px solid #f0f0f0;text-align:right;{row_style}'>{rate}</td>")
            html.append(f"<td style='padding:8px;border-bottom:1px solid #f0f0f0;{row_style}'>{src_display}{open_link}</td>")
            html.append(f"<td style='padding:8px;border-bottom:1px solid #f0f0f0;{row_style}'>{matched}</td>")
            html.append("</tr>")

        html.append("</table>")

    # Events block
    html.append("<h3 style='margin:18px 0 8px 0;'>Brooklyn Events (signals)</h3>")
    if events:
        html.append("<ul style='margin-top:0;'>")
        for e in events[:8]:
            t = e.get("title","")
            when = e.get("date","")
            v = e.get("venue","")
            html.append(f"<li><b>{t}</b> <span style='color:#666;font-size:12px;'>— {when} {v}</span></li>")
        html.append("</ul>")
    else:
        html.append("<div style='color:#666;font-size:12px;'>No events returned.</div>")

    html.append("</div>")

    return subject, "\n".join(text_lines), "".join(html)


# ---------- CONFIG ----------
def load_config() -> Tuple[str, str, List[Hotel]]:
    with open("hotels.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    portfolio = str(cfg.get("portfolio_name") or "Revenue Intel")
    location = str(cfg.get("location") or "Brooklyn, New York, United States")

    hotels = []
    for h in cfg.get("hotels", []):
        hotels.append(Hotel(
            key=str(h.get("key")),
            display_name=str(h.get("display_name")),
            query=str(h.get("query")),
            property_token=str(h.get("property_token") or ""),
        ))
    return portfolio, location, hotels

def save_tokens(tokens: Dict[str, str]) -> None:
    with open("hotels.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for h in cfg.get("hotels", []):
        k = h.get("key")
        if k in tokens and tokens[k]:
            h["property_token"] = tokens[k]
    with open("hotels.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


# ---------- MAIN ----------
def main():
    portfolio, location, hotels = load_config()

    # choose Bay Ridge as “your” hotel for now
    your = next((h for h in hotels if h.key == "bay_ridge"), None)
    if not your:
        raise RuntimeError("Missing bay_ridge hotel entry in hotels.json")

    # If ANY tokens missing, bootstrap once and stop (turn-key setup)
    missing = [h for h in hotels if not h.property_token]
    if missing:
        tokens = bootstrap_tokens(hotels, location)
        # auto-write tokens that were found
        save_tokens(tokens)

        # Tell you what to do next in email (so it’s turn-key)
        subject = f"{portfolio} — Setup: property_token needed"
        txt = "Tokens were fetched and written into hotels.json where available.\n\nIf any remain blank, refine the query text in hotels.json and re-run.\n"
        html = "<div style='font-family:Arial;'><h3>Setup required</h3><p>Tokens were fetched and written into <b>hotels.json</b> where available.</p>" \
               "<p>If any token remains blank, refine the <b>query</b> text and run again.</p></div>"

        send_email(subject, html, txt)
        print("Setup email sent. Re-run after tokens are populated.")
        return

    # pull events signals
    events = fetch_brooklyn_events()

    # build windows
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
