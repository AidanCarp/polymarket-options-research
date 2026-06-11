#!/usr/bin/env python3
"""
collect_daily.py
Fetches today's SI/GC option prices (Barchart OnDemand API) and Polymarket
"yes" probabilities, then appends new rows to:
  daily_options.csv     — date, underlying, strike, F, option_price
  daily_polymarket.csv  — date, underlying, strike, pm_prob

Requires env var:
  BARCHART_API_KEY  — free key from https://www.barchart.com/ondemand/free-api-key
                      (25 calls/day on the Basic free tier; this script uses 2)

Run: python collect_daily.py
     or via GitHub Actions (see .github/workflows/collect_daily.yml)
"""

import csv
import os
import sys
from datetime import datetime, timezone
import requests

DATA_DIR         = os.path.dirname(os.path.abspath(__file__))
BARCHART_KEY     = os.environ.get("BARCHART_API_KEY", "")
OPTIONS_CSV      = os.path.join(DATA_DIR, "daily_options.csv")
POLYMARKET_CSV   = os.path.join(DATA_DIR, "daily_polymarket.csv")
OPTIONS_FIELDS   = ["date", "underlying", "strike", "F", "option_price"]
PM_FIELDS        = ["date", "underlying", "strike", "pm_prob"]

# ── Contract tables ────────────────────────────────────────────────────────────
# Barchart OnDemand symbol → (underlying label, strike in $/oz)
# SI strikes are in cents/oz in the symbol (7000 ¢ = $70.00/oz)
# GC strikes are already in $/oz in the symbol
OPTION_CONTRACTS = {
    "SIQ26C6500^0":  ("SI",   65.0),
    "SIQ26C7000^0":  ("SI",   70.0),
    "SIQ26C7500^0":  ("SI",   75.0),
    "SIQ26C8000^0":  ("SI",   80.0),
    "SIQ26C8500^0":  ("SI",   85.0),
    "SIQ26C9000^0":  ("SI",   90.0),
    "SIQ26C9500^0":  ("SI",   95.0),
    "SIQ26C10000^0": ("SI",  100.0),
    "SIQ26C11000^0": ("SI",  110.0),
    "SIQ26C12000^0": ("SI",  120.0),
    "SIQ26C14000^0": ("SI",  140.0),
    "GCQ26C4600^0":  ("GC", 4600.0),
    "GCQ26C4800^0":  ("GC", 4800.0),
    "GCQ26C5000^0":  ("GC", 5000.0),
    "GCQ26C5200^0":  ("GC", 5200.0),
    "GCQ26C5400^0":  ("GC", 5400.0),
    "GCQ26C5600^0":  ("GC", 5600.0),
    "GCQ26C5800^0":  ("GC", 5800.0),
    "GCQ26C6000^0":  ("GC", 6000.0),
    "GCQ26C6200^0":  ("GC", 6200.0),
    "GCQ26C6500^0":  ("GC", 6500.0),
    "GCQ26C7000^0":  ("GC", 7000.0),
    "GCQ26C8000^0":  ("GC", 8000.0),
}

UNDERLYING_SYMBOLS = {"SI": "SIQ26", "GC": "GCQ26"}

# Polymarket "yes" token IDs (P(price > strike at June 2026 expiry))
PM_TOKENS = {
    "SI": {
         65: "83255198284961021522744975502577749660289761462119244630666928834550912637321",
         70: "11352502407822932465033186884283716468972859806612400377285769823372235577870",
         75: "64421299184851764888931054566640386328496621371620863500638358820445483634606",
         80: "111843114693857390983486068842086902975393052463018785788470804197607748709084",
         85: "109307690708325298695637158462270155889332008779910257934342711645845653510967",
         90: "105262476612663704758948158538809234993782885966448251464212857276110908305869",
         95: "20373780355039587946947921584212852591377751064902537354405479497879267330053",
        100: "54260098841642181939004878810404149178750287936719634666289443216010972822896",
        110: "23795636223281663426103142884087722365584501611082747992219968008455523264820",
        120: "34181493756413934965830606089571434442450447113141432931484510318729119136065",
        140: "107815291782368793318738647627481517343900945506454291650622064328756176506703",
    },
    "GC": {
        4600: "13028416245861859499650243273656770624805587748008215499335161565895014412662",
        4800: "44260049863930732587873753954163795593006986732474120541961732377065192692684",
        5000: "643041756421434685795313102297400846154123867125470159295228443607023076586",
        5200: "106385576643130661608600403167425880087720394417403497360787449120601654809218",
        5400: "90671288097995649082455223709734281568854849783762093482783802318774275280508",
        5600: "42899079046647064062817190717107037393692217495061216188592762055980279721828",
        5800: "3823827057212626071130170215091635673229111960416675747586832177811876926195",
        6000: "19398808305088983878604623316121536816822492384737303857970713820305334788727",
        6200: "96630671650447323341356569618316505523162316834650175610156382881851874800368",
        6500: "88796594327131239115152846681586783256340064802729531845476340242387373025184",
        7000: "71439269605198381086482578023891435827307721006677676774267334734011524437718",
        8000: "81311441313488227081578039705938570474171957374812448996962559052662169298751",
    },
}

# ── CSV helpers ────────────────────────────────────────────────────────────────

def existing_dates(path):
    """Return set of dates already in the CSV (empty set if file doesn't exist)."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {row["date"] for row in csv.DictReader(f)}


def append_rows(path, fieldnames, rows):
    """Append rows to CSV, writing header first if the file is new."""
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerows(rows)

# ── Polymarket collector ───────────────────────────────────────────────────────

def collect_polymarket():
    """Fetch the latest CLOB price for each token and append to daily_polymarket.csv."""
    done = existing_dates(POLYMARKET_CSV)
    rows = []

    for underlying, tokens in PM_TOKENS.items():
        for strike, token_id in tokens.items():
            try:
                r = requests.get(
                    "https://clob.polymarket.com/prices-history",
                    params={"market": token_id, "interval": "1d", "fidelity": 1440},
                    timeout=15,
                )
                r.raise_for_status()
                history = r.json().get("history", [])
                if not history:
                    print(f"  [PM] no data  {underlying} ${strike}")
                    continue

                last     = history[-1]
                row_date = datetime.fromtimestamp(last["t"], tz=timezone.utc).date().isoformat()

                if row_date in done:
                    print(f"  [PM] already have {row_date}  {underlying} ${strike}, skipping")
                    continue

                rows.append({"date": row_date, "underlying": underlying,
                             "strike": float(strike), "pm_prob": last["p"]})
                print(f"  [PM] {underlying} ${strike:>5}  {row_date}  prob={last['p']:.4f}")

            except Exception as exc:
                print(f"  [PM] ERROR {underlying} ${strike}: {exc}", file=sys.stderr)

    if rows:
        append_rows(POLYMARKET_CSV, PM_FIELDS, rows)
        print(f"[PM] appended {len(rows)} rows -> {POLYMARKET_CSV}")
    else:
        print("[PM] nothing new to append")

# ── Barchart collector ─────────────────────────────────────────────────────────

def collect_barchart():
    """
    Fetch option close prices and underlying last prices from Barchart OnDemand.
    Uses getQuote (1 call for options, 1 call for underlyings = 2 API calls total).
    """
    if not BARCHART_KEY:
        print("[BC] BARCHART_API_KEY not set -- skipping option prices.", file=sys.stderr)
        print("     Get a free key at https://www.barchart.com/ondemand/free-api-key")
        return

    done = existing_dates(OPTIONS_CSV)

    # --- underlying spot prices (1 API call) ----------------------------------
    und_prices = {}
    try:
        r = requests.get(
            "https://ondemand.websol.barchart.com/getQuote.json",
            params={"apikey": BARCHART_KEY,
                    "symbols": ",".join(UNDERLYING_SYMBOLS.values())},
            timeout=15,
        )
        r.raise_for_status()
        for q in r.json().get("results", []):
            for label, sym in UNDERLYING_SYMBOLS.items():
                if q.get("symbol") == sym:
                    und_prices[label] = q.get("close") or q.get("lastPrice")
                    print(f"  [BC] {label} ({sym}) close=${und_prices[label]}")
    except Exception as exc:
        print(f"  [BC] ERROR fetching underlyings: {exc}", file=sys.stderr)

    # --- option quotes (1 API call for all symbols) ---------------------------
    option_symbols = ",".join(OPTION_CONTRACTS.keys())
    option_quotes  = {}
    try:
        r = requests.get(
            "https://ondemand.websol.barchart.com/getQuote.json",
            params={"apikey": BARCHART_KEY, "symbols": option_symbols},
            timeout=15,
        )
        r.raise_for_status()
        for q in r.json().get("results", []):
            sym   = q.get("symbol", "")
            price = q.get("close") or q.get("lastPrice")
            trade = q.get("tradeTime", "")[:10]  # YYYY-MM-DD
            option_quotes[sym] = (price, trade)
            print(f"  [BC] {sym:<22}  close={price}  date={trade}")
    except Exception as exc:
        print(f"  [BC] ERROR fetching options: {exc}", file=sys.stderr)

    if not option_quotes:
        print("[BC] no option data returned -- market may be closed or symbol format wrong")
        return

    # --- build rows -----------------------------------------------------------
    rows = []
    for bc_sym, (underlying, strike) in OPTION_CONTRACTS.items():
        if bc_sym not in option_quotes:
            continue
        price, trade_date = option_quotes[bc_sym]
        if not price or not trade_date:
            continue
        if trade_date in done:
            print(f"  [BC] already have {trade_date}  {underlying} K=${strike}, skipping")
            continue
        rows.append({
            "date":         trade_date,
            "underlying":   underlying,
            "strike":       strike,
            "F":            und_prices.get(underlying, ""),
            "option_price": price,
        })

    if rows:
        append_rows(OPTIONS_CSV, OPTIONS_FIELDS, rows)
        print(f"[BC] appended {len(rows)} rows -> {OPTIONS_CSV}")
    else:
        print("[BC] nothing new to append")

# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Daily data collector ===")
    print(f"DATA_DIR: {DATA_DIR}\n")

    print("-- Polymarket --")
    collect_polymarket()

    print("\n-- Barchart --")
    collect_barchart()

    print("\nDone.")
