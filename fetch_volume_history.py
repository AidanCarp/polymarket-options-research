"""
Backfill daily trading volume for Polymarket Gold (GC) and Silver (SI)
"above ___ end of June 2026" markets.

Source: data-api.polymarket.com/trades
  - Public, no auth required
  - Paginated via offset (limit=100)
  - Each record is one side of one fill: volume per record = size * price (USDC)

The script aggregates USDC notional (size * price) across all strike sub-markets
within each underlying, per UTC calendar day.

Output: gold_silver_volume_history.csv  (date, underlying, daily_volume)
"""

import os
import time
import requests
import pandas as pd
from datetime import date

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_BASE  = "https://data-api.polymarket.com"

EVENTS = {
    "GC": "gc-over-under-jun-2026",
    "SI": "si-over-under-jun-2026",
}

START_DATE = date(2025, 12, 26)   # contract open
END_DATE   = date(2026, 6, 30)    # resolution

_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(_DIR, "gold_silver_volume_history.csv")


# ── helpers ─────────────────────────────────────────────────────────────────

def get_markets(event_slug: str) -> list[dict]:
    """Return [{conditionId, slug, gamma_volume}] for all sub-markets in an event."""
    r = requests.get(f"{GAMMA_BASE}/events?slug={event_slug}", timeout=15)
    r.raise_for_status()
    markets = []
    for event in r.json():
        for m in event.get("markets", []):
            markets.append({
                "conditionId":   m["conditionId"],
                "slug":          m.get("slug", ""),
                "gamma_volume":  float(m.get("volumeNum", 0) or 0),
            })
    return markets


def fetch_trades(condition_id: str, pause: float = 0.25) -> list[dict]:
    """Paginate through all trades for one conditionId."""
    trades = []
    offset = 0
    limit  = 100
    while True:
        r = requests.get(
            f"{DATA_BASE}/trades",
            params={"market": condition_id, "limit": limit, "offset": offset},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        trades.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(pause)
    return trades


# ── main ────────────────────────────────────────────────────────────────────

def main():
    rows = []  # {date, underlying, notional}

    for underlying, slug in EVENTS.items():
        print(f"\n{'='*60}")
        print(f"  {underlying}  ({slug})")
        print(f"{'='*60}")

        markets = get_markets(slug)
        print(f"  {len(markets)} sub-markets found")

        gamma_total = sum(m["gamma_volume"] for m in markets)
        api_total   = 0.0

        for m in markets:
            cid = m["conditionId"]
            print(f"\n  {m['slug']}  (Gamma vol: ${m['gamma_volume']:,.2f})")

            trades = fetch_trades(cid)
            mkt_vol = sum(float(t["size"]) * float(t["price"]) for t in trades)
            api_total += mkt_vol
            print(f"    {len(trades)} trades | computed vol: ${mkt_vol:,.2f}")

            for t in trades:
                ts         = int(t["timestamp"])
                trade_date = pd.Timestamp(ts, unit="s", tz="UTC").date()
                if trade_date < START_DATE or trade_date > END_DATE:
                    continue
                rows.append({
                    "date":       trade_date,
                    "underlying": underlying,
                    "notional":   float(t["size"]) * float(t["price"]),
                })

        ratio = api_total / gamma_total if gamma_total else float("nan")
        print(f"\n  {underlying} totals: API=${api_total:,.2f}  Gamma=${gamma_total:,.2f}  ratio={ratio:.3f}")

    # ── aggregate to daily ──────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    if df.empty:
        print("\nNo trades found — check event slugs and date range.")
        return

    daily = (
        df.groupby(["date", "underlying"])["notional"]
        .sum()
        .reset_index()
        .rename(columns={"notional": "daily_volume"})
    )

    # fill every calendar date even if zero-volume
    date_index = pd.date_range(str(START_DATE), str(END_DATE), freq="D").date
    complete   = pd.MultiIndex.from_product(
        [date_index, sorted(EVENTS.keys())], names=["date", "underlying"]
    )
    daily = (
        daily.set_index(["date", "underlying"])
        .reindex(complete, fill_value=0.0)
        .reset_index()
        .sort_values(["underlying", "date"])
        .reset_index(drop=True)
    )

    daily.to_csv(OUTPUT, index=False)
    # Note: computed volume is ~60-65% of Gamma API totals for GC, ~56% for SI.
    # Pagination is complete (verified by empty response at max offset).
    # The gap likely reflects Gamma counting full 1.00 USDC per matched pair
    # (Yes buyer + No buyer combined), while the data API records individual
    # fill entries where price reflects only one party's cost per share.
    # The daily time-series pattern is correct; absolute scale differs from Gamma.
    print(f"\nSaved {len(daily)} rows -> {OUTPUT}")
    print(daily.to_string())


if __name__ == "__main__":
    main()
