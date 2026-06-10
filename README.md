# Prediction Markets vs. Options-Implied Probabilities: Silver and Gold

This project compares the risk-neutral probabilities embedded in COMEX futures options prices with the equivalent probabilities quoted on Polymarket, for Silver (SI) and Gold (GC) over the first half of 2026.

## Research question

Do prediction markets price tail events in commodity futures consistently with the options market? Specifically, is the Polymarket "yes" price for "Silver settles above $X" on a given day close to N(d2) derived from the corresponding call option via the Black Scholes model?

Under Black-76, N(d2) is the risk-neutral probability that the futures price exceeds the strike at expiry. A Polymarket contract paying $1 if the same event occurs is theoretically worth the same amount, modulo liquidity, transaction costs, and any calendar basis between the two instruments.

## Instruments

| Underlying | Futures contract | Strikes covered | Polymarket event |
|---|---|---|---|
| Silver (SI) | SIQ26 (Aug 2026) | $65, $70, $75, $80, $85, $90, $95, $100 /oz | "Will SI settle over $X on the final trading day of June 2026?" |
| Gold (GC) | GCQ26 (Aug 2026) | $4600, $4800, $5000, $5200 /oz | "Gold above $X end of June?" |



## Repository structure

```
.
├── Jupyternotebook.ipynb            # Main analysis notebook
├── collect_daily.py                 # Daily data collector (run via GitHub Actions)
│
├── daily_polymarket.csv             # Accumulated Polymarket probabilities (appended daily)
├── daily_options.csv                # Accumulated option prices from Barchart (appended daily)
├── merged_iv_polymarket.csv         # Pipeline output: IV + N(d2) merged with Polymarket
│
├── siq26_daily_historical-*.csv     # Silver futures daily price history (Barchart download)
├── gcq26_daily_historical-*.csv     # Gold futures daily price history (Barchart download)
├── siq6_*c_price-history-*.csv      # Silver option price histories by strike (Barchart download)
├── gcq6_*c_price-history-*.csv      # Gold option price histories by strike (Barchart download)
│
└── .github/
    └── workflows/
        └── collect_daily.yml        # GitHub Actions workflow (runs Mon-Fri at 9am UTC)
```

## Data sources

### Historical option prices (Barchart)

The `siq6_*` and `gcq6_*` CSV files were downloaded manually from [barchart.com](https://www.barchart.com). Each file contains the full price history for one strike. The last line of every file is a `"Downloaded from Barchart.com..."` footer that the pipeline strips automatically.

Unit conventions (important for Black-76):
- **Silver**: underlying price in \$/oz; option premium in \$/oz; strike in the *filename* is in ¢/oz (e.g. `7000c` = \$70.00/oz strike)
- **Gold**: underlying price, option premium, and strike are all in \$/oz

### Polymarket probabilities

Fetched from the [Polymarket CLOB API](https://clob.polymarket.com/prices-history) using the market token IDs hardcoded in the notebook and `collect_daily.py`. No authentication is required.

## Running the analysis

### Requirements

```
pip install numpy pandas scipy requests matplotlib
```

Python 3.11 or newer recommended.

### Full historical pipeline (notebook)

Open `Jupyternotebook.ipynb` and run the cells in order. The pipeline cells (added below the original Polymarket fetch cells) will:

1. Discover all Barchart CSV files in the project folder automatically
2. Load underlying futures price histories
3. Back out Black-76 implied volatility for every (date, strike) row via Brent's method
4. Compute N(d2) — the risk-neutral probability P(F_T > K) under Black's model
5. Re-fetch Polymarket price histories from the CLOB API
6. Inner-join options and Polymarket data on (date, underlying, strike)
7. Save `merged_iv_polymarket.csv` and `nd2_vs_polymarket.png`

Key parameters at the top of the config cell:

| Variable | Default | Description |
|---|---|---|
| `EXPIRY_SI` | 2026-07-28 | SIQ26 options last trading day |
| `EXPIRY_GC` | 2026-07-28 | GCQ26 options last trading day |
| `RISK_FREE` | 0.045 | Annualised risk-free rate |

Verify expiry dates against the [CME Group contract specifications](https://www.cmegroup.com/markets/metals/precious/silver.html) before using results in the paper.

### Daily data collector (script)

`collect_daily.py` can be run locally or via GitHub Actions. It appends one row per (underlying, strike) to `daily_polymarket.csv` and `daily_options.csv`, and is idempotent — re-running on the same day skips rows that already exist.

**Polymarket collection** — no key required:

```bash
python collect_daily.py
```

**Barchart option prices** — requires a free API key:

1. Register at <https://www.barchart.com/ondemand/free-api-key> (Basic tier: 25 calls/day; the script uses 2)
2. Set the environment variable before running:

```bash
export BARCHART_API_KEY=your_key_here
python collect_daily.py
```

## GitHub Actions setup

The workflow in `.github/workflows/collect_daily.yml` runs the collector automatically on weekdays at 9am UTC and commits any new rows back to the repository.

**One-time setup:**

```bash
# Initialise the repo and push to GitHub
git init
git add .
git commit -m "init: research paper polymarket pipeline"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Then add your Barchart key as a repository secret:
- GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
- Name: `BARCHART_API_KEY`
- Value: your key

The workflow will commit under the `github-actions[bot]` identity. If no new data is available (market holiday, weekend), it exits cleanly without making a commit.

The Polymarket collector runs regardless of whether `BARCHART_API_KEY` is set, so `daily_polymarket.csv` accumulates every weekday even before you add the key.

## Black-76 model reference

For a European call on a futures contract:

```
C = e^{-rT} * [F * N(d1) - K * N(d2)]

d1 = [ln(F/K) + (σ²/2) * T] / (σ * √T)
d2 = d1 - σ * √T
```

where F is the futures price, K is the strike, T is time to expiry in years, r is the risk-free rate, and σ is the implied volatility. N(d2) is the risk-neutral probability that F_T > K at expiry, which is the quantity compared against the Polymarket "yes" price.

Implied volatility is backed out numerically using Brent's method (`scipy.optimize.brentq`), searching over σ ∈ (0, 30).
