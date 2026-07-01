# Prediction Markets vs. Options-Implied Probabilities: Silver and Gold

This repository accompanies a research paper comparing the risk-neutral probabilities embedded in COMEX futures options prices with equivalent probabilities quoted on Polymarket, for Silver (SI) and Gold (GC) over the first half of 2026.

## Research question

Do prediction markets price tail events in commodity futures consistently with the options market? Specifically, is the Polymarket "yes" price for "Silver/Gold settles above $X" on a given day close to N(d2) derived from the corresponding COMEX call option under the Black-76 model?

Under Black-76, N(d2) is the risk-neutral probability that the futures price exceeds the strike at expiry. A Polymarket contract paying $1 if the same event occurs is theoretically worth the same amount, modulo liquidity, transaction costs, and any calendar basis between the two instruments.

## Key findings

Polymarket systematically underprices the risk-neutral probability N(d2) for both assets throughout the sample. The gap is large, persistent, and statistically overwhelming under all specifications.

| | Gold (GC) | Silver (SI) |
|---|---|---|
| Mean Δ = N(d2) − PM price | −0.089 | −0.055 |
| Plain t-statistic | −29.5 | −21.6 |
| HAC t-statistic (Newey-West, lag 5) | −15.9 | −14.6 |
| HAC t-statistic (Newey-West, lag 20) | −9.3 | −9.6 |
| AR(1) coefficient (daily mean Δ) | 0.768 | 0.466 |
| Half-life of gap | 2.6 trading days | 0.9 trading days |
| Convergence R² (Δ ~ T-to-expiry) | 0.309 | 0.056 |

The gap converges toward zero as contracts approach expiry — especially strongly for Gold — consistent with late-stage price discovery rather than a fixed structural mispricing. A realized risk-premium adjustment explains less than 1% of the Gold gap and goes in the wrong direction for Silver. All 24 tracked Polymarket contracts expired out-of-the-money on 2026-06-30.

## Sample

| | |
|---|---|
| **Period** | 2025-12-29 to 2026-06-29 |
| **Gold observations** | 1,326 (date × strike pairs) |
| **Silver observations** | 1,345 (date × strike pairs) |
| **Polymarket resolution** | 2026-06-30 17:00 EDT — all tracked strikes expired out-of-the-money |

## Instruments

| Underlying | Futures contract | Strikes covered | Polymarket event |
|---|---|---|---|
| Silver (SI) | SIU26 (Sep 2026) | $60, $65, $70, $75, $80, $85, $90, $95, $100, $110, $120, $140 /oz | "SI settles above $X on final trading day of June 2026?" |
| Gold (GC) | GCQ26 (Aug 2026) | $4600, $4800, $5000, $5200, $5400, $5600, $5800, $6000, $6200, $6500, $7000, $8000 /oz | "GC settles above $X on final trading day of June 2026?" |

### Gold futures basis

For Gold, the Active Month against which Polymarket resolves shifted during the sample period: GCM26 (June 2026 futures) was the Active Month from contract open until 2026-05-28, after which GCQ26 (August 2026 futures) became the Active Month. The pipeline uses GCM26 settlement prices as the Black-76 forward input for the pre-roll period and GCQ26 from 2026-05-29 onwards. GCM26 historical options data was unavailable; GCQ26 implied volatility is used throughout as the volatility input.

## Repository structure

```
.
├── _run_pipeline.py               # Standalone script: discovers Barchart CSVs,
│                                  #   computes IV + N(d2), fetches Polymarket,
│                                  #   saves merged_iv_polymarket.csv
├── Jupyternotebook.ipynb          # Full analysis notebook (all sections, charts)
├── fetch_volume_history.py        # Fetches Polymarket CLOB volume history
│
├── merged_iv_polymarket.csv       # Pipeline output: IV + N(d2) merged with Polymarket prices
├── gold_silver_volume_history.csv # Polymarket market volume by date and asset
│
├── stats_summary.csv              # One-sample t-test on Δ, per asset and per strike
├── hac_robustness.csv             # Newey-West HAC t-statistics (lags 5, 10, 20)
├── convergence_regression.csv     # OLS: Δ ~ T-to-expiry, per asset and pooled
├── timeseries_diagnostics.csv     # AR(1) coefficients, half-lives, ADF stationarity tests
├── risk_premium_estimate.csv      # Realized drift → probability-space risk-premium adjustment
├── regime_robustness.csv          # t-test split by low-IV / high-IV regime
├── extreme_strike_robustness.csv  # t-test with and without the two most extreme OTM strikes
├── null_benchmark.csv             # Permutation null: observed Δ vs 1,000 relabeling nulls
├── bidask_analysis.csv            # Bid-ask spread robustness (Silver)
├── bidask_gc_analysis.csv         # Bid-ask spread robustness (Gold)
│
├── nd2_vs_polymarket.png          # Main scatter/time-series chart
├── moneyness_decomposition.png    # Δ decomposed by moneyness
│
├── requirements.txt
└── .gitignore
```

> **Note on Barchart data:** The raw options and futures price CSVs downloaded from Barchart (`siu6_*`, `gcq6_*`, `siu26_*`, `gcq26_*`, `gcm26_*`) are proprietary under Barchart's Terms of Service and are not included in this repository. To reproduce results from scratch, download the equivalent files from [barchart.com](https://www.barchart.com) and place them in the project directory — the pipeline discovers them automatically by filename pattern.

## Running the analysis

### Requirements

```
pip install -r requirements.txt
```

Python 3.11 or newer. The notebook uses `statsmodels` for HAC standard errors and ADF tests, and `matplotlib` for all charts — both are listed in `requirements.txt`.

### Step 1 — obtain Barchart data (required for replication)

Download price-history CSVs from barchart.com for each instrument and save them in the project directory. The pipeline expects the standard Barchart export filenames:

- **Silver futures:** `siu26_daily_historical-data-<date>.csv`
- **Gold futures:** `gcq26_daily_historical-data-<date>.csv` and `gcm26_daily_historical-data-<date>.csv`
- **Silver options:** `siu6_<strike>c_price-history-<date>.csv`
- **Gold options:** `gcq6_<strike>c_price-history-<date>.csv`

Filename conventions:
- Silver option strikes in filenames are in ¢/oz — e.g. `7000c` = $70.00/oz. Barchart drops a trailing zero for the $100 strike, so that file is `siu6_1000c_...` (not `10000c`); the pipeline handles this automatically.
- Gold option strikes are in $/oz — e.g. `4600c` = $4,600/oz.

### Step 2 — run the pipeline

```bash
python _run_pipeline.py
```

This script:
1. Discovers all Barchart CSV files in the project directory automatically
2. Loads and aligns underlying futures price histories (applying the GCM26→GCQ26 basis adjustment for Gold)
3. Backs out Black-76 implied volatility for every (date, strike) row via Brent's method
4. Computes N(d2) — the risk-neutral probability P(F_T > K)
5. Fetches Polymarket price histories from the CLOB API (no authentication required)
6. Inner-joins options and Polymarket data on (date, underlying, strike)
7. Saves `merged_iv_polymarket.csv`

### Step 3 — run the notebook

Open `Jupyternotebook.ipynb` and run all cells. The notebook reads `merged_iv_polymarket.csv` and produces all tables, statistical tests, robustness checks, and charts used in the paper.

Key config parameters (top of notebook):

| Variable | Value | Description |
|---|---|---|
| `EXPIRY_SI` | 2026-09-25 | SIU26 options last trading day |
| `EXPIRY_GC` | 2026-07-28 | GCQ26 options last trading day |
| `RISK_FREE` | 0.045 | Annualised risk-free rate |
| `GC_BASIS_CUTOFF` | 2026-05-29 | Date Gold Active Month rolled from GCM26 to GCQ26 |

## Black-76 model reference

For a European call on a futures contract:

```
C = e^{-rT} * [F * N(d1) - K * N(d2)]

d1 = [ln(F/K) + (σ²/2) * T] / (σ * √T)
d2 = d1 - σ * √T
```

where F is the futures price, K is the strike, T is time to expiry in years, r is the risk-free rate, and σ is the implied volatility. N(d2) is the risk-neutral probability that F_T > K at expiry — the quantity compared against the Polymarket "yes" price throughout this paper.

Implied volatility is backed out numerically using Brent's method (`scipy.optimize.brentq`), searching over σ ∈ (0, 30).

## Data sources

- **Options and futures prices:** [Barchart.com](https://www.barchart.com) (not redistributed — see note above)
- **Prediction market prices:** [Polymarket CLOB API](https://clob.polymarket.com/prices-history) — public, no authentication required
