"""Re-generate all downstream CSVs from the corrected merged_iv_polymarket.csv."""
import os, re
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy import stats as scipy_stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

DATA_DIR           = os.path.dirname(os.path.abspath(__file__))
EXPIRY_SI          = datetime(2026, 9, 25)
EXPIRY_GC          = datetime(2026, 7, 28)
RISK_FREE          = 0.045
GC_BASIS_CUTOFF    = pd.Timestamp("2026-05-29")
PM_RESOLUTION_DATE = pd.Timestamp("2026-06-30")

# ── helper functions (identical to notebook Cell 4) ───────────────────────────
def load_barchart_csv(path):
    df = pd.read_csv(path)
    df = df[~df["Time"].astype(str).str.startswith("Downloaded")]
    df["Time"] = pd.to_datetime(df["Time"])
    df["Latest"] = pd.to_numeric(df["Latest"], errors="coerce")
    return df.sort_values("Time").reset_index(drop=True)

def _d1d2(F, K, T, sigma):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def black76_call(F, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return np.exp(-r * max(T, 0)) * max(F - K, 0.0)
    d1, d2 = _d1d2(F, K, T, sigma)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def black76_nd2(F, K, T, r, sigma):
    if T <= 0:
        return 1.0 if F > K else 0.0
    if np.isnan(sigma) or sigma <= 0:
        return np.nan
    _, d2 = _d1d2(F, K, T, sigma)
    return norm.cdf(d2)

def implied_vol(C_mkt, F, K, T, r):
    if C_mkt <= 0 or T <= 0 or F <= 0 or K <= 0:
        return np.nan
    lower_bound = np.exp(-r * T) * max(F - K, 0.0)
    if C_mkt <= lower_bound + 1e-9:
        return np.nan
    try:
        return brentq(lambda s: black76_call(F, K, T, r, s) - C_mkt,
                      1e-6, 30.0, xtol=1e-7, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan

# ── load merged CSV (already has corrected T, iv, nd2) ───────────────────────
print("Loading merged_iv_polymarket.csv...")
merged = pd.read_csv(os.path.join(DATA_DIR, "merged_iv_polymarket.csv"))
merged["delta"] = merged["nd2"] - merged["pm_prob"]
print(f"  {len(merged)} rows, date range {merged['date'].min()} -> {merged['date'].max()}")

# ── load price series for risk premium cell (Cell 40) ────────────────────────
files = os.listdir(DATA_DIR)
und_si_file  = next((f for f in files if re.match(r"siu\d+_daily_historical", f)), None)
und_gc_file  = next((f for f in files if re.match(r"gcq\d+_daily_historical", f)), None)
und_gcm_file = next((f for f in files if re.match(r"gcm\d+_daily_historical", f)), None)

si_spot = (load_barchart_csv(os.path.join(DATA_DIR, und_si_file))
           [["Time", "Latest"]].rename(columns={"Latest": "F"}))
gc_spot = (load_barchart_csv(os.path.join(DATA_DIR, und_gc_file))
           [["Time", "Latest"]].rename(columns={"Latest": "F"}))

if und_gcm_file:
    gcm_spot = (load_barchart_csv(os.path.join(DATA_DIR, und_gcm_file))
                [["Time", "Latest"]].rename(columns={"Latest": "F_M26"}))
    gc_adj = gc_spot.merge(gcm_spot, on="Time", how="left")
    gc_adj["F"] = np.where(gc_adj["Time"] < GC_BASIS_CUTOFF, gc_adj["F_M26"], gc_adj["F"])
    gc_spot_adj = gc_adj[["Time", "F"]].dropna(subset=["F"])
else:
    gc_spot_adj = gc_spot

# ── Cell 14: t-test analysis → stats_summary.csv ─────────────────────────────
print("\n--- Cell 14: t-test ---")

def _ttest_row(underlying, strike_label, delta_series):
    delta  = delta_series.dropna().values
    n      = len(delta)
    t_stat, p_val = scipy_stats.ttest_1samp(delta, popmean=0)
    sem    = delta.std(ddof=1) / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    return {
        "underlying": underlying, "strike": strike_label, "n": n,
        "mean_delta": delta.mean(), "std_delta": delta.std(ddof=1),
        "t_stat": t_stat, "p_value": p_val,
        "ci_lo_95": delta.mean() - t_crit * sem,
        "ci_hi_95": delta.mean() + t_crit * sem,
        "reject_H0": bool(p_val < 0.05),
    }

overall_rows = [_ttest_row(und, "ALL", grp["delta"]) for und, grp in merged.groupby("underlying")]
strike_rows  = [_ttest_row(und, k, grp["delta"]) for (und, k), grp in merged.groupby(["underlying", "strike"])]
stats_df = pd.concat([pd.DataFrame(overall_rows).sort_values("underlying"),
                      pd.DataFrame(strike_rows).sort_values(["underlying", "strike"])],
                     ignore_index=True)
stats_df.to_csv(os.path.join(DATA_DIR, "stats_summary.csv"), index=False)
print(f"Saved -> stats_summary.csv")
print(stats_df[stats_df.strike == "ALL"][["underlying", "n", "mean_delta", "t_stat", "p_value"]].to_string(index=False))

# ── Cell 16: convergence regression → convergence_regression.csv ─────────────
print("\n--- Cell 16: convergence regression ---")
groups = [("SI", merged[merged["underlying"] == "SI"]),
          ("GC", merged[merged["underlying"] == "GC"]),
          ("Pooled", merged)]
rows = []
for label, sub in groups:
    sub = sub.dropna(subset=["delta", "T_years"])
    res = scipy_stats.linregress(sub["T_years"].values, sub["delta"].values)
    rows.append({"group": label, "n": len(sub), "intercept": res.intercept,
                 "coef_T": res.slope, "se_T": res.stderr,
                 "t_stat": res.slope / res.stderr, "p_value": res.pvalue,
                 "r_squared": res.rvalue ** 2})
reg_df = pd.DataFrame(rows)
reg_df.to_csv(os.path.join(DATA_DIR, "convergence_regression.csv"), index=False)
print(f"Saved -> convergence_regression.csv")
print(reg_df.to_string(index=False))

# ── Cell 18: bid-ask Silver → bidask_analysis.csv ────────────────────────────
print("\n--- Cell 18: bid-ask SI ---")
SI_SPREADS = {
     60: (68.0, 41.0), 65: (50.0, 56.0), 70: (24.0, 77.0), 75: (25.0, 86.0),
     80: (10.0, 94.0), 85: (7.0, 96.0),  90: (3.7, 98.2),  95: (3.6, 98.8),
    100: (1.8, 99.3), 110: (1.3, 99.0), 120: (2.0, 98.8), 140: (1.0, 99.2),
}
mean_deltas_si = merged[merged["underlying"] == "SI"].groupby("strike")["delta"].mean()
rows = []
for strike, (yes_c, no_c) in SI_SPREADS.items():
    mid = (yes_c + 100 - no_c) / 200
    half_spread = (yes_c + no_c - 100) / 200
    mean_delta = mean_deltas_si.get(float(strike), float("nan"))
    abs_delta = abs(mean_delta)
    rows.append({"strike": strike, "yes_ask_c": yes_c, "no_ask_c": no_c,
                 "mid": mid, "half_spread": half_spread, "mean_delta": mean_delta,
                 "abs_mean_delta": abs_delta, "exceeds_half_spread": bool(abs_delta > half_spread),
                 "margin": abs_delta - half_spread})
ba_df = pd.DataFrame(rows)
ba_df.to_csv(os.path.join(DATA_DIR, "bidask_analysis.csv"), index=False)
print(f"Saved -> bidask_analysis.csv")

# ── Cell 20: bid-ask Gold → bidask_gc_analysis.csv ───────────────────────────
print("\n--- Cell 20: bid-ask GC ---")
GC_SPREADS = {
    4600: (13.0, 89.0), 4800: (4.0, 96.1), 5000: (3.0, 98.0), 5200: (2.0, 99.1),
    5400: (1.8, 99.4),  5600: (1.7, 99.5), 5800: (1.4, 99.3), 6000: (1.0, 99.4),
    6200: (0.9, 99.5),  6500: (0.8, 99.4), 7000: (0.4, 99.7), 8000: (0.5, 99.8),
}
mean_deltas_gc = merged[merged["underlying"] == "GC"].groupby("strike")["delta"].mean()
rows = []
for strike, (yes_c, no_c) in GC_SPREADS.items():
    mid = (yes_c + 100 - no_c) / 200
    half_spread = (yes_c + no_c - 100) / 200
    mean_delta = mean_deltas_gc.get(float(strike), float("nan"))
    abs_delta = abs(mean_delta)
    rows.append({"strike": strike, "yes_ask_c": yes_c, "no_ask_c": no_c,
                 "mid": mid, "half_spread": half_spread, "mean_delta": mean_delta,
                 "abs_mean_delta": abs_delta, "exceeds_half_spread": bool(abs_delta > half_spread),
                 "margin": abs_delta - half_spread})
ba_gc_df = pd.DataFrame(rows)
ba_gc_df.to_csv(os.path.join(DATA_DIR, "bidask_gc_analysis.csv"), index=False)
print(f"Saved -> bidask_gc_analysis.csv")

# ── Cell 22: regime robustness → regime_robustness.csv ───────────────────────
print("\n--- Cell 22: regime robustness ---")
regime_rows = []
for und, grp in merged.dropna(subset=["iv", "delta"]).groupby("underlying"):
    median_iv = grp["iv"].median()
    for regime_label, mask in [("low-IV", grp["iv"] <= median_iv), ("high-IV", grp["iv"] > median_iv)]:
        row = _ttest_row(und, regime_label, grp.loc[mask, "delta"])
        row["median_iv"] = median_iv
        regime_rows.append(row)
regime_df = (pd.DataFrame(regime_rows)
             .rename(columns={"strike": "regime"})
             [["underlying", "regime", "median_iv", "n", "mean_delta", "std_delta",
               "t_stat", "p_value", "ci_lo_95", "ci_hi_95", "reject_H0"]])
regime_df.to_csv(os.path.join(DATA_DIR, "regime_robustness.csv"), index=False)
print(f"Saved -> regime_robustness.csv")

# ── Cell 30: extreme-strike robustness → extreme_strike_robustness.csv ───────
print("\n--- Cell 30: extreme-strike robustness ---")
EXCLUDE_STRIKES = {"SI": [120.0, 140.0], "GC": [7000.0, 8000.0]}
extreme_rows = []
for und, grp in merged.groupby("underlying"):
    excluded = EXCLUDE_STRIKES.get(und, [])
    excl_grp = grp[~grp["strike"].isin(excluded)]
    row_all = _ttest_row(und, "all_strikes", grp["delta"])
    row_all["excluded_strikes"] = ""
    row_all["n_excluded"] = 0
    extreme_rows.append(row_all)
    row_excl = _ttest_row(und, "excl_extreme_otm", excl_grp["delta"])
    row_excl["excluded_strikes"] = ", ".join(f"${k:.0f}" for k in excluded)
    row_excl["n_excluded"] = len(grp) - len(excl_grp)
    extreme_rows.append(row_excl)
extreme_df = (pd.DataFrame(extreme_rows)
              .rename(columns={"strike": "subset"})
              [["underlying", "subset", "excluded_strikes", "n_excluded", "n",
                "mean_delta", "std_delta", "t_stat", "p_value", "ci_lo_95", "ci_hi_95", "reject_H0"]])
extreme_df.to_csv(os.path.join(DATA_DIR, "extreme_strike_robustness.csv"), index=False)
print(f"Saved -> extreme_strike_robustness.csv")

# ── Cell 32: permutation null benchmark → null_benchmark.csv ─────────────────
print("\n--- Cell 32: permutation null benchmark ---")
N_PERM = 1000
rng = np.random.default_rng(42)
null_rows = []
for und, grp in merged.dropna(subset=["nd2", "pm_prob"]).groupby("underlying"):
    nd2_vals = grp["nd2"].values
    pm_vals  = grp["pm_prob"].values
    n        = len(grp)
    observed_mean = nd2_vals.mean() - pm_vals.mean()
    pooled = np.concatenate([nd2_vals, pm_vals])
    null_means = np.empty(N_PERM)
    for i in range(N_PERM):
        shuffled = rng.permutation(pooled)
        null_means[i] = shuffled[:n].mean() - shuffled[n:].mean()
    ci_lo, ci_hi = np.percentile(null_means, [2.5, 97.5])
    n_extreme = int((np.abs(null_means) >= abs(observed_mean)).sum())
    null_rows.append({"underlying": und, "n_obs": n, "n_permutations": N_PERM,
                      "observed_mean_delta": observed_mean,
                      "null_mean": null_means.mean(), "null_std": null_means.std(ddof=1),
                      "null_ci_lo_95": ci_lo, "null_ci_hi_95": ci_hi,
                      "n_extreme": n_extreme, "empirical_p_value": n_extreme / N_PERM})
null_df = pd.DataFrame(null_rows)
null_df.to_csv(os.path.join(DATA_DIR, "null_benchmark.csv"), index=False)
print(f"Saved -> null_benchmark.csv")

# ── Cell 34: risk-free sensitivity → risk_free_sensitivity.csv ───────────────
print("\n--- Cell 34: risk-free sensitivity (slow, row-by-row IV solve) ---")
RISK_FREE_ALT = 0.037
iv_alt_vals, nd2_alt_vals = [], []
for _, row in merged.iterrows():
    iv_a  = implied_vol(row["option_price"], row["F"], row["strike"], row["T_years"], RISK_FREE_ALT)
    nd2_a = black76_nd2(row["F"], row["strike"], row["T_years"], RISK_FREE_ALT, iv_a)
    iv_alt_vals.append(iv_a)
    nd2_alt_vals.append(nd2_a)
merged["iv_alt"]    = iv_alt_vals
merged["nd2_alt"]   = nd2_alt_vals
merged["delta_alt"] = merged["nd2_alt"] - merged["pm_prob"]

def _sens_row(und, strike_label, d_base, d_alt):
    d_base = d_base.dropna().values
    d_alt  = d_alt.dropna().values
    t_b, p_b = scipy_stats.ttest_1samp(d_base, popmean=0)
    t_a, p_a = scipy_stats.ttest_1samp(d_alt,  popmean=0)
    m_b, m_a = d_base.mean(), d_alt.mean()
    pct = (m_a - m_b) / abs(m_b) * 100 if m_b != 0 else float("nan")
    return {"underlying": und, "strike": strike_label, "n": len(d_base),
            "mean_delta_r0045": m_b, "mean_delta_r0037": m_a, "pct_change_mean_delta": pct,
            "t_stat_r0045": t_b, "p_value_r0045": p_b, "reject_H0_r0045": bool(p_b < 0.05),
            "t_stat_r0037": t_a, "p_value_r0037": p_a, "reject_H0_r0037": bool(p_a < 0.05),
            "significance_changes": bool(p_b < 0.05) != bool(p_a < 0.05)}

overall_sens = [_sens_row(und, "ALL", grp["delta"], grp["delta_alt"]) for und, grp in merged.groupby("underlying")]
strike_sens  = [_sens_row(und, str(k), grp["delta"], grp["delta_alt"]) for (und, k), grp in merged.groupby(["underlying", "strike"])]
sens_df = pd.concat([pd.DataFrame(overall_sens).sort_values("underlying"),
                     pd.DataFrame(strike_sens).sort_values(["underlying", "strike"])], ignore_index=True)
sens_df.to_csv(os.path.join(DATA_DIR, "risk_free_sensitivity.csv"), index=False)
print(f"Saved -> risk_free_sensitivity.csv")

# ── Cell 36: HAC robustness → hac_robustness.csv ─────────────────────────────
print("\n--- Cell 36: HAC robustness ---")
HAC_LAGS = [5, 10, 20]
hac_rows = []
for und, grp in merged.dropna(subset=["delta"]).groupby("underlying"):
    delta = grp.sort_values("date")["delta"].values
    n     = len(delta)
    t_plain, p_plain = scipy_stats.ttest_1samp(delta, popmean=0)
    for lag in HAC_LAGS:
        res = sm.OLS(delta, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lag}, use_t=True)
        hac_rows.append({"underlying": und, "n": n, "hac_maxlags": lag,
                         "mean_delta": float(res.params[0]),
                         "plain_t_stat": t_plain, "plain_p_value": p_plain,
                         "hac_t_stat": float(res.tvalues[0]), "hac_p_value": float(res.pvalues[0]),
                         "reject_H0_plain": bool(p_plain < 0.05),
                         "reject_H0_hac": bool(res.pvalues[0] < 0.05)})
hac_df = pd.DataFrame(hac_rows)
hac_df.to_csv(os.path.join(DATA_DIR, "hac_robustness.csv"), index=False)
print(f"Saved -> hac_robustness.csv")
print(hac_df.to_string(index=False))

# ── Cell 38: time-series diagnostics → timeseries_diagnostics.csv ────────────
print("\n--- Cell 38: time-series diagnostics ---")
date_delta = (merged.groupby(["underlying", "date"])["delta"].mean()
              .reset_index().sort_values(["underlying", "date"]))
diag_rows = []
for und, grp in date_delta.groupby("underlying"):
    y = grp["delta"].values
    n = len(y)
    y_t, y_lag = y[1:], y[:-1]
    ar1_res = sm.OLS(y_t, sm.add_constant(y_lag)).fit()
    rho    = float(ar1_res.params[1])
    rho_se = float(ar1_res.bse[1])
    half_life = np.log(0.5) / np.log(abs(rho)) if 0 < abs(rho) < 1 else float("nan")
    adf_stat, adf_pval, adf_lags_used, _, adf_crit, _ = adfuller(y, autolag="AIC", regression="c")
    diag_rows.append({"underlying": und, "n_dates": n, "ar1_coeff": rho, "ar1_se": rho_se,
                      "half_life_days": half_life, "adf_stat": adf_stat, "adf_pval": adf_pval,
                      "adf_lags_used": adf_lags_used,
                      "adf_crit_1pct": adf_crit["1%"], "adf_crit_5pct": adf_crit["5%"],
                      "adf_crit_10pct": adf_crit["10%"], "stationary_5pct": bool(adf_pval < 0.05)})
    print(f"  {und}: AR(1) rho={rho:.5f}, half_life={half_life:.2f} days, ADF p={adf_pval:.4f}")
diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(os.path.join(DATA_DIR, "timeseries_diagnostics.csv"), index=False)
print(f"Saved -> timeseries_diagnostics.csv")

# ── Cell 40: risk premium estimate → risk_premium_estimate.csv ────────────────
print("\n--- Cell 40: risk premium estimate ---")
K_GC_REF      = 5_400.0
K_SI_REF      =    85.0
MEAN_DELTA_GC = float(merged[merged["underlying"] == "GC"]["delta"].mean())
MEAN_DELTA_SI = float(merged[merged["underlying"] == "SI"]["delta"].mean())

gc_ts = gc_spot_adj.sort_values("Time").reset_index(drop=True).copy()
si_ts = si_spot.sort_values("Time").reset_index(drop=True).copy()
gc_ts["log_ret"] = np.log(gc_ts["F"] / gc_ts["F"].shift(1))
si_ts["log_ret"] = np.log(si_ts["F"] / si_ts["F"].shift(1))
gc_log = gc_ts["log_ret"].dropna()
si_log = si_ts["log_ret"].dropna()

def _rp_stats(log_ret, label):
    n        = len(log_ret)
    mu_log   = log_ret.mean() * 252
    sigma_ann = log_ret.std(ddof=1) * np.sqrt(252)
    mu_arith = mu_log + 0.5 * sigma_ann**2
    return dict(asset=label, n_days=n, mu_log_ann=mu_log, sigma_ann=sigma_ann,
                ito_correction=0.5*sigma_ann**2, mu_arith_ann=mu_arith,
                sharpe_log_xs=(mu_log - RISK_FREE) / sigma_ann,
                cum_log_ret=log_ret.sum())

s_gc = _rp_stats(gc_log, "GC (stitched GCM26/GCQ26)")
s_si = _rp_stats(si_log, "SI (SIU26)")

today_ts = pd.Timestamp(merged["date"].max())
T_gc_exp = (pd.Timestamp(EXPIRY_GC) - today_ts).days / 365.0
T_si_exp = (pd.Timestamp(EXPIRY_SI) - today_ts).days / 365.0
F_gc_now = gc_ts["F"].iloc[-1]
F_si_now = si_ts["F"].iloc[-1]

def _prob_adj(F, K, mu_arith, sigma_ann, T):
    d2_q = (np.log(F / K) - 0.5 * sigma_ann**2 * T) / (sigma_ann * np.sqrt(T))
    d2_p = d2_q + (mu_arith / sigma_ann) * np.sqrt(T)
    P_q, P_p = norm.cdf(d2_q), norm.cdf(d2_p)
    return dict(d2_q=d2_q, P_q=P_q, d2_p=d2_p, P_p=P_p, delta_P=P_p - P_q)

pa_gc = _prob_adj(F_gc_now, K_GC_REF, s_gc["mu_arith_ann"], s_gc["sigma_ann"], T_gc_exp)
pa_si = _prob_adj(F_si_now, K_SI_REF, s_si["mu_arith_ann"], s_si["sigma_ann"], T_si_exp)

rp_rows = []
for (s, pa, K, F, T, obs_d) in [(s_gc, pa_gc, K_GC_REF, F_gc_now, T_gc_exp, MEAN_DELTA_GC),
                                  (s_si, pa_si, K_SI_REF, F_si_now, T_si_exp, MEAN_DELTA_SI)]:
    frac = pa["delta_P"] / abs(obs_d) if obs_d != 0 else np.nan
    rp_rows.append({"asset": s["asset"], "n_days": s["n_days"],
                    "mu_log_ann": round(s["mu_log_ann"], 6), "sigma_ann": round(s["sigma_ann"], 6),
                    "ito_correction": round(s["ito_correction"], 6),
                    "mu_arith_ann": round(s["mu_arith_ann"], 6),
                    "sharpe_log_xs": round(s["sharpe_log_xs"], 4),
                    "cum_log_ret": round(s["cum_log_ret"], 6),
                    "ref_strike_K": K, "latest_futures_F": round(F, 4),
                    "T_to_expiry_years": round(T, 6),
                    "d2_Q": round(pa["d2_q"], 6), "d2_P": round(pa["d2_p"], 6),
                    "N_d2_Q": round(pa["P_q"], 8), "N_d2_P": round(pa["P_p"], 8),
                    "delta_P_risk_premium": round(pa["delta_P"], 8),
                    "observed_mean_delta": obs_d,
                    "fraction_explained": round(frac, 6)})

rp_df = pd.DataFrame(rp_rows)
rp_df.to_csv(os.path.join(DATA_DIR, "risk_premium_estimate.csv"), index=False)
print(f"Saved -> risk_premium_estimate.csv")
print(rp_df[["asset", "observed_mean_delta", "delta_P_risk_premium", "fraction_explained"]].to_string(index=False))

print("\n=== All downstream CSVs regenerated successfully ===")
