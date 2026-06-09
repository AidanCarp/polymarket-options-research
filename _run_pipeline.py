import os, re, sys
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import requests

warnings.filterwarnings("ignore")

DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
EXPIRY_SI = datetime(2026, 7, 28)
EXPIRY_GC = datetime(2026, 7, 28)
RISK_FREE = 0.045

print(f"DATA_DIR : {DATA_DIR}")
print(f"Expiry SI: {EXPIRY_SI.date()}  |  Expiry GC: {EXPIRY_GC.date()}")
print(f"Risk-free: {RISK_FREE:.1%}")

# ── helpers ────────────────────────────────────────────────────────────────────
def load_barchart_csv(path):
    df = pd.read_csv(path)
    df = df[~df["Time"].astype(str).str.startswith("Downloaded")]
    df["Time"]   = pd.to_datetime(df["Time"])
    df["Latest"] = pd.to_numeric(df["Latest"], errors="coerce")
    return df.sort_values("Time").reset_index(drop=True)

def discover_files(data_dir):
    files  = os.listdir(data_dir)
    opt_si = sorted(f for f in files if re.match(r"siq\d+_\d+c_price-history", f))
    opt_gc = sorted(f for f in files if re.match(r"gcq\d+_\d+c_price-history", f))
    und_si = next((f for f in files if re.match(r"siq\d+_daily_historical", f)), None)
    und_gc = next((f for f in files if re.match(r"gcq\d+_daily_historical", f)), None)
    print("\nSI option files :", opt_si)
    print("GC option files :", opt_gc)
    print("SI underlying   :", und_si)
    print("GC underlying   :", und_gc)
    return opt_si, opt_gc, und_si, und_gc

def parse_si_strike(filename):
    m = re.search(r"siq\d+_(\d+)c_", filename)
    if not m:
        return None
    nominal = int(m.group(1))
    if nominal == 1000:
        nominal = 10000          # Barchart dropped a zero for the $100 strike
    return nominal / 100.0       # cents/oz -> $/oz

def parse_gc_strike(filename):
    m = re.search(r"gcq\d+_(\d+)c_", filename)
    return float(m.group(1)) if m else None

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
    lower = np.exp(-r * T) * max(F - K, 0.0)
    if C_mkt <= lower + 1e-9:
        return np.nan
    try:
        return brentq(lambda s: black76_call(F, K, T, r, s) - C_mkt,
                      1e-6, 30.0, xtol=1e-7, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan

# ── discover files & load underlyings ─────────────────────────────────────────
opt_si_files, opt_gc_files, und_si_file, und_gc_file = discover_files(DATA_DIR)

si_spot = (load_barchart_csv(os.path.join(DATA_DIR, und_si_file))
           [["Time", "Latest"]].rename(columns={"Latest": "F"}))
gc_spot = (load_barchart_csv(os.path.join(DATA_DIR, und_gc_file))
           [["Time", "Latest"]].rename(columns={"Latest": "F"}))

print(f"\nSI spot: {len(si_spot)} days  |  latest F = ${si_spot['F'].iloc[-1]:.3f}/oz")
print(f"GC spot: {len(gc_spot)} days  |  latest F = ${gc_spot['F'].iloc[-1]:.2f}/oz")

# ── process options ────────────────────────────────────────────────────────────
def process_options(option_files, parse_strike_fn, spot_df, expiry, label):
    rows = []
    for fname in option_files:
        K = parse_strike_fn(fname)
        if K is None:
            print(f"  [skip] {fname}")
            continue
        opt = (load_barchart_csv(os.path.join(DATA_DIR, fname))
               [["Time", "Latest"]].rename(columns={"Latest": "option_price"}))
        opt = opt.merge(spot_df, on="Time", how="inner")
        n_valid = 0
        for _, row in opt.iterrows():
            T   = (expiry - row["Time"].to_pydatetime()).days / 365.0
            iv  = implied_vol(row["option_price"], row["F"], K, T, RISK_FREE)
            nd2 = black76_nd2(row["F"], K, T, RISK_FREE, iv)
            n_valid += int(not np.isnan(iv))
            rows.append({"date": row["Time"].date(), "underlying": label,
                         "strike": K, "F": row["F"],
                         "option_price": row["option_price"],
                         "T_years": T, "iv": iv, "nd2": nd2})
        print(f"  {fname}  ->  K=${K:.2f}  |  {n_valid}/{len(opt)} valid IVs")
    return pd.DataFrame(rows)

print("\nProcessing Silver options...")
si_opts = process_options(opt_si_files, parse_si_strike, si_spot, EXPIRY_SI, "SI")
print("\nProcessing Gold options...")
gc_opts = process_options(opt_gc_files, parse_gc_strike, gc_spot, EXPIRY_GC, "GC")
options_df = pd.concat([si_opts, gc_opts], ignore_index=True)
print(f"\nTotal option rows: {len(options_df)}")

# ── fetch Polymarket ───────────────────────────────────────────────────────────
SI_MARKETS = {
    100: "54260098841642181939004878810404149178750287936719634666289443216010972822896",
    95:  "20373780355039587946947921584212852591377751064902537354405479497879267330053",
    90:  "105262476612663704758948158538809234993782885966448251464212857276110908305869",
    85:  "109307690708325298695637158462270155889332008779910257934342711645845653510967",
    80:  "111843114693857390983486068842086902975393052463018785788470804197607748709084",
    75:  "64421299184851764888931054566640386328496621371620863500638358820445483634606",
    70:  "11352502407822932465033186884283716468972859806612400377285769823372235577870",
    65:  "83255198284961021522744975502577749660289761462119244630666928834550912637321",
}
GC_MARKETS = {
    5200: "106385576643130661608600403167425880087720394417403497360787449120601654809218",
    5000: "643041756421434685795313102297400846154123867125470159295228443607023076586",
    4800: "44260049863930732587873753954163795593006986732474120541961732377065192692684",
    4600: "13028416245861859499650243273656770624805587748008215499335161565895014412662",
}

def fetch_polymarket(markets_dict, label):
    dfs = []
    for strike, token_id in markets_dict.items():
        r = requests.get(
            "https://clob.polymarket.com/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": 1440},
            timeout=15,
        )
        history = r.json().get("history", [])
        if not history:
            print(f"  [no data] {label} ${strike}")
            continue
        df = pd.DataFrame(history)
        df["date"]       = pd.to_datetime(df["t"], unit="s").dt.date
        df["underlying"] = label
        df["strike"]     = float(strike)
        df["pm_prob"]    = df["p"]
        dfs.append(df[["date", "underlying", "strike", "pm_prob"]])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

print("\nFetching SI Polymarket prices...")
si_pm = fetch_polymarket(SI_MARKETS, "SI")
print(f"  {len(si_pm)} rows")
print("Fetching GC Polymarket prices...")
gc_pm = fetch_polymarket(GC_MARKETS, "GC")
print(f"  {len(gc_pm)} rows")
pm_df = pd.concat([si_pm, gc_pm], ignore_index=True)
print(f"Polymarket total: {len(pm_df)} rows")

# ── merge ──────────────────────────────────────────────────────────────────────
# Polymarket API can emit two ticks per calendar day; keep the last one
pm_dedup = pm_df.sort_values("date").drop_duplicates(
    subset=["date", "underlying", "strike"], keep="last"
)
merged = options_df.merge(pm_dedup, on=["date", "underlying", "strike"], how="inner")
print(f"\nMerged rows : {len(merged)}")
print(f"Date range  : {merged['date'].min()}  ->  {merged['date'].max()}")
print(f"SI strikes  : {sorted(merged.loc[merged.underlying=='SI','strike'].unique())}")
print(f"GC strikes  : {sorted(merged.loc[merged.underlying=='GC','strike'].unique())}")

out_path = os.path.join(DATA_DIR, "merged_iv_polymarket.csv")
merged.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")

# ── latest snapshot ────────────────────────────────────────────────────────────
latest = merged[merged["date"] == merged["date"].max()]
print(f"\n-- Latest snapshot ({merged['date'].max()}) ------------------------------------------")
print(latest[["underlying", "strike", "F", "option_price", "iv", "nd2", "pm_prob"]]
      .sort_values(["underlying", "strike"])
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
