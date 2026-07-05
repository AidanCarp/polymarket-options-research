import os, re, sys
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import requests

warnings.filterwarnings("ignore")

DATA_DIR           = os.path.dirname(os.path.abspath(__file__))
EXPIRY_SI          = datetime(2026, 9, 25)   # SIU26 last trading day (4th-to-last BD of Sep 2026; Sep 7=Labor Day)
EXPIRY_GC          = datetime(2026, 7, 28)   # GCQ26 last trading day (4th-to-last BD of Jul 2026; Jul 3=Independence Day observed)
RISK_FREE          = 0.045
GC_BASIS_CUTOFF    = pd.Timestamp("2026-05-29")  # GCM26 -> GCQ26 roll date
PM_RESOLUTION_DATE = pd.Timestamp("2026-06-30")  # Polymarket markets resolve on this date

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
    files   = os.listdir(data_dir)
    opt_si  = sorted(f for f in files if re.match(r"siu\d+_\d+c_price-history", f))
    opt_gc  = sorted(f for f in files if re.match(r"gcq\d+_\d+c_price-history", f))
    und_si  = next((f for f in files if re.match(r"siu\d+_daily_historical", f)), None)
    und_gc  = next((f for f in files if re.match(r"gcq\d+_daily_historical", f)), None)
    und_gcm = next((f for f in files if re.match(r"gcm\d+_daily_historical", f)), None)
    print("\nSI option files :", opt_si)
    print("GC option files :", opt_gc)
    print("SI underlying   :", und_si)
    print("GC underlying   :", und_gc)
    print("GCM26 underlying:", und_gcm)
    return opt_si, opt_gc, und_si, und_gc, und_gcm

def parse_si_strike(filename):
    m = re.search(r"siu\d+_(\d+)c_", filename)
    if not m:
        return None
    nominal = int(m.group(1))
    if nominal < 2000:
        nominal *= 10            # Barchart drops trailing zero for $100+ strikes (1000->10000, 1100->11000, …)
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
opt_si_files, opt_gc_files, und_si_file, und_gc_file, und_gcm_file = discover_files(DATA_DIR)

si_spot = (load_barchart_csv(os.path.join(DATA_DIR, und_si_file))
           [["Time", "Latest"]].rename(columns={"Latest": "F"}))
gc_spot = (load_barchart_csv(os.path.join(DATA_DIR, und_gc_file))
           [["Time", "Latest"]].rename(columns={"Latest": "F"}))

print(f"\nSI spot: {len(si_spot)} days  |  latest F = ${si_spot['F'].iloc[-1]:.3f}/oz")
print(f"GC spot: {len(gc_spot)} days  |  latest F = ${gc_spot['F'].iloc[-1]:.2f}/oz")

# ── Gold basis adjustment (pre-roll: substitute GCM26 price for GCQ26) ────────
# For dates before GC_BASIS_CUTOFF, Polymarket resolves on GCM26 (June 2026)
# settlement. We correct the Black-76 forward: F_adj = F_GCQ26 - basis,
# where basis = GCQ26 - GCM26, which equals F_GCM26.
if und_gcm_file:
    gcm_spot = (load_barchart_csv(os.path.join(DATA_DIR, und_gcm_file))
                [["Time", "Latest"]].rename(columns={"Latest": "F_M26"}))
    gc_adj = gc_spot.merge(gcm_spot, on="Time", how="left")
    basis  = gc_adj["F"] - gc_adj["F_M26"]
    n_adj  = (gc_adj["Time"] < GC_BASIS_CUTOFF).sum()
    print(f"\nGC basis adjustment: {n_adj} pre-{GC_BASIS_CUTOFF.date()} rows adjusted")
    print(f"  Basis range (GCQ26 - GCM26): ${basis.dropna().min():.2f} - ${basis.dropna().max():.2f}")
    gc_adj["F"] = np.where(
        gc_adj["Time"] < GC_BASIS_CUTOFF,
        gc_adj["F_M26"],
        gc_adj["F"],
    )
    gc_spot_adj = gc_adj[["Time", "F"]].dropna(subset=["F"])
else:
    print("\n[warn] GCM26 file not found -- Gold basis adjustment skipped")
    gc_spot_adj = gc_spot

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
            T   = (PM_RESOLUTION_DATE.to_pydatetime() - row["Time"].to_pydatetime()).days / 365.0
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
gc_opts = process_options(opt_gc_files, parse_gc_strike, gc_spot_adj, EXPIRY_GC, "GC")
options_df = pd.concat([si_opts, gc_opts], ignore_index=True)
print(f"\nTotal option rows: {len(options_df)}")

# ── fetch Polymarket ───────────────────────────────────────────────────────────
SI_MARKETS = {
    140: "107815291782368793318738647627481517343900945506454291650622064328756176506703",
    120: "34181493756413934965830606089571434442450447113141432931484510318729119136065",
    110: "23795636223281663426103142884087722365584501611082747992219968008455523264820",
    100: "54260098841642181939004878810404149178750287936719634666289443216010972822896",
    95:  "20373780355039587946947921584212852591377751064902537354405479497879267330053",
    90:  "105262476612663704758948158538809234993782885966448251464212857276110908305869",
    85:  "109307690708325298695637158462270155889332008779910257934342711645845653510967",
    80:  "111843114693857390983486068842086902975393052463018785788470804197607748709084",
    75:  "64421299184851764888931054566640386328496621371620863500638358820445483634606",
    70:  "11352502407822932465033186884283716468972859806612400377285769823372235577870",
    65:  "83255198284961021522744975502577749660289761462119244630666928834550912637321",
    60:  "90387383267944296392171062355757876389905759723509196749861274650078762605424",
}
GC_MARKETS = {
    8000: "81311441313488227081578039705938570474171957374812448996962559052662169298751",
    7000: "71439269605198381086482578023891435827307721006677676774267334734011524437718",
    6500: "88796594327131239115152846681586783256340064802729531845476340242387373025184",
    6200: "96630671650447323341356569618316505523162316834650175610156382881851874800368",
    6000: "19398808305088983878604623316121536816822492384737303857970713820305334788727",
    5800: "3823827057212626071130170215091635673229111960416675747586832177811876926195",
    5600: "42899079046647064062817190717107037393692217495061216188592762055980279721828",
    5400: "90671288097995649082455223709734281568854849783762093482783802318774275280508",
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
pm_dedup = pm_df.sort_values("date").drop_duplicates(
    subset=["date", "underlying", "strike"], keep="last"
)
merged = options_df.merge(pm_dedup, on=["date", "underlying", "strike"], how="inner")

# Drop post-resolution rows: Polymarket emits 0/1 prices after settlement
# and GCQ26 options continue trading through July 28, so without this guard
# a re-run after resolution would silently pollute every downstream analysis.
merged = merged[pd.to_datetime(merged["date"]) < PM_RESOLUTION_DATE].copy()

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

# ── delta summary statistics ───────────────────────────────────────────────────
merged["delta"] = merged["nd2"] - merged["pm_prob"]
print(f"\n-- Delta summary (nd2 - pm_prob) by asset --")
print(merged.groupby("underlying")["delta"]
      .describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
      .to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\n-- Delta summary (nd2 - pm_prob) by asset and strike --")
print(merged.groupby(["underlying", "strike"])["delta"]
      .describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
      .to_string(float_format=lambda x: f"{x:.4f}"))
