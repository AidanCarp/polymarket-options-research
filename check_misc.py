import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.stats import norm

# ---- Check: collect_daily.py idempotency logic ----
# existing_dates() reads the CSV and builds a set of dates already there.
# Then for each new row, if row_date in done: skip.
# Issue: 'done' is a set of dates (strings) already in the file.
# But 'done' is built ONCE at the start of collect_polymarket().
# If multiple strikes are fetched in the same run and share a date,
# the first row with that date gets written, then subsequent rows
# with the same date for different strikes would NOT be in 'done'
# (since done doesn't update as rows are appended in-memory).
# BUT each (date, underlying, strike) tuple is unique in the CSV,
# so 'done' only needs to guard against re-running on the same day.
# 'done' is a set of dates, not (date, strike) tuples.
# This means: if we run once on date D and write SI $60 successfully,
# then re-run, we'd skip ALL SI strikes (done={D}) even if some failed.
# However, collecting ALL strikes in one run is the intended usage.
# The real risk: partial failure -> can't retry individual strikes.
# This is a MINOR operational concern, not a mathematical bug.
print("collect_daily.py idempotency: guards by date only (not date+strike).")
print("Minor: can't retry individual failed strikes without deleting the date.")
print()

# ---- collect_daily.py: Polymarket uses interval='1d' not 'max' ----
# In the notebook/script, fetch_polymarket uses interval='max' to get all history.
# In collect_daily.py, it uses interval='1d'.
# interval='1d' returns only recent data (typically last 1 day).
# This is intentional for collect_daily.py (just the latest point).
# But fidelity=1440 (minutes per candle = 1440 = daily) is consistent in both.
print("collect_daily.py: interval='1d' (vs 'max' in notebook). INTENTIONAL for daily collector.")
print()

# ---- check: expiry date for SIU26 ----
# SIU26 = Silver September 2026 contract
# Silver futures (SI) on COMEX: last trading day = 3rd-to-last business day of delivery month
# September 2026: business days in last week:
#   Sep 2026: Sep 28 is Mon, Sep 29 Tue, Sep 30 Wed
#   3rd-to-last = Sep 28? Let's count from end:
#   Sep 30 (Wed) = last BD, Sep 29 (Tue) = 2nd-to-last, Sep 28 (Mon) = 3rd-to-last
# WAIT: The code uses Sep 26 (Saturday). That can't be right - Sep 26 2026 is a Saturday.
import datetime
d = datetime.date(2026, 9, 26)
print(f"EXPIRY_SI = 2026-09-26, which is a: {d.strftime('%A')}")
# Let's check the actual CME rule: "third to last business day of delivery month"
# September 2026 calendar:
# Sep 28 Mon, Sep 29 Tue, Sep 30 Wed are the last 3 business days
# -> last trading day should be Sep 28, 2026 (Mon) not Sep 26 (Sat)
# OR, for SI, it might be 1 business day before first notice day.
# CME SI specs: "Business day prior to the 15th calendar day of the contract month"
# for NOTICE day, and options expire... actually SI options expire differently.
# SI futures options: "Last day of trading is the 4th-to-last business day of the month"
# September 2026 business days in last week: Sep 28, 29, 30 (3 BDs)
# Extending: Sep 25 Fri = 4th-to-last BD.
# Sep 26 is Saturday -- NOT a business day.
# Possible: the notebook author may have meant Sep 25 (Fri) or Sep 28 (Mon).
# Sep 26 being a Saturday is almost certainly wrong.
print(f"Sep 2026 calendar (last days): Mon=28, Tue=29, Wed=30")
print(f"Sep 25 is: {datetime.date(2026, 9, 25).strftime('%A')} (likely 4th-to-last BD)")
print(f"Sep 26 is: {datetime.date(2026, 9, 26).strftime('%A')} -- NOT a trading day!")
print()
print("POTENTIAL BUG: EXPIRY_SI = 2026-09-26 is a Saturday.")
print("CME SIU26 options last trading day is likely Sep 25 or Sep 28, 2026.")
print("This affects T calculation by 1-3 calendar days, introducing systematic error in d2.")

# Quantify the T impact
today = datetime.date(2026, 6, 22)
T_sat = (datetime.date(2026, 9, 26) - today).days / 365.0
T_fri = (datetime.date(2026, 9, 25) - today).days / 365.0
T_mon = (datetime.date(2026, 9, 28) - today).days / 365.0
print(f"T to Sep 26 (Sat): {T_sat:.6f} yr")
print(f"T to Sep 25 (Fri): {T_fri:.6f} yr  (diff = {(T_sat-T_fri)*365:.1f} calendar days)")
print(f"T to Sep 28 (Mon): {T_mon:.6f} yr  (diff = {(T_mon-T_sat)*365:.1f} calendar days)")

# Impact on d2 at typical SI values
F, K, sigma = 32.0, 85.0, 0.91  # Silver at ~$32, strike $85, vol 91%
for T, label in [(T_sat, 'Sep 26'), (T_fri, 'Sep 25'), (T_mon, 'Sep 28')]:
    d2 = (np.log(F/K) - 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    nd2 = norm.cdf(d2)
    print(f"  T={T:.6f} ({label}): d2={d2:.6f}, N(d2)={nd2:.8f}")
print()

# ---- check: expiry date for GCQ26 ----
# GCQ26 = Gold August 2026 contract
# CME GC options: last trading day = "4th business day prior to the end of the expiration month"
# For August 2026: Aug 31 is Monday. Last 4 BDs: Aug 31, Aug 28, Aug 27, Aug 26
# 4th-to-last = Aug 26? But the code uses Jul 28.
# Wait: GCQ26 is Aug 2026 but there may be a different rule for GC options.
# Actually, GC options (monthly) expire on the 4th-last BUSINESS DAY before
# the LAST TRADING DAY OF THE FUTURES.
# GCQ26 FUTURES last trade: Jul 28, 2026 (code says this directly).
# Gold options for a given month expire before the futures. Let me check:
# Actually "GCQ26" = August 2026 gold FUTURES (Q = August in CME codes)
# CME gold futures: LTD = third-to-last business day of delivery month
# August 2026: Aug 31 Mon, Aug 28 Fri, Aug 27 Thu -> 3rd-to-last = Aug 27
# But code uses Jul 28. That's a full month before delivery.
# THIS IS ACTUALLY GC OPTIONS EXPIRY: Gold options on futures expire ~month before futures
# GC options (standard) expire on 2nd Friday of month before delivery:
# For Aug 2026 delivery: 2nd Friday of July 2026 = Jul 10? or Jul 11?
# Actually CME Gold Options: "The expiration of options contracts is the 4th business day
# before the end of the month prior to the contract month of the underlying futures"
# Contract month = August 2026, month prior = July 2026
# End of July 2026 = Jul 31 (Fri)
# 4 BDs before Jul 31: Jul 31, Jul 30, Jul 29, Jul 28 -> 4th BD before end = Jul 28 (Tue? Mon?)
d_gc = datetime.date(2026, 7, 28)
print(f"EXPIRY_GC = 2026-07-28, which is a: {d_gc.strftime('%A')}")
print("GCQ26 options: Jul 28 = Tuesday. This could be correct per CME rule.")
print("CME rule: 4 BDs before end of month prior to delivery (July 2026, end=Jul 31 Fri)")
print("Jul 31 Fri = 1st, Jul 30 Thu = 2nd, Jul 29 Wed = 3rd, Jul 28 Tue = 4th BD before end")
print("-> Jul 28, 2026 CORRECT for GCQ26 options expiry.")
print()

# Summary
print("=== EXPIRY SUMMARY ===")
print(f"EXPIRY_GC = Jul 28 (Tue): CORRECT per CME 4-BD rule")
print(f"EXPIRY_SI = Sep 26 (Sat): LIKELY WRONG - September 26, 2026 is a Saturday")
print(f"SI options probably expire Sep 25, 2026 (Fri) or Sep 28, 2026 (Mon)")
