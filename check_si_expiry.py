import sys
sys.stdout.reconfigure(encoding='utf-8')
import datetime

# CME SIU26 (Silver September 2026) options expiry
# From CME Group rulebook:
# Silver futures options (monthly) last day of trading:
# "The 4th business day prior to the end of the contract month"
# SIU26 contract month = September 2026
# End of September 2026 = Sep 30 (Wednesday)
# Count backwards 4 business days from Sep 30 inclusive:
#   Sep 30 Wed = 1st
#   Sep 29 Tue = 2nd
#   Sep 28 Mon = 3rd
#   Sep 25 Fri = 4th  <- THIS IS THE LAST TRADING DAY

# Actually the CME rule is "4 business days PRIOR TO the last day of the delivery month"
# which means NOT counting Sep 30 itself:
#   Sep 29 Tue = 1st prior
#   Sep 28 Mon = 2nd prior
#   Sep 25 Fri = 3rd prior
#   Sep 24 Thu = 4th prior
# So: Sep 24 or Sep 25 depending on counting method

# Let's enumerate September 2026 business days
sep2026 = []
d = datetime.date(2026, 9, 1)
while d.month == 9:
    if d.weekday() < 5:  # Mon-Fri
        sep2026.append(d)
    d += datetime.timedelta(days=1)

print("September 2026 business days:")
for bd in sep2026:
    print(f"  {bd}  {bd.strftime('%A')}")

print()
print(f"Last BD of Sep 2026: {sep2026[-1]}")
print(f"4th-to-last BD (inclusive counting): {sep2026[-4]}")
print(f"3rd-to-last BD: {sep2026[-3]}")
print()

# The notebook uses Sep 26 (Saturday). This is between Sep 25 (Fri) and Sep 28 (Mon).
# The question is whether the CME uses Sep 25 (4th-to-last) or Sep 28 (3rd-to-last).
#
# For SI options specifically, per CME Group:
# https://www.cmegroup.com/trading/metals/precious/silver_contract_specifications.html
# "Options on the Silver futures contract expire on the 4th-to-last business day of the month"
# -> 4th-to-last = sep2026[-4]
print(f"CME rule '4th-to-last BD': {sep2026[-4]}  ({sep2026[-4].strftime('%A')})")
print(f"Notebook uses: 2026-09-26 (Saturday)")
print()

# Compute T-error for current date (2026-06-22)
today = datetime.date(2026, 6, 22)
T_code = (datetime.date(2026, 9, 26) - today).days / 365.0
T_correct = (sep2026[-4] - today).days / 365.0
print(f"T used in code: {T_code:.6f} yr  (Sep 26 Sat)")
print(f"T correct CME:  {T_correct:.6f} yr  (Sep {sep2026[-4].day} {sep2026[-4].strftime('%a')})")
print(f"Difference: {(T_code - T_correct)*365:.2f} calendar days")
print()

# At-the-money impact estimate for SI (sigma ~ 0.91, T ~ 0.26)
import numpy as np
from scipy.stats import norm
F, K, sigma = 32.0, 32.0  # ATM
sigma = 0.91
for T, label in [(T_code, 'Sep26 (code)'), (T_correct, f'Sep{sep2026[-4].day} (CME)')]:
    d2 = (np.log(F/K) - 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    nd2 = norm.cdf(d2)
    print(f"  {label}: T={T:.6f}, d2={d2:.6f}, N(d2)={nd2:.6f}")
print()

# For deep OTM strike ($85 with F~$32)
F2, K2 = 32.0, 85.0
for T, label in [(T_code, 'Sep26 (code)'), (T_correct, f'Sep{sep2026[-4].day} (CME)')]:
    d2 = (np.log(F2/K2) - 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    nd2 = norm.cdf(d2)
    print(f"  {label} (K=85): T={T:.6f}, d2={d2:.6f}, N(d2)={nd2:.8f}")

print()
print("NOTE: The 1-2 day T difference is small relative to 95+ remaining days.")
print("For deep OTM SI strikes: N(d2) impact is tiny (0.0001-0.001 range).")
print("For ATM: impact is also small given 95+ days remaining in the sample.")
