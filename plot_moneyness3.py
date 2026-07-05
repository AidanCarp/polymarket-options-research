import matplotlib.pyplot as plt
import numpy as np

gc_strikes = [4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000, 6200, 6500, 7000, 8000]
gc_delta = [-0.0985, -0.0829, -0.0839, -0.1064, -0.1033, -0.1151, -0.1007, -0.0837, -0.0774, -0.0897, -0.0628, -0.0685]
gc_ci_lo = [-0.1176, -0.0996, -0.1022, -0.1282, -0.1244, -0.1418, -0.1253, -0.1044, -0.0984, -0.1124, -0.0783, -0.0864]
gc_ci_hi = [-0.0795, -0.0662, -0.0655, -0.0845, -0.0822, -0.0885, -0.0762, -0.0631, -0.0564, -0.0670, -0.0473, -0.0506]

si_strikes = [60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 140]
si_delta = [-0.0668, -0.0483, -0.0575, -0.0700, -0.0379, -0.0432, -0.0497, -0.0427, -0.0655, -0.0610, -0.0530, -0.0331]
si_ci_lo = [-0.0817, -0.0715, -0.0769, -0.0874, -0.0535, -0.0587, -0.0667, -0.0597, -0.0844, -0.0767, -0.0702, -0.0491]
si_ci_hi = [-0.0519, -0.0250, -0.0382, -0.0526, -0.0224, -0.0277, -0.0327, -0.0256, -0.0467, -0.0453, -0.0358, -0.0172]

gc_f_mean = 4700
si_f_mean = 77

gc_moneyness = [k / gc_f_mean for k in gc_strikes]
si_moneyness = [k / si_f_mean for k in si_strikes]

# Flip to absolute values
gc_abs = np.abs(gc_delta)
gc_abs_lo = np.abs(gc_ci_hi)  # note: flip because ci_hi is least negative = smallest abs
gc_abs_hi = np.abs(gc_ci_lo)  # ci_lo is most negative = largest abs

si_abs = np.abs(si_delta)
si_abs_lo = np.abs(si_ci_hi)
si_abs_hi = np.abs(si_ci_lo)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
fig.subplots_adjust(wspace=0.35)

ax1.fill_between(gc_moneyness, gc_abs_lo, gc_abs_hi, alpha=0.15, color='#1a1a1a')
ax1.plot(gc_moneyness, gc_abs, color='#1a1a1a', linewidth=1.4, marker='o', markersize=4, zorder=3)
ax1.set_xlabel('Moneyness (strike / $F_0$)', fontsize=8)
ax1.set_ylabel('Mean $|\\Delta|$ (Polymarket $-$ N($d_2$))', fontsize=8)
ax1.set_title('Gold (GCQ26)', fontsize=9, fontweight='bold')
ax1.tick_params(labelsize=7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(bottom=0, top=0.17)

ax2.fill_between(si_moneyness, si_abs_lo, si_abs_hi, alpha=0.15, color='#888888')
ax2.plot(si_moneyness, si_abs, color='#888888', linewidth=1.4, marker='o', markersize=4, zorder=3)
ax2.set_xlabel('Moneyness (strike / $F_0$)', fontsize=8)
ax2.set_title('Silver (SIU26)', fontsize=9, fontweight='bold')
ax2.tick_params(labelsize=7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(bottom=0, top=0.17)

import os as _os
_out = _os.path.dirname(_os.path.abspath(__file__))
plt.savefig(_os.path.join(_out, 'moneyness_decomposition.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(_os.path.join(_out, 'moneyness_decomposition.png'), bbox_inches='tight', dpi=300)
print("Saved")
