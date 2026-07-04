import matplotlib.pyplot as plt
import numpy as np

gc_strikes = [4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000, 6200, 6500, 7000, 8000]
gc_delta = [-0.0991, -0.0831, -0.0839, -0.1062, -0.1032, -0.1150, -0.1006, -0.0836, -0.0773, -0.0896, -0.0628, -0.0685]
gc_ci_lo = [-0.1182, -0.0998, -0.1022, -0.1281, -0.1243, -0.1416, -0.1252, -0.1042, -0.0983, -0.1123, -0.0783, -0.0864]
gc_ci_hi = [-0.0800, -0.0664, -0.0655, -0.0844, -0.0821, -0.0884, -0.0760, -0.0630, -0.0563, -0.0669, -0.0473, -0.0506]

si_strikes = [60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 140]
si_delta = [-0.0784, -0.0555, -0.0617, -0.0722, -0.0391, -0.0437, -0.0498, -0.0426, -0.0653, -0.0606, -0.0526, -0.0328]
si_ci_lo = [-0.0930, -0.0785, -0.0809, -0.0895, -0.0545, -0.0593, -0.0669, -0.0597, -0.0842, -0.0763, -0.0698, -0.0488]
si_ci_hi = [-0.0638, -0.0326, -0.0425, -0.0550, -0.0236, -0.0282, -0.0328, -0.0255, -0.0464, -0.0449, -0.0354, -0.0169]

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

plt.savefig('/mnt/user-data/outputs/moneyness_decomposition.pdf', bbox_inches='tight', dpi=300)
plt.savefig('/mnt/user-data/outputs/moneyness_decomposition.png', bbox_inches='tight', dpi=300)
print("Saved")
