import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DATA_DIR, "merged_iv_polymarket.csv"))

fig, ax = plt.subplots(figsize=(5, 5))

# GOLD_COLOR = "#B8860B"  SILVER_COLOR = "#4A6FA5"
for asset, color, label in [
    ("GC", "#B8860B", "Gold (GCQ26)"),
    ("SI", "#4A6FA5", "Silver (SIU26)"),
]:
    sub = df[df["underlying"] == asset]
    ax.scatter(sub["nd2"], sub["pm_prob"],
               color=color, alpha=0.25, s=6, linewidths=0,
               label=f"{label} ($n={len(sub):,}$)", zorder=3)

# 45-degree identity line
lims = [0, 1]
ax.plot(lims, lims, linestyle="--", color="#aaaaaa", linewidth=1.0,
        label="45° identity", zorder=2)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

ax.set_xlabel("Black-76 $N(d_2)$", fontsize=8)
ax.set_ylabel("Polymarket probability", fontsize=8)
ax.set_title("$N(d_2)$ vs Polymarket probability", fontsize=9, fontweight="bold")
ax.tick_params(labelsize=7)

ax.legend(fontsize=7, frameon=False, loc="upper left")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out_pdf = os.path.join(DATA_DIR, "nd2_vs_polymarket.pdf")
out_png = os.path.join(DATA_DIR, "nd2_vs_polymarket.png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Saved:\n  {out_pdf}\n  {out_png}")
