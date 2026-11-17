import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress, mannwhitneyu
from sklearn.metrics import roc_auc_score
from svlib import SP, NH, NAMES, load, arrays, load_conditions

# ── Settings ─────────────────────────────────────────────────────────────────
# AUC of each habitat output within bins of the habitat's label share (0-0.1, ..., 0.9-1):
# positives = birds whose share falls in the bin, negatives = birds that do not use the habitat
LO = np.arange(10) / 10
CENTERS = LO + 0.05
COND_ORDER = (["reference", "A1", "A2", "A5", "A7"] + [f"A3_h{h}_r{r}" for h in (56, 16, 4, 1) for r in (1, 2, 3)]
              + [f"A4_theta{t}_L{l}" for t in (0, 90) for l in (17, 41, 101)]
              + [f"A6_{c}" for c in ("blue", "purple", "red", "yellow", "green", "rufous", "light_brown", "dark_brown")])


# ── Helpers ──────────────────────────────────────────────────────────────────
def bin_aucs(share, score):
    neg = share == 0
    out = np.full(10, np.nan)
    for j, lo in enumerate(LO):
        pos = (share > lo) & (share <= lo + 0.1)
        if pos.any():
            out[j] = roc_auc_score(pos[pos | neg], score[pos | neg])
    return out


def slope(aucs):
    ok = ~np.isnan(aucs)
    return linregress(CENTERS[ok], aucs[ok]) if ok.sum() >= 3 else None


# ── Slope heatmap (main model) ───────────────────────────────────────────────
# slope of AUC vs. label share, per condition and habitat: does the AUC rise with how much a bird uses the habitat?
data = load_conditions("main")
mat = np.array([[getattr(slope(bin_aucs(data[c]["H"][:, h], data[c]["S"][:, h])), "slope", np.nan) for h in range(NH)]
                for c in COND_ORDER])
del data
gc.collect()
fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=0.4, aspect="auto")
ax.set_xticks(range(NH), NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(COND_ORDER)), COND_ORDER, fontsize=8, family="monospace")
for i, j in zip(*np.where(~np.isnan(mat))):
    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
            color="white" if abs((mat[i, j] - 0.15) / 0.5) > 0.28 else "black")
ax.set_title("Specialization->AUC trend slope, per condition x habitat (non-cumulative) - regression head", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
plt.tight_layout()
plt.savefig(f"{SP}/summary_slope_heatmap_reg.png", dpi=130)
plt.close(fig)

# ── Reference curves (sham model, clean images) ──────────────────────────────
# green points: positives score above negatives (one-sided Mann-Whitney, p < 0.05)
H, S, _ = arrays(load("sham"))
fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
for h, ax in enumerate(axes.flat):
    share, score = H[:, h], S[:, h]
    aucs = bin_aucs(share, score)
    ax.plot(CENTERS, aucs, color=plt.get_cmap("tab10")(h), lw=1.8)
    ax.axhline(0.5, ls="--", color="grey", lw=0.8)
    ax.set_title(NAMES[h], fontsize=11)
    ax.set_xticks(CENTERS, [f"{b:.1f}" for b in LO], rotation=45, fontsize=8)
    ax.grid(alpha=0.3)
    for j, lo in enumerate(LO):
        pos = (share > lo) & (share <= lo + 0.1)
        if np.isnan(aucs[j]):
            continue
        col = "green" if mannwhitneyu(score[pos], score[share == 0], alternative="greater").pvalue < 0.05 else "gray"
        ax.plot(CENTERS[j], aucs[j], marker="o", color=col, ms=6, zorder=3)
        ax.text(CENTERS[j], aucs[j] + 0.02, f"n={pos.sum()}", ha="center", va="bottom", fontsize=6, color=col)
    lr = slope(aucs)
    if lr is not None:
        ok = ~np.isnan(aucs)
        ax.plot(CENTERS[ok], lr.intercept + lr.slope * CENTERS[ok], ls=":", color="black", lw=1.2)
        ax.text(0.03, 0.03, f"slope={lr.slope:.2f}, " + ("p<.001" if lr.pvalue < 0.001 else f"p={lr.pvalue:.3f}"),
                transform=ax.transAxes, fontsize=8, color="green" if lr.pvalue < 0.05 else "gray")
for ax in axes[:, 0]:
    ax.set_ylabel("AUC")
for ax in axes[-1, :]:
    ax.set_xlabel("Fractional mass bin (start)")
fig.suptitle(f"AUC by habitat-mass bin — reference (non-cumulative), sham model, regression head, n={len(H)}", fontsize=13)
plt.tight_layout()
plt.savefig(f"{SP}/reference_noncum_reg_sham.png", dpi=120)
plt.close(fig)
print("wrote summary_slope_heatmap_reg.png, reference_noncum_reg_sham.png")
