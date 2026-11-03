import os
import gc
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress, mannwhitneyu
from sklearn.metrics import roc_auc_score

from paths import RUNS_DIR, LABELS_JSON

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = RUNS_DIR
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
hab_names = json.load(open(LABELS_JSON))

bins = np.arange(0, 10) / 10
centers = bins + 0.05


# ── Helpers ──────────────────────────────────────────────────────────────────
def per_habitat_auc_table(df):
    out = {}
    for cond in sorted(df["condition_name"].unique()):
        sub = df[df["condition_name"] == cond]
        support = (np.stack(sub["habitat"].to_numpy()) > 0).astype(float)
        prob = np.stack(sub["reg_prob"].to_numpy())
        aucs = []
        for c in range(support.shape[1]):
            col = support[:, c]
            aucs.append(None if col.min() == col.max() else float(roc_auc_score(col, prob[:, c])))
        out[cond] = {"n": int(len(sub)), "auc_per_habitat": aucs}
        print("  table:", cond, "done", flush=True)
    return out


# ── Section A: main model ────────────────────────────────────────────────────
print("=== main: loading ===", flush=True)
with open(f"{BASE}/main_fold0/eval/predictions.pkl", "rb") as f:
    d0 = pickle.load(f)
with open(f"{BASE}/main_fold1/eval/predictions.pkl", "rb") as f:
    d1 = pickle.load(f)
with open(f"{BASE}/main_fold2/eval/predictions.pkl", "rb") as f:
    d2 = pickle.load(f)
df_main = pd.concat((d0, d1, d2), ignore_index=True)
del d0, d1, d2

main_table = per_habitat_auc_table(df_main)

ref_df = df_main[df_main["condition_name"] == "reference"].reset_index(drop=True)
gt_full = np.stack(ref_df["habitat"].to_numpy())
gt_bin_full = (gt_full > 0).astype(float)
stem_to_idx = {s: i for i, s in enumerate(ref_df["stem"].to_numpy())}

all_conditions = sorted(df_main["condition_name"].unique())
heatmap_results = {}
for cond in all_conditions:
    sub = df_main[df_main["condition_name"] == cond]
    pred = np.stack(sub["reg_prob"].to_numpy())
    idx = np.array([stem_to_idx[s] for s in sub["stem"].to_numpy()])
    gt_c, gt_bin_c = gt_full[idx], gt_bin_full[idx]

    slopes = np.full(10, np.nan)
    mean_auc = np.full(10, np.nan)
    for hab in range(10):
        select_gt = gt_c[:, hab]
        select_gt_bin = gt_bin_c[:, hab]
        select_pred = pred[:, hab]
        auc_bins = np.full(10, np.nan)
        for j, lo in enumerate(bins):
            hi = lo + 0.1
            in_bin = (select_gt > lo) & (select_gt <= hi)
            indice = in_bin | (select_gt_bin == 0)
            gbs, ps = select_gt_bin[indice], select_pred[indice]
            auc_bins[j] = roc_auc_score(gbs, ps) if gbs.sum() > 0 else np.nan
        ok = ~np.isnan(auc_bins)
        if ok.sum() >= 3:
            lr = linregress(centers[ok], auc_bins[ok])
            slopes[hab] = lr.slope
        mean_auc[hab] = np.nanmean(auc_bins)
    heatmap_results[cond] = {"slope": slopes.tolist(), "mean_auc": mean_auc.tolist()}
    print("  heatmap:", cond, "done", flush=True)

json.dump({"conditions": heatmap_results, "habitat_names": hab_names}, open(f"{OUT}/summary_stats_reg.json", "w"))

# ── Slope heatmap ────────────────────────────────────────────────────────────
cond_order = ["reference", "A1", "A2", "A5", "A7"] + \
    [f"A3_h{h}_r{r}" for h in (56, 16, 4, 1) for r in (1, 2, 3)] + \
    [f"A4_theta{t}_L{l}" for t in (0, 90) for l in (17, 41, 101)] + \
    [f"A6_{c}" for c in ("blue", "purple", "red", "yellow", "green", "rufous", "light_brown", "dark_brown")]

mat = np.array([heatmap_results[c]["slope"] for c in cond_order])
fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=0.4, aspect="auto")
ax.set_xticks(range(10))
ax.set_xticklabels(hab_names, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(cond_order)))
ax.set_yticklabels(cond_order, fontsize=8, family="monospace")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if np.isnan(v):
            continue
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs((v - 0.15) / 0.5) > 0.28 else "black")
ax.set_title("Specialization->AUC trend slope, per condition x habitat (non-cumulative) - regression head", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
plt.tight_layout()
plt.savefig(f"{OUT}/summary_slope_heatmap_reg.png", dpi=130)
plt.close(fig)

del df_main, ref_df, gt_full, gt_bin_full
gc.collect()
print("=== main: done, freed ===", flush=True)

# ── Section B: sham model, plain folds (table) ───────────────────────────────
print("=== sham (plain folds): loading ===", flush=True)
with open(f"{BASE}/sham_fold0/eval/predictions.pkl", "rb") as f:
    d0 = pickle.load(f)
with open(f"{BASE}/sham_fold1/eval/predictions.pkl", "rb") as f:
    d1 = pickle.load(f)
with open(f"{BASE}/sham_fold2/eval/predictions.pkl", "rb") as f:
    d2 = pickle.load(f)
df_sham = pd.concat((d0, d1, d2), ignore_index=True)
del d0, d1, d2

sham_table = per_habitat_auc_table(df_sham)

del df_sham
gc.collect()
print("=== sham (plain folds): done, freed ===", flush=True)

json.dump({"habitat_names": hab_names, "data": {"main": main_table, "sham": sham_table}},
          open(f"{OUT}/auc_per_habitat_per_condition_reg.json", "w"))
print("wrote auc_per_habitat_per_condition_reg.json")

# ── Section C: sham model, reseed_a7check fold2 (reference curve) ────────────
print("=== sham (reseed_a7check): loading ===", flush=True)
with open(f"{BASE}/sham_fold0/eval/predictions.pkl", "rb") as f:
    d0 = pickle.load(f)
with open(f"{BASE}/sham_fold1/eval/predictions.pkl", "rb") as f:
    d1 = pickle.load(f)
with open(f"{BASE}/sham_fold2_reseed_a7check/eval/predictions.pkl", "rb") as f:
    d2 = pickle.load(f)
df_sham2 = pd.concat((d0, d1, d2), ignore_index=True)
del d0, d1, d2

ref_df2 = df_sham2[df_sham2["condition_name"] == "reference"].reset_index(drop=True)
gt = np.stack(ref_df2["habitat"].to_numpy())
gt_bin = (gt > 0).astype(float)
pred_ref = np.stack(ref_df2["reg_prob"].to_numpy())
n_images = gt.shape[0]
ALPHA = 0.05
cmap = plt.get_cmap("tab10")

res = np.full((10, 10), np.nan)
pvals = np.full((10, 10), np.nan)
nb = np.zeros((10, 10))
for hab in range(10):
    select_gt = gt[:, hab]
    select_gt_bin = gt_bin[:, hab]
    select_pred = pred_ref[:, hab]
    for j, lo in enumerate(bins):
        hi = lo + 0.1
        in_bin = (select_gt > lo) & (select_gt <= hi)
        nb[hab, j] = in_bin.sum()
        indice = in_bin | (select_gt_bin == 0)
        gbs, ps = select_gt_bin[indice], select_pred[indice]
        res[hab, j] = roc_auc_score(gbs, ps) if gbs.sum() > 0 else np.nan
        pos = select_pred[in_bin & (select_gt_bin == 1)]
        neg = select_pred[select_gt_bin == 0]
        if len(pos) > 0 and len(neg) > 0:
            pvals[hab, j] = mannwhitneyu(pos, neg, alternative="greater").pvalue

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
for h, ax in enumerate(axes.flat):
    ax.plot(centers, res[h], color=cmap(h), lw=1.8)
    ax.axhline(0.5, ls="--", color="grey", lw=0.8)
    ax.set_title(hab_names[h], fontsize=11)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{b:.1f}" for b in bins], rotation=45, fontsize=8)
    ax.grid(alpha=0.3)
    for x, y, n, pv in zip(centers, res[h], nb[h], pvals[h]):
        if np.isnan(y):
            continue
        col = "green" if pv < ALPHA else "gray"
        ax.plot(x, y, marker="o", color=col, ms=6, zorder=3)
        ax.text(x, y + 0.02, f"n={int(n)}", ha="center", va="bottom", fontsize=6, color=col)
    ok = ~np.isnan(res[h])
    if ok.sum() >= 3:
        lr = linregress(centers[ok], res[h][ok])
        ax.plot(centers[ok], lr.intercept + lr.slope * centers[ok], ls=":", color="black", lw=1.2)
        lr_p = "p<.001" if lr.pvalue < 0.001 else f"p={lr.pvalue:.3f}"
        ax.text(0.03, 0.03, f"slope={lr.slope:.2f}, {lr_p}", transform=ax.transAxes, fontsize=8,
                color="green" if lr.pvalue < ALPHA else "gray")
for ax in axes[:, 0]:
    ax.set_ylabel("AUC")
for ax in axes[-1, :]:
    ax.set_xlabel("Fractional mass bin (start)")
fig.suptitle(f"AUC by habitat-mass bin — reference (non-cumulative), sham model, regression head, n={n_images}", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/reference_noncum_reg_sham.png", dpi=120)
plt.close(fig)

del df_sham2, ref_df2
gc.collect()
print("=== sham (reseed_a7check): done, freed ===", flush=True)
print("ALL DONE")
