import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from svlib import BASE, SP, VIEWS, bootstrap_counts, ci

# ── Settings ─────────────────────────────────────────────────────────────────
# Where the attention goes along the body, per crop size: clean images of fold 0 (sham model, the
# only run whose evaluation kept the attention weights), 10 crop draws (variance_crop.py).
# Images have the head on the left and the tail on the right.
PRED = f"{BASE}/sham_fold0/eval/predictions_variance.pkl"
N_BINS = 5  # body split head -> tail
SIZES = (128, 224)
COLOR = {128: "#0072B2", 224: "#D55E00"}

# ── Crop positions ───────────────────────────────────────────────────────────
d = pd.read_pickle(PRED)[["stem", "family", "meta", "attention"]]
A, M = np.stack(d.attention.to_numpy()), np.stack(d.meta.to_numpy())
size, cx = M[:, :, 0], M[:, :, 2] + M[:, :, 0] / 2
# position along the body: crop centre between the leftmost and rightmost crop centres of that image
# (over all draws and both sizes), 0 = head end, 1 = tail end
span = pd.DataFrame({"stem": np.repeat(d.stem.to_numpy(), cx.shape[1]), "cx": cx.ravel()}).groupby("stem").cx.agg(["min", "max"])
lo, hi = (span.loc[d.stem, k].to_numpy()[:, None] for k in ("min", "max"))
pos = np.minimum(((cx - lo) / np.maximum(hi - lo, 1) * N_BINS).astype(int), N_BINS - 1)

crops = pd.DataFrame({"family": np.repeat(d.family.to_numpy(), cx.shape[1]),
                      "view": np.repeat(d.stem.str.split("_").str[4].to_numpy(), cx.shape[1]),
                      "size": size.ravel(), "bin": pos.ravel(), "att": A.ravel()})
fams = sorted(crops.family.unique())
crops["fidx"] = crops.family.map({f: i for i, f in enumerate(fams)})
counts = bootstrap_counts(fams).astype(np.float64)  # [B, families]

# ── Attention per position, relative to the mean of that crop size ───────────
# ratio = mean attention of the crops in that bin / mean attention of all crops of that size;
# CI from resampling whole families (sums and counts per family, reweighted)
rows = []
for view in VIEWS + ("all",):
    sub = crops if view == "all" else crops[crops.view == view]
    for s in SIZES:
        c = sub[sub["size"] == s]
        tab = c.groupby(["fidx", "bin"]).att.agg(["sum", "count"]).unstack(fill_value=0).reindex(range(len(fams)), fill_value=0)
        att_sum, n = tab["sum"].to_numpy(), tab["count"].to_numpy().astype(float)  # [families, bins]
        w_sum, w_n = counts @ att_sum, counts @ n                                  # [B, bins]
        ratio_b = (w_sum / w_n) / (w_sum.sum(1) / w_n.sum(1))[:, None]
        ratio = (att_sum.sum(0) / n.sum(0)) / (att_sum.sum() / n.sum())
        lo_b, hi_b = ci(ratio_b.T)
        for b in range(N_BINS):
            rows.append({"view": view, "crop_size": s, "bin": b, "position": f"{b / N_BINS:.1f}-{(b + 1) / N_BINS:.1f}",
                         "att_ratio": ratio[b], "att_ratio_lo": lo_b[b], "att_ratio_hi": hi_b[b],
                         "share_of_crops": n.sum(0)[b] / n.sum()})
out = pd.DataFrame(rows)
out.to_csv(f"{SP}/attention_position.csv", index=False, float_format="%.4f")
print(out[out.view == "all"].to_string(index=False, float_format="%.3f"))

# ── Figure: attention along the body (left), where the crops fall (right) ────
x = (np.arange(N_BINS) + 0.5) / N_BINS
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), width_ratios=[1.6, 1], layout="constrained")
for s in SIZES:
    r = out[(out.view == "all") & (out.crop_size == s)]
    a.fill_between(x, r.att_ratio_lo, r.att_ratio_hi, color=COLOR[s], alpha=0.2, lw=0)
    a.plot(x, r.att_ratio, "o-", color=COLOR[s], lw=2, ms=6, label=f"{s} px crops")
    for v, ls in zip(VIEWS, (":", "--", "-.")):
        rv = out[(out.view == v) & (out.crop_size == s)]
        a.plot(x, rv.att_ratio, ls=ls, color=COLOR[s], lw=0.9, alpha=0.7)
a.axhline(1, color="#888888", lw=0.8)
a.set_xticks([0, 0.5, 1], ["head", "middle", "tail"])
a.set_ylabel("attention / mean of that crop size")
a.set_title("A. Attention along the body (band: 95% CI over families)", fontsize=10, loc="left")
views = [plt.Line2D([], [], color="#666666", lw=0.9, ls=ls, label=v) for v, ls in zip(VIEWS, (":", "--", "-."))]
a.legend(handles=a.get_legend_handles_labels()[0] + views, frameon=False, fontsize=9, loc="upper center", ncol=2)
w = 0.38 / N_BINS
for k, s in enumerate(SIZES):
    r = out[(out.view == "all") & (out.crop_size == s)]
    b.bar(x + (k - 0.5) * w, r.share_of_crops, w, color=COLOR[s], label=f"{s} px")
b.set_xticks([0, 0.5, 1], ["head", "middle", "tail"])
b.set_ylabel("share of the crops of that size")
b.set_title("B. Where the crops fall", fontsize=10, loc="left")
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(f"{SP}/attention_position.png", dpi=130)
plt.close(fig)
print("wrote attention_position.csv, attention_position.png")
