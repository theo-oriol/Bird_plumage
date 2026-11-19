import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from svlib import SP, NAMES, load, arrays

# ── Settings ─────────────────────────────────────────────────────────────────
# Exploration, not in the report: the PCA of habitat_gradient.py (main model, clean images) with the
# outputs under every ablation except the single-colour ones (A6) projected on the same axes, to see
# which ablations keep the habitat structure and which collapse it. A3: first random seed only.
MODEL = "main"
CONDS = ["reference", "A1", "A2", "A5", "A7"] + [f"A4_theta{t}_L{l}" for t in (0, 90) for l in (17, 41, 101)] \
        + [f"A3_h{h}_r1" for h in (56, 16, 4, 1)]
TITLE = {"reference": "Clean", "A1": "A1 greyscale", "A2": "A2 greyscale, same lightness", "A5": "A5 Fourier slope",
         "A7": "A7 colour only (no light/dark)",
         **{f"A4_theta{t}_L{l}": f"A4 blur {'along' if t == 0 else 'across'} body, L={l}" for t in (0, 90) for l in (17, 41, 101)},
         **{f"A3_h{h}_r1": f"A3 shuffle {h} px blocks" for h in (56, 16, 4, 1)}}
# as in habitat_gradient.py
COLS = [0, 1, 2, 3, 4, 6, 7, 8, 9]
N_GROUPS = 5
SHORT = ["Forest", "Savanna", "Shrubland", "Grassland", "Desert", "Artificial-Terr.", "Neritic", "Wetlands", "Rocky Areas"]
COLOR = ["#009E73", "#E69F00", "#CC79A7", "#B8A800", "#D55E00", "#888888", "#0072B2", "#56B4E9", "#222222"]


def groups_of(centres):
    lab = fcluster(linkage(centres, "ward"), N_GROUPS, "maxclust")
    return sorted((list(np.flatnonzero(lab == g)) for g in np.unique(lab)), key=min)


# ── PCA fitted on the clean outputs, every condition projected on it ─────────
d = load(MODEL, CONDS)
pipe = make_pipeline(StandardScaler(), PCA())
proj, out = {}, {}
for cond in CONDS:  # reference first: the PCA is fitted on it
    sub = d[d.condition_name == cond]
    H, S, _ = arrays(sub)
    logS = np.log(S[:, COLS] + 1e-6)
    Y = pipe.fit_transform(logS) if cond == "reference" else pipe.transform(logS)
    main = H[:, COLS].argmax(1)
    centres = np.array([Y[main == h].mean(0) for h in range(len(COLS))])
    proj[cond] = (Y, centres, sub.species.to_numpy(), main)
    # spread = total variance on all clean axes, as a share of the clean images' spread
    out[cond] = {"spread": float(Y.var(0).sum() / proj["reference"][0].var(0).sum()),
                 "spread_2axes": float(Y[:, :2].var(0).sum() / proj["reference"][0][:, :2].var(0).sum()),
                 "groups": [[NAMES[COLS[h]] for h in g] for g in groups_of(centres)]}
    print(f"{cond:16s} spread {out[cond]['spread']:.2f}  groups: "
          + " | ".join(" + ".join(SHORT[h] for h in g) for g in groups_of(centres)), flush=True)

# ── Figure: one panel per condition, same axes and limits ────────────────────
Yr = proj["reference"][0]
lim = np.percentile(Yr[:, :2], [0.5, 99.5], axis=0)
pad = 0.05 * (lim[1] - lim[0])
fig, axes = plt.subplots(3, 5, figsize=(22, 13), layout="constrained")
for a, cond in zip(axes.flat, CONDS):
    Y, centres, species, main = proj[cond]
    sp = pd.DataFrame(Y[:, :2]).groupby([species, main]).mean()
    for h in range(len(COLS)):
        pts = sp.xs(h, level=1).to_numpy()
        a.scatter(pts[:, 0], pts[:, 1], s=4, c=COLOR[h], alpha=0.2, lw=0)
    # hollow rings = clean habitat centres, with a line to where each centre moves under the ablation
    c0 = proj["reference"][1][:, :2]
    for h in range(len(COLS)):
        a.plot([c0[h, 0], centres[h, 0]], [c0[h, 1], centres[h, 1]], color=COLOR[h], lw=1)
    a.scatter(c0[:, 0], c0[:, 1], s=40, facecolors="none", edgecolors=COLOR, lw=1.2, zorder=3)
    a.scatter(centres[:, 0], centres[:, 1], s=70, c=COLOR, edgecolors="white", lw=1.2, zorder=4)
    a.set_xlim(lim[0, 0] - pad[0], lim[1, 0] + pad[0])
    a.set_ylim(lim[0, 1] - pad[1], lim[1, 1] + pad[1])
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(f"{TITLE[cond]}\nspread {100 * out[cond]['spread']:.0f}% of clean", fontsize=10, loc="left")
handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[h], label=SHORT[h]) for h in range(len(COLS))]
fig.legend(handles=handles, loc="outside lower center", ncol=len(COLS), frameon=False, fontsize=10)
fig.suptitle(f"{MODEL} model: outputs under each ablation on the clean PCA axes (axis 1 horizontal, axis 2 vertical); "
             "rings = clean habitat centres", fontsize=12)
fig.savefig(f"{SP}/habitat_gradient_ablations_{MODEL}.png", dpi=110)
plt.close(fig)

json.dump(out, open(f"{SP}/habitat_gradient_ablations_{MODEL}.json", "w"), indent=1)
print(f"wrote habitat_gradient_ablations_{MODEL}.{{png,json}}")
