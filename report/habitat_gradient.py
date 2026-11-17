import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from svlib import SP, NAMES, load, arrays

# ── Settings ─────────────────────────────────────────────────────────────────
# Habitat structure in the model's outputs on clean images: PCA of the log outputs
# (each habitat standardised) and groups of habitats from their centres in that space;
# the labels' PCA variance and the habitat correlations are saved for the text.
COLS = [0, 1, 2, 3, 4, 6, 7, 8, 9]  # Marine Intertidal dropped: almost never a main habitat
N_GROUPS = 5
SHORT = ["Forest", "Savanna", "Shrubland", "Grassland", "Desert", "Artificial-Terr.", "Neritic", "Wetlands", "Rocky Areas"]
# Okabe-Ito colours (colour-blind safe) + grey; every habitat centre is also labelled by name
COLOR = ["#009E73", "#E69F00", "#CC79A7", "#B8A800", "#D55E00", "#888888", "#0072B2", "#56B4E9", "#222222"]


# ── Helpers ──────────────────────────────────────────────────────────────────
def clusters(points):
    # Ward clustering into N_GROUPS; groups listed in habitat order (Forest first)
    lab = fcluster(linkage(points, "ward"), N_GROUPS, "maxclust")
    return sorted((list(np.flatnonzero(lab == g)) for g in np.unique(lab)), key=min)


def pca(X):
    pipe = make_pipeline(StandardScaler(), PCA())
    return pipe.fit_transform(X), pipe[-1].explained_variance_ratio_


def label_centres(ax, pts):
    # label each centre at the first free spot around it (16 directions, growing radius)
    ax.figure.canvas.draw()
    P = ax.transData.transform(pts)
    inv = ax.transData.inverted()
    boxes = []
    for i in np.argsort([-(np.linalg.norm(P - p, axis=1) < 60).sum() for p in P]):  # crowded first
        w, h = 7.6 * len(SHORT[i]) + 8, 15
        for r in range(14, 300, 8):
            spots = [P[i] + r * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
            spots = [(x - w if x < P[i, 0] else x, y - h / 2) for x, y in spots]
            spots = [(x, y) for x, y in spots
                     if not any(x < q[0] + q[2] and q[0] < x + w and y < q[1] + h and q[1] < y + h for q in boxes)
                     and not any(x - 7 < p[0] < x + w + 7 and y - 7 < p[1] < y + h + 7 for p in P)]
            if spots:
                break
        x, y = spots[0]
        boxes.append((x, y, w))
        ax.annotate(SHORT[i], pts[i], xytext=inv.transform((x, y + h / 2)), va="center", fontsize=10,
                    fontweight="bold", color="#1a1a1a", zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#444444", lw=0.9, shrinkA=2, shrinkB=5))


# ── Per model ────────────────────────────────────────────────────────────────
out = {"habitats": [NAMES[h] for h in COLS]}
for model in ("main", "sham"):
    d = load(model)
    H, S, _ = arrays(d)
    H, S, species = H[:, COLS], S[:, COLS], d.species.to_numpy()
    main = H.argmax(1)
    logS = np.log(S + 1e-6)
    Y, var_model = pca(logS)
    _, var_labels = pca(H)
    corr_labels, corr_model = np.corrcoef(H.T), np.corrcoef(logS.T)
    no_forest = np.triu_indices(len(COLS) - 1, 1)
    # groups of habitats: clustering of the habitat centres on the two axes shown in the figure,
    # the clustering on all axes kept as a check
    centres = np.array([Y[main == h].mean(0) for h in range(len(COLS))])
    groups = clusters(centres[:, :2])
    out[model] = {
        "var_labels": var_labels.tolist(), "var_model": var_model.tolist(),
        "corr_labels": corr_labels.tolist(), "corr_model": corr_model.tolist(),
        "r_corr_noforest": float(np.corrcoef(corr_labels[1:, 1:][no_forest], corr_model[1:, 1:][no_forest])[0, 1]),
        "groups": [[NAMES[COLS[h]] for h in g] for g in groups],
        "merge_heights": linkage(centres[:, :2], "ward")[:, 2].tolist(),
        "groups_all_axes": [[NAMES[COLS[h]] for h in g] for g in clusters(centres)],
    }
    print(f"{model}: 2 axes = {var_labels[:2].sum():.2f} of label variance, {var_model[:2].sum():.2f} of output variance")
    print("  groups: " + " | ".join(" + ".join(SHORT[h] for h in g) for g in groups))

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, a = plt.subplots(figsize=(12, 5.6))
    # A: one dot per species (mean of its images), coloured by main habitat; large dots = habitat centres;
    # dashed outlines = the groups found by the clustering
    sp = pd.DataFrame(Y[:, :2]).groupby([species, main]).mean()
    for h in range(len(COLS)):
        pts = sp.xs(h, level=1).to_numpy()
        a.scatter(pts[:, 0], pts[:, 1], s=7, c=COLOR[h], alpha=0.25, lw=0)
    c2 = centres[:, :2]
    pad = 0.04 * np.ptp(Y[:, :2], axis=0)
    for g in groups:
        m = c2[g]
        mid = m.mean(0)
        a.add_patch(Ellipse(mid, *(2 * (np.abs(m - mid).max(0) + pad)), fill=False,
                            ec="#333333", ls="--", lw=1.2, zorder=3))
    a.scatter(c2[:, 0], c2[:, 1], s=90, c=COLOR, edgecolors="white", lw=1.5, zorder=4)
    a.set_xlabel(f"PCA axis 1 ({100 * var_model[0]:.0f}% of variance)")
    a.set_ylabel(f"PCA axis 2 ({100 * var_model[1]:.0f}%)")
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(f"Model outputs ({model} model) on the first two PCA axes; dashed: the {N_GROUPS} groups",
                fontsize=11, loc="left")
    fig.tight_layout()
    label_centres(a, c2)
    fig.savefig(f"{SP}/habitat_gradient_{model}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/habitat_gradient.json", "w"), indent=1)
print("wrote habitat_gradient.json, habitat_gradient_{main,sham}.png")
