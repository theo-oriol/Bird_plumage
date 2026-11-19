import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from svlib import SP, NAMES, load, arrays, bootstrap_counts, ci

# ── Settings ─────────────────────────────────────────────────────────────────
# Habitat structure in the model's outputs on clean images: PCA of the log outputs
# (each habitat standardised) and groups of habitats from their centres in that space. Next to it,
# the clustering of the habitat centres on all axes, for the labels and for the outputs: a comparison
# that loses no variance, to show the model does not just copy the label structure.
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
    # label each centre at the first free spot around it (16 directions, growing radius), nearer to it than to any other centre
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
                     and not any(x - 7 < p[0] < x + w + 7 and y - 7 < p[1] < y + h + 7 for p in P)
                     and np.linalg.norm(P - (x if x >= P[i, 0] else x + w, y + h / 2), axis=1).argmin() == i]
            if spots:
                break
        x, y = spots[0]
        boxes.append((x, y, w))
        ax.annotate(SHORT[i], pts[i], xytext=inv.transform((x, y + h / 2)), va="center", fontsize=10,
                    fontweight="bold", color="#1a1a1a", zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#444444", lw=0.9, shrinkA=2, shrinkB=5))


def tree(a, L, title):
    # dendrogram of the habitat centres, merge heights as a share of the last merge; dashed = 5 groups
    L = L.copy()
    L[:, 2] /= L[-1, 2]
    dendrogram(L, ax=a, orientation="left", labels=SHORT, leaf_font_size=10, link_color_func=lambda _: "#555555")
    for t in a.get_yticklabels():
        t.set_color(COLOR[SHORT.index(t.get_text())])
        t.set_fontweight("bold")
    a.axvline((L[-N_GROUPS, 2] + L[-N_GROUPS + 1, 2]) / 2, color="#333333", ls="--", lw=1.2)
    a.set_xlim(1.02, 0)
    a.set_xlabel("merge height (share of the last merge)")
    a.spines[["top", "left", "right"]].set_visible(False)
    a.set_xticks([0, 0.5, 1])
    a.set_title(title, fontsize=11, loc="left")


def pair_agreement(H, logS, F):
    # per habitat pair: correlation across images of the two label shares and of the two log outputs;
    # agreement = Pearson r between the label and model correlations over the pairs (all 36, and the
    # 28 without Forest), point and 95% CI from family resamples (per-family sums, reweighted)
    X = np.c_[H, logS].astype(np.float64)
    fams, fidx = np.unique(F, return_inverse=True)
    n = np.bincount(fidx, minlength=len(fams)).astype(np.float64)
    sx = np.zeros((len(fams), X.shape[1]))
    np.add.at(sx, fidx, X)
    sxx = np.zeros((len(fams), X.shape[1], X.shape[1]))
    for f in range(len(fams)):
        sxx[f] = X[fidx == f].T @ X[fidx == f]
    W = np.r_[np.ones((1, len(fams))), bootstrap_counts(list(fams)).astype(np.float64)]  # row 0 = point
    N, SX = W @ n, W @ sx
    SXX = np.einsum("bf,fij->bij", W, sxx)
    cov = SXX / N[:, None, None] - np.einsum("bi,bj->bij", SX, SX) / N[:, None, None] ** 2
    sd = np.sqrt(np.einsum("bii->bi", cov))
    C = cov / sd[:, :, None] / sd[:, None, :]
    k = len(COLS)
    res = {}
    for name, iu in (("all", np.triu_indices(k, 1)), ("no_forest", tuple(i + 1 for i in np.triu_indices(k - 1, 1)))):
        cl, cm = C[:, :k, :k][:, iu[0], iu[1]], C[:, k:, k:][:, iu[0], iu[1]]
        r = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(cl, cm)])
        res[name] = {"r": float(r[0]), "r_ci": ci(r[1:]).tolist(),
                     "spearman": float(spearmanr(cl[0], cm[0])[0]), "n_pairs": len(iu[0])}
    return res


def pair_figure(cl, cm, groups, model):
    # one dot per habitat pair: label correlation (x) vs. model correlation (y)
    g = {h: i for i, gr in enumerate(groups) for h in gr}
    fig, a = plt.subplots(figsize=(6.2, 4.6), layout="constrained")
    for i, j in zip(*np.triu_indices(len(COLS), 1)):
        same, forest = g[i] == g[j], i == 0
        a.scatter(cl[i, j], cm[i, j], s=46, marker="s" if forest else "o",
                  c="#0072B2" if same else "#BBBBBB", edgecolors="#333333" if same else "none", lw=0.6, zorder=3)
    # same-group pairs numbered next to their dot, the names listed in a box
    same_pairs = sorted(((i, j) for i, j in zip(*np.triu_indices(len(COLS), 1)) if g[i] == g[j]), key=lambda p: -cm[p])
    for k, (i, j) in enumerate(same_pairs, start=1):
        a.annotate(str(k), (cl[i, j], cm[i, j]), xytext=(0, 7), textcoords="offset points", ha="center",
                   fontsize=8, fontweight="bold", color="#0072B2")
    a.text(0.02, 0.98, "\n".join(f"{k}  {SHORT[i]} – {SHORT[j]}" for k, (i, j) in enumerate(same_pairs, start=1)),
           transform=a.transAxes, va="top", fontsize=7.5, color="#333333")
    handles = [plt.Line2D([], [], marker="o", ls="", color="#0072B2", label="same group"),
               plt.Line2D([], [], marker="o", ls="", color="#BBBBBB", label="different groups"),
               plt.Line2D([], [], marker="s", ls="", color="#777777", label="pair with Forest")]
    a.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    a.set_xlabel("label correlation across images (share A vs. share B)")
    a.set_ylabel("model correlation across images (log outputs)")
    a.set_title(f"Habitat pairs: labels vs. {model} model", fontsize=10, loc="left")
    a.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{SP}/habitat_pairs_{model}.png", dpi=130)
    plt.close(fig)


def panel(a, Y, var, centres, groups, species, main, title):
    # one dot per species (mean of its images), coloured by main habitat; large dots = habitat centres;
    # dashed outlines = the groups
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
    a.set_xlabel(f"PCA axis 1 ({100 * var[0]:.0f}% of variance)")
    a.set_ylabel(f"PCA axis 2 ({100 * var[1]:.0f}%)")
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(title, fontsize=11, loc="left")


# ── Per model ────────────────────────────────────────────────────────────────
out = {"habitats": [NAMES[h] for h in COLS]}
for model in ("main", "sham"):
    d = load(model)
    H, S, F = arrays(d)
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

    # ── Labels vs. outputs on all axes: centre distances, clustering ─────────
    Yl, _ = pca(H)
    centres_l = np.array([Yl[main == h].mean(0) for h in range(len(COLS))])
    L_lab, L_mod = linkage(centres_l, "ward"), linkage(centres, "ward")
    out[model].update({
        "groups_labels": [[NAMES[COLS[h]] for h in g] for g in clusters(centres_l)],
        "merge_share_labels": (L_lab[:, 2] / L_lab[-1, 2]).tolist(),
        "merge_share_model": (L_mod[:, 2] / L_mod[-1, 2]).tolist(),
        "dist_spearman": float(spearmanr(pdist(centres_l), pdist(centres))[0]),
    })
    print(f"  distances labels vs. outputs: Spearman {out[model]['dist_spearman']:.2f}")

    # ── Labels vs. outputs, pair by pair: same correlation pattern? ──────────
    out[model]["pair_agreement"] = pair_agreement(H, logS, F)
    pair_figure(corr_labels, corr_model, groups, model)
    pa = out[model]["pair_agreement"]
    print(f"  pair correlations labels vs. outputs: r {pa['all']['r']:.2f} {np.round(pa['all']['r_ci'], 2)} (36 pairs), "
          f"{pa['no_forest']['r']:.2f} {np.round(pa['no_forest']['r_ci'], 2)} (28 without Forest)")

    # ── Figure: outputs PCA (A), clustering of labels (B) and outputs (C) ────
    fig = plt.figure(figsize=(17, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1], hspace=0.45, wspace=0.12)
    a = fig.add_subplot(gs[:, 0])
    panel(a, Y, var_model, centres, groups, species, main,
          f"A. Model outputs ({model} model) on the first two PCA axes; dashed: the {N_GROUPS} groups")
    tree(fig.add_subplot(gs[0, 1]), L_lab, "B. Habitat labels: clustering of the habitat centres (all axes)")
    tree(fig.add_subplot(gs[1, 1]), L_mod, "C. Model outputs: the same clustering (all axes)")
    label_centres(a, centres[:, :2])
    fig.savefig(f"{SP}/habitat_gradient_{model}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/habitat_gradient.json", "w"), indent=1)
print("wrote habitat_gradient.json, habitat_gradient_{main,sham}.png, habitat_pairs_{main,sham}.png")
