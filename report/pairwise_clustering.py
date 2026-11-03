import json
import os

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# ── Load ─────────────────────────────────────────────────────────────────────
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PW = json.load(open(f"{SP}/pairwise_check.json"))
from paths import LABELS_JSON
names = json.load(open(LABELS_JSON))

# ── Cluster per model ────────────────────────────────────────────────────────
out = {}
for model in ("main", "sham"):
    A = np.array(PW[model]["auc"], dtype=float)
    ok = [int(i) for i in PW[model]["ok"]]
    D = np.abs(A[np.ix_(ok, ok)] - 0.5)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    cuts = {}
    for k in (4, 5, 6, 7):
        lab = fcluster(Z, k, criterion="maxclust")
        groups = {}
        for i, g in zip(ok, lab):
            groups.setdefault(int(g), []).append(names[i])
        cuts[str(k)] = sorted(groups.values(), key=lambda g: names.index(g[0]))
        print(f"{model} k={k}: " + " | ".join(", ".join(g) for g in cuts[str(k)]))
    out[model] = {"habitats": [names[i] for i in ok], "linkage": Z.tolist(), "clusters": cuts}

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/pairwise_clustering.json", "w"), indent=1)
print("wrote pairwise_clustering.json")
