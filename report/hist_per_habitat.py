import gc, json, pickle
import numpy as np
from svlib import BASE, SP, bootstrap_counts, make_wauc, metrics, ci

from paths import LABELS_JSON

# ── Load ─────────────────────────────────────────────────────────────────────
names = json.load(open(LABELS_JSON))
stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True)
pos = {s: i for i, s in enumerate(stems)}
N = len(stems)
H = np.zeros((N, 10), np.float32); fam = np.empty(N, object)
CONDS = ["reference", "A1", "A7", "A3_h1_r1", "A3_h1_r2", "A3_h1_r3"]
net = {c: np.full((N, 10), np.nan) for c in CONDS}
for k in range(3):
    df = pickle.load(open(f"{BASE}/main_fold{k}/eval/predictions.pkl", "rb"))
    df = df[df.condition_name.isin(CONDS)]
    for c, sub in df.groupby("condition_name"):
        ii = np.array([pos[s] for s in sub.stem])
        net[c][ii] = np.stack(sub.reg_prob.to_numpy())
        if c == "reference":
            H[ii] = np.stack(sub.habitat.to_numpy()); fam[ii] = sub.family.to_numpy()
    del df; gc.collect()

# ── Habitat AUCs ─────────────────────────────────────────────────────────────
fams = sorted(set(fam)); fam2i = {f: i for i, f in enumerate(fams)}
fidx = np.array([fam2i[f] for f in fam], np.int32)
wauc = make_wauc(bootstrap_counts(fams))
hab = lambda S: (lambda r: (r[0][:10], r[1][:10]))(metrics(H, S, fidx, wauc))
R = {"hist_all": hab(np.load(f"{SP}/hist_pred_crops_mlp.npy")), "hist_L": hab(np.load(f"{SP}/hist_pred_crops_L_mlp.npy")),
     "hist_ab": hab(np.load(f"{SP}/hist_pred_crops_ab_mlp.npy")), "main_clean": hab(net["reference"]),
     "main_A1": hab(net["A1"]), "main_A7": hab(net["A7"])}
s = [hab(net[f"A3_h1_r{r}"]) for r in (1, 2, 3)]
R["main_h1"] = (np.mean([x[0] for x in s], 0), np.mean([x[1] for x in s], 0))
out = {"habitats": names, "rows": {k: v[0].tolist() for k, v in R.items()}, "diffs": {}}

# ── Print and write ──────────────────────────────────────────────────────────
print(f"{'':40s}" + "".join(f"{k:>11s}" for k in R))
for h in range(10):
    print(f"{names[h][:40]:40s}" + "".join(f"{R[k][0][h]:11.3f}" for k in R))
print(f"{'MEAN':40s}" + "".join(f"{R[k][0].mean():11.3f}" for k in R))
for a, b in (("main_h1", "hist_all"), ("main_clean", "hist_all"), ("hist_ab", "hist_L"), ("main_A1", "hist_L"), ("main_A7", "hist_ab")):
    D = R[a][1] - R[b][1]; lo, hi = ci(D); m = ci(D.mean(0))
    out["diffs"][f"{a} - {b}"] = {"delta": (R[a][0] - R[b][0]).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                                  "mean": float((R[a][0] - R[b][0]).mean()), "mean_ci": m.tolist()}
    print(f"{a} - {b}: {(R[a][0] - R[b][0]).mean():+.3f} [{m[0]:+.3f}, {m[1]:+.3f}]  sig+ {int((lo > 0).sum())} sig- {int((hi < 0).sum())}")
json.dump(out, open(f"{SP}/hist_per_habitat.json", "w"), indent=1)
