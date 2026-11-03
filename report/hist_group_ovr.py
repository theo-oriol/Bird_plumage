import gc, json, pickle
import sys
import numpy as np

# ── Settings ─────────────────────────────────────────────────────────────────
MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
from svlib import BASE, SP, GN, bootstrap_counts, make_wauc, group_ovr, ci

# ── Load ─────────────────────────────────────────────────────────────────────
stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True)
pos = {s: i for i, s in enumerate(stems)}
N = len(stems)
H = np.zeros((N, 10), np.float32); fam = np.empty(N, object)
CONDS = ["reference", "A1", "A7", "A3_h1_r1", "A3_h1_r2", "A3_h1_r3"]
net = {}
for model in ("main", "sham"):
    for c in CONDS:
        net[(model, c)] = np.full((N, 10), np.nan)
    for k in range(3):
        df = pickle.load(open(f"{BASE}/{model}_fold{k}/eval/predictions.pkl", "rb"))
        df = df[df.condition_name.isin(CONDS)]
        for c, sub in df.groupby("condition_name"):
            ii = np.array([pos[s] for s in sub.stem])
            net[(model, c)][ii] = np.stack(sub.reg_prob.to_numpy())
            if c == "reference" and model == "main":
                H[ii] = np.stack(sub.habitat.to_numpy()); fam[ii] = sub.family.to_numpy()
        del df; gc.collect()

# ── Group AUCs ───────────────────────────────────────────────────────────────
fams = sorted(set(fam)); fam2i = {f: i for i, f in enumerate(fams)}
fidx = np.array([fam2i[f] for f in fam], np.int32)
wauc = make_wauc(bootstrap_counts(fams))
go = lambda S: group_ovr(H, S, fidx, wauc, MODE)
R = {"hist_L": go(np.load(f"{SP}/hist_pred_crops_L_mlp.npy")), "hist_ab": go(np.load(f"{SP}/hist_pred_crops_ab_mlp.npy")),
     "hist_all": go(np.load(f"{SP}/hist_pred_crops_mlp.npy"))}
for model in ("main", "sham"):
    R[f"{model}_clean"] = go(net[(model, "reference")])
    R[f"{model}_A1"] = go(net[(model, "A1")])
    R[f"{model}_A7"] = go(net[(model, "A7")])
    s = [go(net[(model, f"A3_h1_r{r}")]) for r in (1, 2, 3)]
    R[f"{model}_h1"] = (np.mean([x[0] for x in s], 0), np.mean([x[1] for x in s], 0))
gs = np.stack([H[:, [[0], [1, 2, 6], [3, 4, 9], [5, 7], [8]][g]].sum(1) for g in range(5)], 1).argmax(1)
out = {"groups": GN, "n_main_group": [int((gs == g).sum()) for g in range(5)],
       "rows": {k: {"auc": v[0].tolist(), "lo": ci(v[1])[0].tolist(), "hi": ci(v[1])[1].tolist(), "mean": float(v[0].mean()),
                    "mean_ci": ci(v[1].mean(0)).tolist()} for k, v in R.items()}, "diffs": {}}

# ── Print and write ──────────────────────────────────────────────────────────
print(f"{'':26s}" + "".join(f"{k:>11s}" for k in R))
for g in range(5):
    print(f"{GN[g]:26s}" + "".join(f"{R[k][0][g]:11.3f}" for k in R))
print(f"{'MEAN':26s}" + "".join(f"{R[k][0].mean():11.3f}" for k in R))
for a, b in (("main_h1", "hist_all"), ("main_clean", "hist_all"), ("hist_ab", "hist_L"), ("hist_all", "hist_ab"),
             ("main_A1", "hist_L"), ("main_A7", "hist_ab"), ("sham_A1", "hist_L")):
    D = R[a][1] - R[b][1]; lo, hi = ci(D); m = ci(D.mean(0))
    out["diffs"][f"{a} - {b}"] = {"delta": (R[a][0] - R[b][0]).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                                  "mean": float((R[a][0] - R[b][0]).mean()), "mean_ci": m.tolist()}
    print(f"{a} - {b}: {(R[a][0] - R[b][0]).mean():+.3f} [{m[0]:+.3f}, {m[1]:+.3f}]  per group " +
          " ".join(f"{d:+.3f}{'*' if l > 0 or h < 0 else ''}" for d, l, h in zip(R[a][0] - R[b][0], lo, hi)))
json.dump(out, open(f"{SP}/hist_group_ovr_{MODE}.json", "w"), indent=1)
