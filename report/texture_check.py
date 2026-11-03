import gc
import json
import pickle
import numpy as np
from svlib import BASE, SP, GN, PAIRS, bootstrap_counts, make_wauc, metrics, ci

# ── Load ─────────────────────────────────────────────────────────────────────
stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True)
pos = {s: i for i, s in enumerate(stems)}
N = len(stems)
H = np.zeros((N, 10), np.float32)
fam = np.empty(N, object)
net = {}
for model in ("main", "sham"):
    for c in ("A1", "A7"):
        net[(model, c)] = np.full((N, 10), np.nan)
    for k in range(3):
        df = pickle.load(open(f"{BASE}/{model}_fold{k}/eval/predictions.pkl", "rb"))
        df = df[df.condition_name.isin(["reference", "A1", "A7"])]
        for c, sub in df.groupby("condition_name"):
            ii = np.array([pos[s] for s in sub.stem])
            if c == "reference":
                H[ii] = np.stack(sub.habitat.to_numpy())
                fam[ii] = sub.family.to_numpy()
            else:
                net[(model, c)][ii] = np.stack(sub.reg_prob.to_numpy())
        del df
        gc.collect()
fams = sorted(set(fam))
fam2i = {f: i for i, f in enumerate(fams)}
fidx = np.array([fam2i[f] for f in fam], np.int32)
wauc = make_wauc(bootstrap_counts(fams))
base = {"L": metrics(H, np.load(f"{SP}/hist_pred_crops_L_mlp.npy"), fidx, wauc, habitats=False),
        "ab": metrics(H, np.load(f"{SP}/hist_pred_crops_ab_mlp.npy"), fidx, wauc, habitats=False)}

# ── Network vs. histogram per channel ────────────────────────────────────────
out = {"pairs": [[GN[a], GN[b]] for a, b in PAIRS]}
for model in ("main", "sham"):
    for c, b in (("A1", "L"), ("A7", "ab")):
        P, BT = metrics(H, net[(model, c)], fidx, wauc, habitats=False)
        D = BT - base[b][1]
        lo, hi = ci(D)
        mci = ci(D.mean(0))
        key = f"{model} {c} - hist {b}"
        out[key] = {"net": P.tolist(), "hist": base[b][0].tolist(), "delta": (P - base[b][0]).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                    "mean_net": float(P.mean()), "mean_hist": float(base[b][0].mean()), "mean": float((P - base[b][0]).mean()), "mean_ci": mci.tolist()}
        nsig_pos = int((lo > 0).sum())
        nsig_neg = int((hi < 0).sum())
        print(f"{key:22s} net {P.mean():.3f} vs hist {base[b][0].mean():.3f}: {P.mean() - base[b][0].mean():+.3f} "
              f"[{mci[0]:+.3f}, {mci[1]:+.3f}]  pairs sig + {nsig_pos} / - {nsig_neg}")
        for p in range(10):
            print(f"    {GN[PAIRS[p][0]]:>24s} vs {GN[PAIRS[p][1]]:<10s} {P[p]:.3f} vs {base[b][0][p]:.3f}  {P[p] - base[b][0][p]:+.3f} [{lo[p]:+.3f}, {hi[p]:+.3f}]")

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/texture_check.json", "w"), indent=1)
