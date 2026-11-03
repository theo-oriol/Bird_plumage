import json
import pickle
import numpy as np
from svlib import BASE, SP, GN, PAIRS, bootstrap_counts, make_wauc, metrics, ci

# ── Load ─────────────────────────────────────────────────────────────────────
stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True)
pos = {s: i for i, s in enumerate(stems)}
H = np.zeros((len(stems), 10), np.float32)
fam = np.empty(len(stems), object)
for k in range(3):
    df = pickle.load(open(f"{BASE}/main_fold{k}/eval/predictions.pkl", "rb"))
    r = df[df.condition_name == "reference"]
    ii = np.array([pos[s] for s in r.stem])
    H[ii] = np.stack(r.habitat.to_numpy())
    fam[ii] = r.family.to_numpy()
    del df, r

# ── Colour vs. lightness ─────────────────────────────────────────────────────
fams = sorted(set(fam))
fam2i = {f: i for i, f in enumerate(fams)}
fidx = np.array([fam2i[f] for f in fam], np.int32)
wauc = make_wauc(bootstrap_counts(fams))
res = {k: metrics(H, np.load(f"{SP}/hist_pred_crops{s}_mlp.npy"), fidx, wauc, habitats=False)
       for k, s in (("all", ""), ("L", "_L"), ("ab", "_ab"))}
out = {"pairs": [[GN[a], GN[b]] for a, b in PAIRS], "mean": {k: float(v[0].mean()) for k, v in res.items()}, "diffs": {}}
for a, b in (("ab", "L"), ("all", "ab"), ("all", "L")):
    D = res[a][1] - res[b][1]
    lo, hi = ci(D)
    out["diffs"][f"{a} - {b}"] = {"delta": (res[a][0] - res[b][0]).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                                  "mean": float((res[a][0] - res[b][0]).mean()), "mean_ci": ci(D.mean(0)).tolist()}
    print(f"{a} - {b}: mean {out['diffs'][f'{a} - {b}']['mean']:+.3f} {np.round(ci(D.mean(0)), 3)}  "
          f"sig pairs {int(((lo > 0) | (hi < 0)).sum())}/10")

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/hist_compare.json", "w"), indent=1)
