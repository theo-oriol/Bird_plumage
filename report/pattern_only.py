import gc
import json
import pickle
import numpy as np
from svlib import BASE, SP, bootstrap_counts, make_wauc, metrics, group_ovr, ci

# ── Load ─────────────────────────────────────────────────────────────────────
stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True)
pos = {s: i for i, s in enumerate(stems)}
N = len(stems)
H = np.zeros((N, 10), np.float32)
fam = np.empty(N, object)
net = {c: np.full((N, 10), np.nan) for c in ("reference", "A2")}
for k in range(3):
    df = pickle.load(open(f"{BASE}/main_fold{k}/eval/predictions.pkl", "rb"))
    df = df[df.condition_name.isin(list(net))]
    for c, sub in df.groupby("condition_name"):
        ii = np.array([pos[s] for s in sub.stem])
        net[c][ii] = np.stack(sub.reg_prob.to_numpy())
        if c == "reference":
            H[ii] = np.stack(sub.habitat.to_numpy())
            fam[ii] = sub.family.to_numpy()
    del df
    gc.collect()
fams = sorted(set(fam))
fam2i = {f: i for i, f in enumerate(fams)}
fidx = np.array([fam2i[f] for f in fam], np.int32)
wauc = make_wauc(bootstrap_counts(fams))

# ── Sources on their own ─────────────────────────────────────────────────────
SRC = {"pattern": net["A2"], "lightness": np.load(f"{SP}/hist_pred_crops_L_mlp.npy"),
       "colour": np.load(f"{SP}/hist_pred_crops_ab_mlp.npy"), "colour_lightness": np.load(f"{SP}/hist_pred_crops_mlp.npy"),
       "everything": net["reference"]}
out = {"levels": ["habitats", "groups", "pairs"], "sources": {}, "diffs": {}}
boots = {}
for name, S in SRC.items():
    hp, hb = metrics(H, S, fidx, wauc)  # 10 habitats + 10 pairs
    gp, gb = group_ovr(H, S, fidx, wauc, "main")
    lv = {"habitats": (hp[:10], hb[:10]), "groups": (gp, gb), "pairs": (hp[10:], hb[10:])}
    boots[name] = {k: v[1] for k, v in lv.items()}
    out["sources"][name] = {k: {"auc": v[0].tolist(), "lo": ci(v[1])[0].tolist(), "hi": ci(v[1])[1].tolist(),
                                "mean": float(np.nanmean(v[0])), "mean_ci": ci(np.nanmean(v[1], 0)).tolist(),
                                "n_above_chance": int((ci(v[1])[0] > 0.5).sum())} for k, v in lv.items()}
for a, b in (("pattern", "lightness"), ("pattern", "colour")):
    out["diffs"][f"{a} - {b}"] = {}
    for lvl in out["levels"]:
        D = np.nanmean(boots[a][lvl] - boots[b][lvl], 0)
        out["diffs"][f"{a} - {b}"][lvl] = {"mean": out["sources"][a][lvl]["mean"] - out["sources"][b][lvl]["mean"], "ci": ci(D).tolist()}

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/pattern_only.json", "w"), indent=1)
for name, v in out["sources"].items():
    print(f"{name:18s} " + "   ".join(f"{l} {v[l]['mean']:.3f} [{v[l]['mean_ci'][0]:.3f}, {v[l]['mean_ci'][1]:.3f}] ({v[l]['n_above_chance']} > 0.5)" for l in out["levels"]))
for k, v in out["diffs"].items():
    print(f"{k:22s} " + "   ".join(f"{l} {v[l]['mean']:+.3f} [{v[l]['ci'][0]:+.3f}, {v[l]['ci'][1]:+.3f}]" for l in out["levels"]))
