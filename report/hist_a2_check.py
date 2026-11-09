import gc
import json
import pickle
import numpy as np
from svlib import BASE, SP, bootstrap_counts, make_wauc, metrics, group_ovr, ci
from order_check import make_sauc

# ── Load ─────────────────────────────────────────────────────────────────────
stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True)
pos = {s: i for i, s in enumerate(stems)}
N = len(stems)
H = np.zeros((N, 10), np.float32)
fam = np.empty(N, object)
fold = np.full(N, -1)
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
            fold[ii] = k
    del df
    gc.collect()
fams = sorted(set(fam))
fam2i = {f: i for i, f in enumerate(fams)}
fidx = np.array([fam2i[f] for f in fam], np.int32)
counts = bootstrap_counts(fams)

# Pooled AUC (as in the report) and within-fold AUC (pairs of images from the same fold only)
fam_fold = np.zeros(len(fams), np.int32)
fam_fold[fidx] = fold
sauc = make_sauc(counts)
AUC = {"pooled": make_wauc(counts),
       "within_fold": lambda s, y, f: sauc(s.astype(np.float64), y, f, fam_fold[f])}

# ── Sources ──────────────────────────────────────────────────────────────────
SRC = {"hist_A2": np.load(f"{SP}/hist_pred_crops_A2_mlp.npy"), "hist_A2_L": np.load(f"{SP}/hist_pred_crops_A2_L_mlp.npy"),
       "net_A2": net["A2"], "net_clean": net["reference"]}
out = {"levels": ["habitats", "groups", "pairs"], "n": N, "sources": {}, "diffs": {}}
boots = {}
for mode, wauc in AUC.items():
    out["sources"][mode], boots[mode] = {}, {}
    for name, S in SRC.items():
        hp, hb = metrics(H, S, fidx, wauc)
        gp, gb = group_ovr(H, S, fidx, wauc, "main")
        lv = {"habitats": (hp[:10], hb[:10]), "groups": (gp, gb), "pairs": (hp[10:], hb[10:])}
        boots[mode][name] = {k: v[1] for k, v in lv.items()}
        out["sources"][mode][name] = {k: {"auc": v[0].tolist(), "lo": ci(v[1])[0].tolist(), "hi": ci(v[1])[1].tolist(),
                                          "mean": float(np.nanmean(v[0])), "mean_ci": ci(np.nanmean(v[1], 0)).tolist(),
                                          "n_above": int((ci(v[1])[0] > 0.5).sum()), "n_below": int((ci(v[1])[1] < 0.5).sum())}
                                      for k, v in lv.items()}
        print(f"{mode:12s} {name:10s} " + "   ".join(
            f"{l} {out['sources'][mode][name][l]['mean']:.3f} [{out['sources'][mode][name][l]['mean_ci'][0]:.3f}, "
            f"{out['sources'][mode][name][l]['mean_ci'][1]:.3f}]" for l in out["levels"]), flush=True)
    # Network on A2 minus the histogram model on the same A2 crops (paired)
    out["diffs"][mode] = {}
    for b in ("hist_A2", "hist_A2_L"):
        out["diffs"][mode][f"net_A2 - {b}"] = {
            l: {"mean": out["sources"][mode]["net_A2"][l]["mean"] - out["sources"][mode][b][l]["mean"],
                "ci": ci(np.nanmean(boots[mode]["net_A2"][l] - boots[mode][b][l], 0)).tolist()} for l in out["levels"]}

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/hist_a2_check.json", "w"), indent=1)
for mode, d in out["diffs"].items():
    for k, v in d.items():
        print(f"{mode:12s} {k:20s} " + "   ".join(f"{l} {v[l]['mean']:+.3f} [{v[l]['ci'][0]:+.3f}, {v[l]['ci'][1]:+.3f}]"
                                                 for l in out["levels"]))
