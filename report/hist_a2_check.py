import json
import numpy as np
from svlib import SP, load_hist_images, hist_pred, family_auc, bootstrap_counts, make_sauc, three_levels, summary, ci

# ── Load ─────────────────────────────────────────────────────────────────────
H, F, fold, net = load_hist_images(("main",), ("reference", "A2"))
fam2i, fidx, wauc = family_auc(F)
fam_fold = np.zeros(len(fam2i), np.int32)
fam_fold[fidx] = fold
sauc = make_sauc(bootstrap_counts(sorted(fam2i)))
# pooled AUC (as in the rest of the report) and within-fold AUC (only pairs of images from the same fold)
AUC = {"pooled": wauc, "within_fold": lambda s, y, f: sauc(s.astype(np.float64), y, f, fam_fold[f])}

# ── Sources ──────────────────────────────────────────────────────────────────
# histogram models trained on the A2 crops vs. the network on A2
SRC = {"hist_A2": hist_pred("_A2"), "hist_A2_L": hist_pred("_A2_L"), "net_A2": net[("main", "A2")],
       "net_clean": net[("main", "reference")]}
out = {"levels": ["habitats", "groups", "pairs"], "n": len(F), "sources": {}, "diffs": {}}
for mode, auc in AUC.items():
    R = {name: three_levels(H, S, fidx, auc) for name, S in SRC.items()}
    out["sources"][mode] = {name: {l: {**summary(v), "n_above": int((ci(v[1])[0] > 0.5).sum()),
                                       "n_below": int((ci(v[1])[1] < 0.5).sum())} for l, v in lv.items()}
                            for name, lv in R.items()}
    # network on A2 minus the histogram model on the same A2 crops (paired)
    out["diffs"][mode] = {f"net_A2 - {b}": {l: {"mean": out["sources"][mode]["net_A2"][l]["mean"] - out["sources"][mode][b][l]["mean"],
                                                 "ci": ci(np.nanmean(R["net_A2"][l][1] - R[b][l][1], 0)).tolist()}
                                             for l in out["levels"]} for b in ("hist_A2", "hist_A2_L")}

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/hist_a2_check.json", "w"), indent=1)
for mode, d in out["diffs"].items():
    for k, v in d.items():
        print(f"{mode:12s} {k:20s} " + "   ".join(f"{l} {v[l]['mean']:+.3f} [{v[l]['ci'][0]:+.3f}, {v[l]['ci'][1]:+.3f}]"
                                                 for l in out["levels"]))
