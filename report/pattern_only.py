import json
import numpy as np
from svlib import SP, load_hist_images, hist_pred, family_auc, three_levels, summary, ci

# ── Load ─────────────────────────────────────────────────────────────────────
H, F, _, net = load_hist_images(("main",), ("reference", "A2"))
_, fidx, wauc = family_auc(F)

# ── Sources on their own ─────────────────────────────────────────────────────
# pattern = network on A2 (greyscale, lightness matched); lightness / colour = histogram models
SRC = {"pattern": net[("main", "A2")], "lightness": hist_pred("_L"), "colour": hist_pred("_ab"),
       "colour_lightness": hist_pred(""), "everything": net[("main", "reference")]}
R = {name: three_levels(H, S, fidx, wauc) for name, S in SRC.items()}
LEVELS = list(R["pattern"])
out = {"levels": LEVELS, "sources": {}, "diffs": {}}
for name, lv in R.items():
    out["sources"][name] = {l: {**summary(v), "n_above_chance": int((ci(v[1])[0] > 0.5).sum())} for l, v in lv.items()}
for a, b in (("pattern", "lightness"), ("pattern", "colour")):
    out["diffs"][f"{a} - {b}"] = {l: {"mean": out["sources"][a][l]["mean"] - out["sources"][b][l]["mean"],
                                      "ci": ci(np.nanmean(R[a][l][1] - R[b][l][1], 0)).tolist()} for l in LEVELS}

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/pattern_only.json", "w"), indent=1)
for k, v in out["diffs"].items():
    print(f"{k:22s} " + "   ".join(f"{l} {v[l]['mean']:+.3f} [{v[l]['ci'][0]:+.3f}, {v[l]['ci'][1]:+.3f}]" for l in LEVELS))
