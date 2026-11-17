import json
from svlib import SP, GN, PAIRS, load_hist_images, hist_pred, family_auc, metrics, compare

# ── Load ─────────────────────────────────────────────────────────────────────
H, F, _, _ = load_hist_images()
_, fidx, wauc = family_auc(F)

# ── Colour vs. lightness ─────────────────────────────────────────────────────
# group-pair AUCs of the histogram models: colour + lightness, lightness only, colour only
res = {k: metrics(H, hist_pred(s), fidx, wauc, habitats=False) for k, s in (("all", ""), ("L", "_L"), ("ab", "_ab"))}
out = {"pairs": [[GN[a], GN[b]] for a, b in PAIRS], "mean": {k: float(v[0].mean()) for k, v in res.items()},
       "diffs": {f"{a} - {b}": compare(res[a], res[b]) for a, b in (("ab", "L"), ("all", "ab"), ("all", "L"))}}
for k, v in out["diffs"].items():
    print(f"{k}: mean {v['mean']:+.3f} [{v['mean_ci'][0]:+.3f}, {v['mean_ci'][1]:+.3f}]")

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/hist_compare.json", "w"), indent=1)
