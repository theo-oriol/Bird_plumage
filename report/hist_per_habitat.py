import json
from svlib import SP, NH, NAMES, load_hist_images, hist_pred, family_auc, metrics, compare, seed_mean

CONDS = ["reference", "A1", "A7", "A3_h1_r1", "A3_h1_r2", "A3_h1_r3"]

# ── Load ─────────────────────────────────────────────────────────────────────
H, F, _, net = load_hist_images(("main",), CONDS)
_, fidx, wauc = family_auc(F)

# ── Habitat AUCs ─────────────────────────────────────────────────────────────
# histogram models vs. the network, per habitat
hab = lambda S: tuple(x[:NH] for x in metrics(H, S, fidx, wauc))
R = {"hist_all": hab(hist_pred("")), "hist_L": hab(hist_pred("_L")), "hist_ab": hab(hist_pred("_ab")),
     "main_clean": hab(net[("main", "reference")]), "main_A1": hab(net[("main", "A1")]), "main_A7": hab(net[("main", "A7")]),
     "main_h1": seed_mean([hab(net[("main", f"A3_h1_r{r}")]) for r in (1, 2, 3)])}
out = {"habitats": NAMES, "rows": {k: v[0].tolist() for k, v in R.items()}, "diffs": {}}
for a, b in (("main_h1", "hist_all"), ("main_clean", "hist_all"), ("hist_ab", "hist_L"), ("main_A1", "hist_L"), ("main_A7", "hist_ab")):
    out["diffs"][f"{a} - {b}"] = d = compare(R[a], R[b])
    print(f"{a} - {b}: {d['mean']:+.3f} [{d['mean_ci'][0]:+.3f}, {d['mean_ci'][1]:+.3f}]")

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/hist_per_habitat.json", "w"), indent=1)
