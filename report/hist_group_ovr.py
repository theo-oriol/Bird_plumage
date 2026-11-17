import json
import sys
from svlib import SP, GN, load_hist_images, hist_pred, family_auc, group_ovr, main_group, summary, compare, seed_mean

MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
CONDS = ["reference", "A1", "A7", "A3_h1_r1", "A3_h1_r2", "A3_h1_r3"]

# ── Load ─────────────────────────────────────────────────────────────────────
H, F, _, net = load_hist_images(("main", "sham"), CONDS)
_, fidx, wauc = family_auc(F)

# ── Group AUCs ───────────────────────────────────────────────────────────────
# histogram models vs. the network (clean, A1, A7, A3 h1) at the group level
go = lambda S: group_ovr(H, S, fidx, wauc, MODE)
R = {"hist_L": go(hist_pred("_L")), "hist_ab": go(hist_pred("_ab")), "hist_all": go(hist_pred(""))}
for model in ("main", "sham"):
    R[f"{model}_clean"], R[f"{model}_A1"], R[f"{model}_A7"] = (go(net[(model, c)]) for c in ("reference", "A1", "A7"))
    R[f"{model}_h1"] = seed_mean([go(net[(model, f"A3_h1_r{r}")]) for r in (1, 2, 3)])
g = main_group(H)
out = {"groups": GN, "n_main_group": [int((g == i).sum()) for i in range(len(GN))],
       "rows": {k: summary(v) for k, v in R.items()}, "diffs": {}}
for a, b in (("main_h1", "hist_all"), ("main_clean", "hist_all"), ("hist_ab", "hist_L"), ("hist_all", "hist_ab"),
             ("main_A1", "hist_L"), ("main_A7", "hist_ab"), ("sham_A1", "hist_L")):
    out["diffs"][f"{a} - {b}"] = d = compare(R[a], R[b])
    print(f"{a} - {b}: {d['mean']:+.3f} [{d['mean_ci'][0]:+.3f}, {d['mean_ci'][1]:+.3f}]")

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/hist_group_ovr_{MODE}.json", "w"), indent=1)
