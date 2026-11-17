import gc
import json
import numpy as np
from svlib import SP, B, GROUPS, GN, PAIRS, ABLATIONS, load_conditions, family_auc, main_group, pair_aucs, per_condition, \
    ablation_row, ci

ROWS = ["reference"] + ABLATIONS

# ── Per model ────────────────────────────────────────────────────────────────
# Group-pair AUCs (10 pairs of the 5 groups) under every ablation, and the change vs. clean images
out = {"groups": GROUPS, "group_names": GN, "pairs": PAIRS, "rows": ROWS, "B": B}
wauc = None
for model in ("main", "sham"):
    data = load_conditions(model)
    ref = data["reference"]
    if wauc is None:
        _, _, wauc = family_auc(ref["F"])
        g = main_group(ref["H"])
        out["n_main_group"] = {GN[i]: int((g == i).sum()) for i in range(len(GN))}
        out["n_fam_main_group"] = {GN[i]: int(len(set(ref["F"][g == i]))) for i in range(len(GN))}
    res, clean = per_condition(data, lambda H, S, f: pair_aucs(H, S, f, wauc))
    del data, ref
    gc.collect()
    rows = {}
    for row in ROWS:
        P, BT, Pr, BTr = ablation_row(res, clean, row)
        (lo, hi), (dlo, dhi) = ci(BT), ci(BT - BTr)
        rows[row] = {"auc": P.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                     "clean_same": Pr.tolist(), "delta": (P - Pr).tolist(), "dlo": dlo.tolist(), "dhi": dhi.tolist()}
    out[model] = rows
    print(f"{model} done", flush=True)

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/group_ablation.json", "w"))
print("wrote group_ablation.json")
