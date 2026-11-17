import gc
import json
import sys
import numpy as np
from svlib import SP, B, GN, ABLATIONS, load_conditions, family_auc, group_ovr, per_condition, ablation_row, ci

# argv[1]: 'main' = a bird belongs to its main group (largest label share), 'presence' = to every group it uses
MODE = sys.argv[1] if len(sys.argv) > 1 else "main"

# ── Per model ────────────────────────────────────────────────────────────────
# One-vs-rest AUC per group under every ablation, and the change vs. clean images
out = {"groups": GN, "rows": ABLATIONS, "B": B}
wauc = None
for model in ("main", "sham"):
    data = load_conditions(model)
    if wauc is None:
        _, _, wauc = family_auc(data["reference"]["F"])
    res, clean = per_condition(data, lambda H, S, f: group_ovr(H, S, f, wauc, MODE))
    del data
    gc.collect()
    P_ref, BT_ref = res["reference"]
    lo, hi = ci(BT_ref)
    rows = {"reference": {"auc": P_ref.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "mean": float(P_ref.mean())}}
    for row in ABLATIONS:
        P, BT, Pr, BTr = ablation_row(res, clean, row)
        D = BT - BTr
        dlo, dhi = ci(D)
        rows[row] = {"auc": P.tolist(), "clean_same": Pr.tolist(), "delta": (P - Pr).tolist(), "dlo": dlo.tolist(),
                     "dhi": dhi.tolist(), "mean_delta": float(np.nanmean(P - Pr)), "mean_ci": ci(np.nanmean(D, 0)).tolist()}
    out[model] = rows
    print(f"{model}: clean " + " ".join(f"{g} {a:.3f}" for g, a in zip(GN, P_ref)), flush=True)

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/group_ovr_ablation_{MODE}.json", "w"))
print(f"wrote group_ovr_ablation_{MODE}.json")
