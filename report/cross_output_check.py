import json
import numpy as np
from sklearn.metrics import roc_auc_score
from svlib import SP, B, NH, NAMES, load, arrays, family_auc, ci

# ── Per model ────────────────────────────────────────────────────────────────
# For each habitat (labels fixed): AUC of every output as the score; is the own output the best?
out = {"habitat_names": NAMES, "B": B}
for model in ("main", "sham"):
    H, S, F = arrays(load(model))
    _, fidx, wauc = family_auc(F)
    M, rows = np.zeros((NH, NH)), []
    for t in range(NH):
        y = H[:, t] > 0
        M[t] = [roc_auc_score(y, S[:, h]) for h in range(NH)]
        best = max((h for h in range(NH) if h != t), key=lambda h: M[t, h])
        lo, hi = ci(wauc(S[:, t], y, fidx)[1] - wauc(S[:, best], y, fidx)[1])
        rows.append({"own": M[t, t], "best_other": best, "best_other_auc": M[t, best],
                     "diff": M[t, t] - M[t, best], "lo": float(lo), "hi": float(hi)})
        print(f"  {model} {NAMES[t][:28]:28s} own {M[t, t]:.3f} | best other {NAMES[best][:24]:24s} {M[t, best]:.3f} "
              f"| diff {M[t, t] - M[t, best]:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    out[model] = {"matrix": M.tolist(), "rows": rows}

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/cross_output_check.json", "w"))
print("wrote cross_output_check.json")
