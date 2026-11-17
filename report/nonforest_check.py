import json
import numpy as np
from sklearn.metrics import roc_auc_score
from svlib import SP, B, NH, NAMES, load, arrays, family_auc, ci

FOREST = 0

# ── Per model ────────────────────────────────────────────────────────────────
# "Not forest" baseline (reversed Forest output as the score), and AUCs among non-forest birds only
out = {"habitat_names": NAMES, "B": B}
for model in ("main", "sham"):
    H, S, F = arrays(load(model))
    nf, nf05 = H[:, FOREST] == 0, H[:, FOREST] < 0.5
    _, fidx, wauc = family_auc(F[nf])  # bootstrap over the non-forest families
    rows = {}
    for t in range(1, NH):
        y = H[:, t] > 0
        yn, Sn = y[nf], S[nf]
        others = {h: roc_auc_score(yn, Sn[:, h]) for h in range(1, NH) if h != t}
        best = max(others, key=others.get)
        own, own_b = wauc(Sn[:, t], yn, fidx)
        _, best_b = wauc(Sn[:, best], yn, fidx)
        (lo, hi), (dlo, dhi) = ci(own_b), ci(own_b - best_b)
        rows[NAMES[t]] = dict(
            prev=float(y.mean()), n_pos=int(y.sum()), n_pos_nf=int(yn.sum()), frac_pos_kept=float(yn.sum() / y.sum()),
            full=roc_auc_score(y, S[:, t]), notforest=roc_auc_score(y, -S[:, FOREST]), nf_own=own,
            nf_lo=float(lo), nf_hi=float(hi), nf_best=NAMES[best], nf_best_auc=others[best], nf_diff=own - others[best],
            nf_dlo=float(dlo), nf_dhi=float(dhi), nf05_own=roc_auc_score(y[nf05], S[nf05][:, t]), n_pos_nf05=int(y[nf05].sum()))
        r = rows[NAMES[t]]
        print(f"  {model} {NAMES[t][:26]:26s} full {r['full']:.3f} | not-forest base {r['notforest']:.3f} | "
              f"non-forest own {own:.3f} [{lo:.3f},{hi:.3f}] | best other {NAMES[best][:14]:14s} {others[best]:.3f} "
              f"diff {own - others[best]:+.3f} [{dlo:+.3f},{dhi:+.3f}]", flush=True)
    out[model] = {"n_images": int(len(S)), "n_nonforest": int(nf.sum()), "n_nonforest_families": int(len(set(F[nf]))),
                  "n_forest_lt05": int(nf05.sum()), "rows": rows}

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/nonforest_check.json", "w"))
print("wrote nonforest_check.json")
