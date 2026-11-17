import json
import numpy as np
from svlib import SP, B, NH, NAMES, EPS, MIN_SIDE, load, arrays, family_auc, ci

MIN_IMG, MIN_FAM = 200, 5

# ── Per model ────────────────────────────────────────────────────────────────
# Birds that use both A and B (unequal shares): does log(output A) - log(output B) rank A-dominated birds first?
out = {"habitat_names": NAMES, "B": B, "min_img": MIN_IMG, "min_fam": MIN_FAM, "min_side": MIN_SIDE}
for model in ("main", "sham"):
    H, S, F = arrays(load(model))
    _, fidx, wauc = family_auc(F)
    L = np.log(S + EPS)
    pairs = {}
    for a in range(NH):
        for b in range(a + 1, NH):
            m = (H[:, a] > 0) & (H[:, b] > 0) & (H[:, a] != H[:, b])
            y = H[m, a] > H[m, b]
            rec = {"n": int(m.sum()), "n_fam": int(len(set(F[m]))), "n_a_larger": int(y.sum()), "n_b_larger": int((~y).sum())}
            if rec["n"] >= MIN_IMG and rec["n_fam"] >= MIN_FAM and y.sum() >= MIN_SIDE and (~y).sum() >= MIN_SIDE:
                auc, boots = wauc(L[m, a] - L[m, b], y, fidx[m])
                lo, hi = ci(boots)
                rec.update(auc=float(auc), lo=float(lo), hi=float(hi))
                print(f"  {model} {NAMES[a][:20]:20s} vs {NAMES[b][:20]:20s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
            pairs[f"{a},{b}"] = rec
    out[model] = pairs

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/within_bird_check.json", "w"))
print("wrote within_bird_check.json")
