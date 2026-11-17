import json
import os
from svlib import SP, B, NH, NAMES, MIN_SIDE, load, arrays, family_auc, ci

MIN_FAM = 5

# ── Per model ────────────────────────────────────────────────────────────────
# Among birds that use A: does the B output find those that also use B?
out = {"habitat_names": NAMES, "B": B, "min_side": MIN_SIDE, "min_fam": MIN_FAM}
for model in os.environ.get("MODELS", "main,sham").split(","):
    H, S, F = arrays(load(model))
    _, fidx, wauc = family_auc(F)
    pairs = {}
    for a in range(NH):
        m = H[:, a] > 0
        fi = fidx[m]
        for b in range(NH):
            if a == b:
                continue
            y = H[m, b] > 0
            rec = {"n": int(m.sum()), "n_fam": int(len(set(fi))), "n_with_b": int(y.sum()), "n_without_b": int((~y).sum()),
                   "n_fam_with_b": int(len(set(fi[y]))), "n_fam_without_b": int(len(set(fi[~y])))}
            if min(y.sum(), (~y).sum()) >= MIN_SIDE and min(rec["n_fam_with_b"], rec["n_fam_without_b"]) >= MIN_FAM:
                auc, boots = wauc(S[m, b], y, fi)
                lo, hi = ci(boots)
                rec.update(auc=float(auc), lo=float(lo), hi=float(hi))
                print(f"  {model} uses {NAMES[a][:20]:20s} -> also {NAMES[b][:20]:20s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
            pairs[f"{a},{b}"] = rec
    out[model] = pairs

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/copresence_check.json", "w"))
print("wrote copresence_check.json")
