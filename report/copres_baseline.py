import json
import sys
import numpy as np
from svlib import SP, NH, load, arrays, family_auc, ci

# output: copres_baseline.json (sham) or copres_baseline_<model>.json
MODEL = sys.argv[1] if len(sys.argv) > 1 else "sham"

# ── Load ─────────────────────────────────────────────────────────────────────
H, S, F = arrays(load(MODEL))
_, fidx, wauc = family_auc(F)
CP = json.load(open(f"{SP}/copresence_check.json"))[MODEL]

# ── Compute ──────────────────────────────────────────────────────────────────
# For each detected cell (among A birds, also B?): own B output vs. every other output and "less typical of A"
out = {}
for a in range(NH):
    for b in range(NH):
        if a == b or "auc" not in CP[f"{a},{b}"]:
            continue
        m = H[:, a] > 0
        y, f = H[m, b] > 0, fidx[m]
        own, own_b = wauc(S[m, b], y, f)
        not_a = wauc(-S[m, a], y, f)
        cands = [(wauc(S[m, k], y, f), k) for k in range(NH) if k not in (a, b)] + [(not_a, -1)]
        (best, best_b), best_k = max(cands, key=lambda c: c[0][0])
        lo, hi = ci(own_b - best_b)
        out[f"{a},{b}"] = dict(own=own, notA=not_a[0], best=best, best_k=best_k, diff=own - best, lo=float(lo), hi=float(hi))

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/copres_baseline{'' if MODEL == 'sham' else '_' + MODEL}.json", "w"))
print("done", len(out))
