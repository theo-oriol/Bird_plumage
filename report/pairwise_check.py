import json
import numpy as np
from svlib import SP, B, NH, NAMES, EPS, load, arrays, family_auc, ci

MIN_IMG, MIN_FAM = 200, 5

# ── Per model ────────────────────────────────────────────────────────────────
# Each image gets its main habitat; for habitats A and B, AUC of log(output A) - log(output B)
out = {"habitat_names": NAMES, "B": B, "min_img": MIN_IMG, "min_fam": MIN_FAM}
for model in ("main", "sham"):
    H, S, F = arrays(load(model))
    _, fidx, wauc = family_auc(F)
    main_h, L = H.argmax(1), np.log(S + EPS)
    counts = {h: {"n_img": int((main_h == h).sum()), "n_fam": int(len(set(F[main_h == h])))} for h in range(NH)}
    ok = [h for h in range(NH) if counts[h]["n_img"] >= MIN_IMG and counts[h]["n_fam"] >= MIN_FAM]
    M, LO, HI = (np.full((NH, NH), np.nan) for _ in range(3))
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            m = (main_h == a) | (main_h == b)
            auc, boots = wauc(L[m, a] - L[m, b], main_h[m] == a, fidx[m])
            M[a, b] = M[b, a] = auc
            LO[a, b], HI[a, b] = LO[b, a], HI[b, a] = ci(boots)
            print(f"  {model} {NAMES[a][:22]:22s} vs {NAMES[b][:22]:22s} AUC {auc:.3f} [{LO[a, b]:.3f}, {HI[a, b]:.3f}]", flush=True)
    out[model] = {"counts": counts, "ok": ok, "auc": M.tolist(), "lo": LO.tolist(), "hi": HI.tolist()}

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/pairwise_check.json", "w"))
print("wrote pairwise_check.json")
