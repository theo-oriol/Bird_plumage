"""4.5 check: is detecting a second habitat B among A birds more than 'less typical of A'?
For each cell (A row, B column), among birds using A: truth = also uses B.
own = B output; notA = reversed A output; best = best of the 9 non-B outputs (incl. reversed A).
Family bootstrap of own - best (paired)."""
import json, numpy as np, svlib
d = svlib.load("sham")
H = np.stack(d.habitat.values).astype(float); P = np.stack(d.reg_prob.values).astype(np.float64)
fams = np.array(sorted(d.family.unique())); fidx = np.searchsorted(fams, d.family.values)
del d
wauc = svlib.make_wauc(svlib.bootstrap_counts(fams))
CP = json.load(open(f"{svlib.SP}/copresence_check.json"))
out = {}
for a in range(10):
    for b in range(10):
        if a == b or "auc" not in CP["sham"][f"{a},{b}"]:
            continue
        m = H[:, a] > 0
        y = H[m, b] > 0
        f = fidx[m]
        own, own_b = wauc(P[m, b], y, f)
        na, na_b = wauc(-P[m, a], y, f)
        cands = [(wauc(P[m, k], y, f), k) for k in range(10) if k not in (a, b)] + [((na, na_b), -1)]
        (bv, bb), bk = max(cands, key=lambda c: c[0][0])
        diff = own_b - bb
        out[f"{a},{b}"] = dict(own=own, notA=na, best=bv, best_k=bk, diff=own - bv,
                               lo=float(np.nanpercentile(diff, 2.5)), hi=float(np.nanpercentile(diff, 97.5)))
json.dump(out, open(f"{svlib.SP}/copres_baseline.json", "w"))
print("done", len(out))
