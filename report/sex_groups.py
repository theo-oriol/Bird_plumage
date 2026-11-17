import json
import numpy as np
from svlib import SP, B, GN, TERTS, VIEWS, SEXES, load, arrays, family_auc, group_ovr, species_with_both_sexes, \
    dimorphism_tertiles, compare, ci

# ── Per model ────────────────────────────────────────────────────────────────
# Group AUCs (one vs. rest) per view, 3 views, sex, and sex within dimorphism tertile; paired differences
out = {"groups": GN, "B": B}
wauc = None
for model in ("sham", "main"):
    d = load(model)
    if wauc is None:
        fam2i, _, wauc = family_auc(d.family)

    def go(sub, S=None):
        H, S0, F = arrays(sub)
        return group_ovr(H, S0 if S is None else S, np.array([fam2i[x] for x in F], np.int32), wauc)

    views = {v: d[d.view == v].sort_values("cat") for v in VIEWS}
    res = {v: go(sub) for v, sub in views.items()}
    res["3 views"] = go(views["Back"], np.mean([np.stack(views[v].reg_prob.to_numpy()) for v in VIEWS], 0))
    both = species_with_both_sexes(d.species, d.sex)
    for sx in SEXES:
        res[sx] = go(d[(d.sex == sx) & d.species.isin(both)])
    tert, _, _ = dimorphism_tertiles(both)
    dd = d[d.species.isin(tert.index)].copy()
    dd["tert"] = dd.species.map(tert)
    for t in TERTS:
        for sx in SEXES:
            res[f"{t}_{sx}"] = go(dd[(dd.tert == t) & (dd.sex == sx)])
    rows = {k: {"auc": P.tolist(), "mean": float(np.nanmean(P))} for k, (P, _) in res.items()}
    diffs = {f"{a} - {b}": compare(res[a], res[b]) for a, b in
             [("Belly", "Back"), ("Side", "Back"), ("3 views", "Back"), ("3 views", "Belly"), ("3 views", "Side"), ("F", "M")]}
    for t in TERTS:
        diffs[f"{t}: F - M"] = compare(res[f"{t}_F"], res[f"{t}_M"])
    gap = {t: np.nanmean(res[f"{t}_F"][1] - res[f"{t}_M"][1], 0) for t in ("low", "high")}
    diffs["trend high - low"] = {"mean": diffs["high: F - M"]["mean"] - diffs["low: F - M"]["mean"],
                                 "mean_ci": ci(gap["high"] - gap["low"]).tolist()}
    out[model] = {"rows": rows, "diffs": diffs}
    print(f"{model}: " + "  ".join(f"{k} {v['mean']:+.3f}" for k, v in diffs.items()), flush=True)

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/sex_groups.json", "w"), indent=1)
