import json
import numpy as np
from svlib import SP, B, GN, VIEWS, SEXES, load, arrays, family_auc, group_ovr, species_with_both_sexes, summary

# ── Per model ────────────────────────────────────────────────────────────────
# Group AUCs for every sex x view (species with both sexes) and the sex gap per view
out = {"groups": GN, "B": B}
wauc = None
for model in ("sham", "main"):
    d = load(model)
    if wauc is None:
        fam2i, _, wauc = family_auc(d.family)
    d = d[d.species.isin(species_with_both_sexes(d.species, d.sex)) & d.sex.isin(SEXES)]
    res, rows = {}, {}
    for s in SEXES:
        for v in VIEWS:
            sub = d[(d.sex == s) & (d.view == v)]
            H, S, F = arrays(sub)
            res[f"{s}_{v}"] = group_ovr(H, S, np.array([fam2i[x] for x in F], np.int32), wauc)
            rows[f"{s}_{v}"] = {"auc": res[f"{s}_{v}"][0].tolist(), "mean": float(np.nanmean(res[f"{s}_{v}"][0])),
                                "n": int(len(sub)), "n_species": int(sub.species.nunique())}

    def con(terms):
        # signed sum of conditions, e.g. (F - M) on Belly minus (F - M) on Back
        r = summary((sum(sg * res[k][0] for k, sg in terms), sum(sg * res[k][1] for k, sg in terms)))
        r["delta"] = r.pop("auc")
        return r

    diffs = {f"{v}: F - M": con([(f"F_{v}", 1), (f"M_{v}", -1)]) for v in VIEWS}
    for s in SEXES:
        for v in ("Belly", "Side"):
            diffs[f"{s}: {v} - Back"] = con([(f"{s}_{v}", 1), (f"{s}_Back", -1)])
    for v in ("Belly", "Side"):
        diffs[f"F - M: {v} - Back"] = con([(f"F_{v}", 1), (f"M_{v}", -1), ("F_Back", -1), ("M_Back", 1)])
    out[model] = {"rows": rows, "diffs": diffs}
    print(f"{model}: " + "  ".join(f"{k} {v['mean']:+.3f}" for k, v in diffs.items()), flush=True)

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/viewsex_groups.json", "w"), indent=1)
