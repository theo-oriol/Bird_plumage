"""View x sex on clean images, 5 group AUCs of 5.1, both models.

Species with both sexes (as in 6.2), each sex x view subset. Contrasts (paired,
same family bootstrap draws): F - M within each view, Belly - Back and Side - Back
within each sex, and the interaction (F - M on belly) - (F - M on back).
"""
import gc
import json
import numpy as np
from svlib import SP, B, GN, load, bootstrap_counts, make_wauc, group_ovr, ci

VIEWS, SEXES = ("Back", "Belly", "Side"), ("M", "F")
out = {"groups": GN, "B": B}
wauc = None
for model in ("sham", "main"):
    d = load(model)
    if wauc is None:
        fams = sorted(d.family.unique())
        fam2i = {f: i for i, f in enumerate(fams)}
        wauc = make_wauc(bootstrap_counts(fams))
    d["fidx"] = d.family.map(fam2i).astype(np.int32)
    nsx = d[d.sex.isin(["M", "F"])].groupby("species").sex.nunique()
    d = d[d.species.isin(set(nsx.index[nsx == 2])) & d.sex.isin(["M", "F"])]
    res, rows = {}, {}
    for s in SEXES:
        for v in VIEWS:
            sub = d[(d.sex == s) & (d.view == v)]
            res[f"{s}_{v}"] = group_ovr(np.stack(sub.habitat.to_numpy()).astype(np.float32),
                                        np.stack(sub.reg_prob.to_numpy()).astype(np.float64), sub.fidx.to_numpy(), wauc)
            rows[f"{s}_{v}"] = {"auc": res[f"{s}_{v}"][0].tolist(), "mean": float(np.nanmean(res[f"{s}_{v}"][0])),
                                "n": int(len(sub)), "n_species": int(sub.species.nunique())}
    del d
    gc.collect()

    def con(terms):
        P = sum(sg * res[k][0] for k, sg in terms)
        BT = sum(sg * res[k][1] for k, sg in terms)
        lo, hi = ci(BT)
        return {"delta": P.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "mean": float(np.nanmean(P)), "mean_ci": ci(np.nanmean(BT, 0)).tolist()}

    diffs = {}
    for v in VIEWS:
        diffs[f"{v}: F - M"] = con([(f"F_{v}", 1), (f"M_{v}", -1)])
    for s in SEXES:
        for v in ("Belly", "Side"):
            diffs[f"{s}: {v} - Back"] = con([(f"{s}_{v}", 1), (f"{s}_Back", -1)])
    for v in ("Belly", "Side"):
        diffs[f"F - M: {v} - Back"] = con([(f"F_{v}", 1), (f"M_{v}", -1), (f"F_Back", -1), (f"M_Back", 1)])
    out[model] = {"rows": rows, "diffs": diffs}
    print(f"\n== {model} ==  " + "  ".join(f"{k} {v['mean']:.3f}" for k, v in rows.items()))
    for k, v in diffs.items():
        per = "  ".join(f"{g[:8]} {x:+.3f}{'*' if (l > 0 or h < 0) else ' '}" for g, x, l, h in zip(GN, v["delta"], v["lo"], v["hi"]))
        print(f"{k:22s} mean {v['mean']:+.3f} [{v['mean_ci'][0]:+.3f}, {v['mean_ci'][1]:+.3f}]   {per}")
json.dump(out, open(f"{SP}/viewsex_groups.json", "w"), indent=1)
