import json
import numpy as np
from svlib import SP, B, GN, PAIRS, NAMES, VIEWS, SEXES, load, arrays, family_auc, metrics, species_with_both_sexes, ci

COMPARE = [("Belly", "Back"), ("Side", "Back"), ("Side", "Belly"),
           ("3 views", "Back"), ("3 views", "Belly"), ("3 views", "Side"), ("F", "M")]

# ── Per model ────────────────────────────────────────────────────────────────
# Habitat and group-pair AUCs per view, for the 3 views averaged per specimen, and per sex
out = {"labels": list(NAMES) + [f"{GN[a]} vs {GN[b]}" for a, b in PAIRS], "B": B}
wauc = None
for model in ("main", "sham"):
    d = load(model)
    if wauc is None:
        fam2i, _, wauc = family_auc(d.family)
    subsets = {v: d[d.view == v].sort_values("cat") for v in VIEWS}  # same specimen order in every view
    assert all((subsets[v].cat.to_numpy() == subsets["Back"].cat.to_numpy()).all() for v in VIEWS)
    both = species_with_both_sexes(d.species, d.sex)
    for sx in SEXES:
        subsets[sx] = d[(d.sex == sx) & d.species.isin(both)]
    res, sizes = {}, {}
    for name, sub in subsets.items():
        H, S, F = arrays(sub)
        f = np.array([fam2i[x] for x in F], np.int32)
        res[name] = metrics(H, S, f, wauc)
        sizes[name] = {"n": int(len(sub)), "n_species": int(sub.species.nunique()), "n_fam": int(sub.family.nunique())}
        if name == "Back":  # 3 views combined: mean output per specimen
            res["3 views"] = metrics(H, np.mean([np.stack(subsets[v].reg_prob.to_numpy()) for v in VIEWS], 0), f, wauc)
            sizes["3 views"] = sizes["Back"]
    rows = {k: {"auc": P.tolist(), "lo": ci(BT)[0].tolist(), "hi": ci(BT)[1].tolist(), **sizes[k]} for k, (P, BT) in res.items()}
    diffs = {}
    for a, b in COMPARE:
        P, D = res[a][0] - res[b][0], res[a][1] - res[b][1]
        diffs[f"{a} - {b}"] = {"delta": P.tolist(), "lo": ci(D)[0].tolist(), "hi": ci(D)[1].tolist(),
                               "mean_hab": float(np.nanmean(P[:10])), "mean_hab_ci": ci(np.nanmean(D[:10], 0)).tolist(),
                               "mean_pair": float(np.nanmean(P[10:])), "mean_pair_ci": ci(np.nanmean(D[10:], 0)).tolist()}
        print(f"{model} {a} - {b}: habitats {diffs[f'{a} - {b}']['mean_hab']:+.3f}  pairs {diffs[f'{a} - {b}']['mean_pair']:+.3f}", flush=True)
    out[model] = {"rows": rows, "diffs": diffs}

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/sex_view_check.json", "w"), indent=1)
