import json
import numpy as np
from svlib import SP, B, TERTS, SEXES, load, arrays, family_auc, metrics, species_with_both_sexes, dimorphism_tertiles, ci

# ── Per model ────────────────────────────────────────────────────────────────
# Female - male AUC gap within each dimorphism tertile (species with both sexes), and its trend high - low
out = {"B": B}
wauc = None
for model in ("main", "sham"):
    d = load(model)
    if wauc is None:
        fam2i, _, wauc = family_auc(d.family)
    tert, q, dim = dimorphism_tertiles(species_with_both_sexes(d.species, d.sex))
    d = d[d.species.isin(tert.index)].copy()
    d["tert"] = d.species.map(tert)
    res, rows = {}, {}
    for t in TERTS:
        for sx in SEXES:
            sub = d[(d.tert == t) & (d.sex == sx)]
            H, S, F = arrays(sub)
            res[(t, sx)] = metrics(H, S, np.array([fam2i[x] for x in F], np.int32), wauc)
            rows[f"{t}_{sx}"] = {"auc": res[(t, sx)][0].tolist(), "n": int(len(sub)),
                                 "n_species": int(sub.species.nunique()), "n_fam": int(sub.family.nunique())}
    gaps, boot = {}, {}
    for t in TERTS:
        P, BT = res[(t, "F")][0] - res[(t, "M")][0], res[(t, "F")][1] - res[(t, "M")][1]
        boot[t] = {"hab": np.nanmean(BT[:10], 0), "pair": np.nanmean(BT[10:], 0)}
        gaps[t] = {"delta": P.tolist(), "lo": ci(BT)[0].tolist(), "hi": ci(BT)[1].tolist(),
                   "mean_hab": float(np.nanmean(P[:10])), "mean_hab_ci": ci(boot[t]["hab"]).tolist(),
                   "mean_pair": float(np.nanmean(P[10:])), "mean_pair_ci": ci(boot[t]["pair"]).tolist()}
    trend = {k: {"delta": gaps["high"][f"mean_{k}"] - gaps["low"][f"mean_{k}"], "ci": ci(boot["high"][k] - boot["low"][k]).tolist()}
             for k in ("hab", "pair")}
    out[model] = {"rows": rows, "gaps": gaps, "trend": trend, "cuts": q.tolist(),
                  "tert_range": {t: [float(dim.loc[tert.index[tert == t], "index"].min()),
                                     float(dim.loc[tert.index[tert == t], "index"].max())] for t in TERTS}}
    print(f"{model}: trend high - low habitats {trend['hab']['delta']:+.3f} {np.round(trend['hab']['ci'], 3)}", flush=True)

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/sex_dimorph.json", "w"), indent=1)
