import gc
import json
import numpy as np
import pandas as pd
from svlib import SP, B, load, bootstrap_counts, make_wauc, metrics, ci

from paths import DIMORPHISM_CSV

# ── Dimorphism index ─────────────────────────────────────────────────────────
DIM = DIMORPHISM_CSV
dim = pd.concat([pd.read_csv(DIM.format(v)).set_index("species").dimorphism.rename(v) for v in ("back", "belly")], axis=1)
dim["index"] = dim.mean(1, skipna=False)
dim = dim.dropna()  # 1,623 species have no index (fewer than 2 images in a sex)

# ── Per model ────────────────────────────────────────────────────────────────
out = {"B": B}
counts = None
for model in ("main", "sham"):
    d = load(model)
    if counts is None:
        fams = sorted(d.family.unique())
        fam2i = {f: i for i, f in enumerate(fams)}
        wauc = make_wauc(bootstrap_counts(fams))
        counts = True
    d["fidx"] = d.family.map(fam2i).astype(np.int32)
    both = d[d.sex.isin(["M", "F"])].groupby("species").sex.nunique()
    sp = sorted(set(both.index[both == 2]) & set(dim.index))
    q = np.quantile(dim.loc[sp, "index"], [1 / 3, 2 / 3])
    tert = pd.cut(dim.loc[sp, "index"], [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
    d = d[d.species.isin(sp)].copy()
    d["tert"] = d.species.map(tert)
    res, rows = {}, {}
    for t in ("low", "mid", "high"):
        for sx in ("M", "F"):
            sub = d[(d.tert == t) & (d.sex == sx)]
            H = np.stack(sub.habitat.to_numpy()).astype(np.float32)
            S = np.stack(sub.reg_prob.to_numpy()).astype(np.float64)
            res[(t, sx)] = metrics(H, S, sub.fidx.to_numpy(), wauc)
            rows[f"{t}_{sx}"] = {"auc": res[(t, sx)][0].tolist(), "n": int(len(sub)),
                                 "n_species": int(sub.species.nunique()), "n_fam": int(sub.family.nunique())}
        print(model, t, rows[f"{t}_F"]["n_species"], "species", flush=True)
    gaps = {}
    for t in ("low", "mid", "high"):
        P = res[(t, "F")][0] - res[(t, "M")][0]
        BT = res[(t, "F")][1] - res[(t, "M")][1]
        lo, hi = ci(BT)
        gaps[t] = {"delta": P.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                   "mean_hab": float(np.nanmean(P[:10])), "mean_hab_ci": ci(np.nanmean(BT[:10], 0)).tolist(),
                   "mean_pair": float(np.nanmean(P[10:])), "mean_pair_ci": ci(np.nanmean(BT[10:], 0)).tolist(),
                   "boot_hab": np.nanmean(BT[:10], 0), "boot_pair": np.nanmean(BT[10:], 0)}
    trend = {}
    for k in ("hab", "pair"):
        D = gaps["high"][f"boot_{k}"] - gaps["low"][f"boot_{k}"]
        trend[k] = {"delta": gaps["high"][f"mean_{k}"] - gaps["low"][f"mean_{k}"], "ci": ci(D).tolist()}
    for t in gaps:
        gaps[t].pop("boot_hab")
        gaps[t].pop("boot_pair")
    out[model] = {"rows": rows, "gaps": gaps, "trend": trend, "cuts": q.tolist(),
                  "tert_range": {t: [float(dim.loc[tert.index[tert == t], "index"].min()),
                                     float(dim.loc[tert.index[tert == t], "index"].max())] for t in ("low", "mid", "high")}}
    del d
    gc.collect()

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/sex_dimorph.json", "w"), indent=1)
for model in ("main", "sham"):
    print(f"== {model}  cuts {np.round(out[model]['cuts'], 3)}")
    for t, g in out[model]["gaps"].items():
        print(f"  {t:5s} F-M habitats {g['mean_hab']:+.3f} {np.round(g['mean_hab_ci'], 3)}  pairs {g['mean_pair']:+.3f} {np.round(g['mean_pair_ci'], 3)}")
    for k, v in out[model]["trend"].items():
        print(f"  trend high-low {k}: {v['delta']:+.3f} {np.round(v['ci'], 3)}")
