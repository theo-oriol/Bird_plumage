import gc
import json
import numpy as np
import pandas as pd
from svlib import SP, B, GN, load, bootstrap_counts, make_wauc, group_ovr, ci

from paths import DIMORPHISM_CSV

# ── Dimorphism index ─────────────────────────────────────────────────────────
DIM = DIMORPHISM_CSV
dim = pd.concat([pd.read_csv(DIM.format(v)).set_index("species").dimorphism.rename(v) for v in ("back", "belly")], axis=1)
dim["index"] = dim.mean(1, skipna=False)
dim = dim.dropna()


# ── Helpers ──────────────────────────────────────────────────────────────────
def arr(sub):
    return np.stack(sub.habitat.to_numpy()).astype(np.float32), np.stack(sub.reg_prob.to_numpy()).astype(np.float64), sub.fidx.to_numpy()


def diff(a, b):
    P, BT = a[0] - b[0], a[1] - b[1]
    lo, hi = ci(BT)
    mb = np.nanmean(BT, 0)
    return {"delta": P.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "mean": float(np.nanmean(P)), "mean_ci": ci(mb).tolist()}, mb


# ── Per model ────────────────────────────────────────────────────────────────
out = {"groups": GN, "B": B}
wauc = None
for model in ("sham", "main"):
    d = load(model)
    if wauc is None:
        fams = sorted(d.family.unique())
        fam2i = {f: i for i, f in enumerate(fams)}
        wauc = make_wauc(bootstrap_counts(fams))
    d["fidx"] = d.family.map(fam2i).astype(np.int32)
    res = {}
    views = {v: d[d.view == v].sort_values("cat") for v in ("Back", "Belly", "Side")}
    for v, sub in views.items():
        res[v] = group_ovr(*arr(sub), wauc)
    H, _, f = arr(views["Back"])
    S3 = np.mean([np.stack(views[v].reg_prob.to_numpy()) for v in views], 0)
    res["3 views"] = group_ovr(H, S3, f, wauc)
    both = d[d.sex.isin(["M", "F"])].groupby("species").sex.nunique()
    both = set(both.index[both == 2])
    for sx in ("M", "F"):
        res[sx] = group_ovr(*arr(d[(d.sex == sx) & d.species.isin(both)]), wauc)
    sp = sorted(both & set(dim.index))
    q = np.quantile(dim.loc[sp, "index"], [1 / 3, 2 / 3])
    tert = pd.cut(dim.loc[sp, "index"], [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
    dd = d[d.species.isin(sp)].copy()
    dd["tert"] = dd.species.map(tert)
    for t in ("low", "mid", "high"):
        for sx in ("M", "F"):
            res[f"{t}_{sx}"] = group_ovr(*arr(dd[(dd.tert == t) & (dd.sex == sx)]), wauc)
    del d, dd, views
    gc.collect()
    rows = {k: {"auc": P.tolist(), "mean": float(np.nanmean(P))} for k, (P, _) in res.items()}
    diffs = {}
    for a, b in [("Belly", "Back"), ("Side", "Back"), ("3 views", "Back"), ("3 views", "Belly"), ("3 views", "Side"), ("F", "M")]:
        diffs[f"{a} - {b}"] = diff(res[a], res[b])[0]
    gap_boot = {}
    for t in ("low", "mid", "high"):
        diffs[f"{t}: F - M"], gap_boot[t] = diff(res[f"{t}_F"], res[f"{t}_M"])
    diffs["trend high - low"] = {"mean": diffs["high: F - M"]["mean"] - diffs["low: F - M"]["mean"],
                                 "mean_ci": ci(gap_boot["high"] - gap_boot["low"]).tolist()}
    out[model] = {"rows": rows, "diffs": diffs}
    print(f"\n== {model} ==  clean group AUCs: " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in rows.items() if "_" not in k))
    for k, v in diffs.items():
        per = "  ".join(f"{g[:8]} {x:+.3f}{'*' if (l > 0 or h < 0) else ' '}" for g, x, l, h in zip(GN, v["delta"], v["lo"], v["hi"])) if "delta" in v else ""
        print(f"{k:18s} mean {v['mean']:+.3f} [{v['mean_ci'][0]:+.3f}, {v['mean_ci'][1]:+.3f}]   {per}")

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/sex_groups.json", "w"), indent=1)
