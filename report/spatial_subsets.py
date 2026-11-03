"""Cost of removing the spatial structure, measured without the shuffled-image bias.

Pixel shuffling (A3 h1) keeps exactly the colour/lightness histogram of the crops.
Instead of asking the network to read shuffled images, the histogram model of 5.3
(MLP on the histogram of the same 16 crops, same family folds) reads that same
information. Cost of the spatial structure in a subset = group AUC of the
histogram model - group AUC of the main network on clean images (5 group AUCs of
5.1, averaged). Subsets as in section 6. Family bootstrap B=500, same draws, so
contrasts between subsets are paired. Saves point values and bootstrap draws.
"""
import json
import numpy as np
import pandas as pd
from svlib import SP, B, GN, load, bootstrap_counts, make_wauc, group_ovr

from paths import DIMORPHISM_CSV
DIM = DIMORPHISM_CSV
TERTS, VIEWS, SEXES = ("low", "mid", "high"), ("Back", "Belly", "Side"), ("M", "F")
SUBSETS = (list(VIEWS) + list(SEXES) + [f"{s}_{v}" for s in SEXES for v in VIEWS] + list(TERTS)
           + [f"{t}_{s}" for t in TERTS for s in SEXES] + [f"{t}_{s}_{v}" for t in TERTS for s in SEXES for v in VIEWS])
SOURCES = {"all": "", "L": "_L", "ab": "_ab"}  # full histogram, lightness only, colour only

d = load("main")
stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True)
pos = {s: i for i, s in enumerate(stems)}
ii = d.stem.map(pos).to_numpy()
assert not np.isnan(ii.astype(float)).any()
hist = {k: np.load(f"{SP}/hist_pred_crops{s}_mlp.npy")[ii].astype(np.float64) for k, s in SOURCES.items()}
fams = sorted(d.family.unique())
fam2i = {f: i for i, f in enumerate(fams)}
wauc = make_wauc(bootstrap_counts(fams))
F = d.family.map(fam2i).to_numpy(np.int32)
H = np.stack(d.habitat.to_numpy()).astype(np.float32)
S = np.stack(d.reg_prob.to_numpy()).astype(np.float64)

dim = pd.concat([pd.read_csv(DIM.format(v)).set_index("species").dimorphism.rename(v) for v in ("back", "belly")], axis=1)
dim["index"] = dim.mean(1, skipna=False)
dim = dim.dropna()
nsx = d[d.sex.isin(["M", "F"])].groupby("species").sex.nunique()
both = set(nsx.index[nsx == 2])
sp = sorted(both & set(dim.index))
q = np.quantile(dim.loc[sp, "index"], [1 / 3, 2 / 3])
tert = pd.cut(dim.loc[sp, "index"], [-np.inf, q[0], q[1], np.inf], labels=list(TERTS)).astype(str)
view, sex = d.view.to_numpy(), d.sex.to_numpy()
isboth = d.species.isin(both).to_numpy()
tr = d.species.map(tert).fillna("").to_numpy()


def mask(s):
    if s in VIEWS:
        return view == s
    if s in SEXES:
        return (sex == s) & isboth
    p = s.split("_")
    if p[0] in SEXES:
        return (sex == p[0]) & isboth & (view == p[1])
    m = tr == p[0]
    if len(p) > 1:
        m &= sex == p[1]
    if len(p) > 2:
        m &= view == p[2]
    return m


rows, boots = {}, {}
for s in SUBSETS:
    m = mask(s)
    Pn, Bn = group_ovr(H[m], S[m], F[m], wauc)
    rows[s] = {"n": int(m.sum()), "net": float(np.nanmean(Pn))}
    for k in SOURCES:
        Ph, Bh = group_ovr(H[m], hist[k][m], F[m], wauc)
        ok = ~np.isnan(Pn - Ph)
        rows[s][f"hist_{k}"] = float(np.mean(Ph[ok]))
        rows[s][f"cost_{k}"] = float(np.mean((Ph - Pn)[ok]))   # hist - net: what is lost without the spatial structure
        boots[f"{s}|{k}"] = np.nanmean((Bh - Bn)[ok], 0)
    print(f"{s:16s} n={rows[s]['n']:6d}  net {rows[s]['net']:.3f}  hist {rows[s]['hist_all']:.3f}  cost {rows[s]['cost_all']:+.3f}", flush=True)
np.savez_compressed(f"{SP}/spatial_subsets.npz", **boots)
json.dump({"rows": rows, "subsets": SUBSETS, "B": B, "groups": GN}, open(f"{SP}/spatial_subsets.json", "w"), indent=1)
