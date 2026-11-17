import json
import numpy as np
from svlib import SP, B, GN, TERTS, VIEWS, SEXES, load, arrays, family_auc, group_ovr, hist_pred, \
    species_with_both_sexes, dimorphism_tertiles, subset_mask

SUBSETS = (list(VIEWS) + list(SEXES) + [f"{s}_{v}" for s in SEXES for v in VIEWS] + list(TERTS)
           + [f"{t}_{s}" for t in TERTS for s in SEXES] + [f"{t}_{s}_{v}" for t in TERTS for s in SEXES for v in VIEWS])
SOURCES = {"all": "", "L": "_L", "ab": "_ab"}  # full histogram, lightness only, colour only

# ── Load ─────────────────────────────────────────────────────────────────────
d = load("main")
pos = {s: i for i, s in enumerate(np.load(f"{SP}/hist_stems.npy", allow_pickle=True))}
ii = d.stem.map(pos).to_numpy()
hist = {k: hist_pred(s)[ii].astype(np.float64) for k, s in SOURCES.items()}
H, S, F = arrays(d)
_, fidx, wauc = family_auc(F)
both = species_with_both_sexes(d.species, d.sex)
tert, _, _ = dimorphism_tertiles(both)
view, sex, isboth = d.view.to_numpy(), d.sex.to_numpy(), d.species.isin(both).to_numpy()
tr = d.species.map(tert).fillna("").to_numpy()

# ── Per subset ───────────────────────────────────────────────────────────────
# upper bound of what spatial structure adds: group AUCs of the network vs. the histogram models (no layout)
rows, boots = {}, {}
for s in SUBSETS:
    m = subset_mask(s, view, sex, isboth, tr)
    Pn, Bn = group_ovr(H[m], S[m], fidx[m], wauc)
    rows[s] = {"n": int(m.sum()), "net": float(np.nanmean(Pn))}
    for k in SOURCES:
        Ph, Bh = group_ovr(H[m], hist[k][m], fidx[m], wauc)
        ok = ~np.isnan(Pn - Ph)
        rows[s][f"hist_{k}"] = float(np.mean(Ph[ok]))
        rows[s][f"cost_{k}"] = float(np.mean((Ph - Pn)[ok]))  # hist - net: what is lost without the spatial structure
        boots[f"{s}|{k}"] = np.nanmean((Bh - Bn)[ok], 0)
    print(f"{s:16s} n={rows[s]['n']:6d}  net {rows[s]['net']:.3f}  hist {rows[s]['hist_all']:.3f}  cost {rows[s]['cost_all']:+.3f}", flush=True)

# ── Write ────────────────────────────────────────────────────────────────────
np.savez_compressed(f"{SP}/spatial_subsets.npz", **boots)
json.dump({"rows": rows, "subsets": SUBSETS, "B": B, "groups": GN}, open(f"{SP}/spatial_subsets.json", "w"), indent=1)
