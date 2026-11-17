import gc
import json
import sys
import numpy as np
from svlib import SP, B, GN, ABLATIONS, TERTS, VIEWS, SEXES, load_conditions, family_auc, group_ovr, per_condition, \
    ablation_row, species_with_both_sexes, dimorphism_tertiles, subset_mask

# argv[1] 'viewsex': sex x view subsets only (species with both sexes), saved under another name
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
SUBSETS = ([f"{s}_{v}" for s in SEXES for v in VIEWS] if MODE == "viewsex" else
           list(VIEWS) + list(SEXES) + list(TERTS) + [f"{t}_{s}" for t in TERTS for s in SEXES]
           + [f"{t}_{s}_{v}" for t in TERTS for s in SEXES for v in VIEWS])
OUTN = "ablation_subsets_groups" if MODE == "all" else f"ablation_subsets_groups_{MODE}"

# ── Load ─────────────────────────────────────────────────────────────────────
data = load_conditions("main", s_dtype=np.float32)
ref = data["reference"]
_, _, wauc = family_auc(ref["F"])
both = species_with_both_sexes(ref["species"], ref["sex"])
tert, q, _ = dimorphism_tertiles(both)
for d in data.values():
    d["both"] = np.isin(d["species"], list(both))
    d["tert"] = np.array([tert.get(s, "") for s in d["species"]])
score = lambda H, S, f: group_ovr(H, S.astype(np.float64), f, wauc)

# ── Ablation cost per subset ─────────────────────────────────────────────────
# group AUCs on the subset under every ablation; cost = ablated - clean (A6: clean images of the same birds)
rows, boots = {}, {}
for s in SUBSETS:
    sub = {}
    for cond, d in data.items():
        m = subset_mask(s, d["view"], d["sex"], d["both"], d["tert"])
        sub[cond] = {k: d[k][m] for k in ("stem", "fidx", "H", "S")}
    res, clean = per_condition(sub, score)
    P_ref = res["reference"][0]
    rows[s] = {"reference": {"auc": P_ref.tolist(), "mean": float(np.nanmean(P_ref)), "n": int(len(sub["reference"]["stem"]))}}
    for row in ABLATIONS:
        P, BT, Pr, BTr = ablation_row(res, clean, row)
        ok = ~np.isnan(P - Pr)
        rows[s][row] = {"delta": (P - Pr).tolist(), "mean_delta": float(np.mean((P - Pr)[ok])) if ok.any() else None,
                        "n_groups": int(ok.sum())}
        boots[f"{s}|{row}"] = np.nanmean((BT - BTr)[ok], 0) if ok.any() else np.full(B, np.nan)
    del sub, res, clean
    gc.collect()
    print(f"  {s}: n={rows[s]['reference']['n']} clean {rows[s]['reference']['mean']:.3f}  A1 {rows[s]['A1']['mean_delta']:+.3f}", flush=True)

# ── Write ────────────────────────────────────────────────────────────────────
np.savez_compressed(f"{SP}/{OUTN}.npz", **boots)
json.dump({"rows": rows, "subsets": SUBSETS, "ablations": ABLATIONS, "groups": GN, "cuts": q.tolist(), "B": B,
           "n_species": {t: int((tert == t).sum()) for t in TERTS}}, open(f"{SP}/{OUTN}.json", "w"))
print(f"wrote {OUTN}.json")
