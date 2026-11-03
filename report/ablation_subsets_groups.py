import gc
import json
import pickle
import sys
import time
import numpy as np
import pandas as pd
from svlib import BASE, SP, B, GN, parse_stems, bootstrap_counts, make_wauc, group_ovr

from paths import DIMORPHISM_CSV

# ── Settings ─────────────────────────────────────────────────────────────────
DIM = DIMORPHISM_CSV
SEED_GROUPS = {f"A3_h{h}": [f"A3_h{h}_r{r}" for r in (1, 2, 3)] for h in (56, 16, 4, 1)}
SINGLE = ["A1", "A2", "A5", "A7", "A4_theta0_L17", "A4_theta0_L41", "A4_theta0_L101",
          "A4_theta90_L17", "A4_theta90_L41", "A4_theta90_L101",
          "A6_blue", "A6_purple", "A6_red", "A6_yellow", "A6_green", "A6_rufous", "A6_light_brown", "A6_dark_brown"]
ROWS = SINGLE + list(SEED_GROUPS)
TERTS, VIEWS, SEXES = ("low", "mid", "high"), ("Back", "Belly", "Side"), ("M", "F")
SUBSETS = (list(VIEWS) + list(SEXES) + list(TERTS) + [f"{t}_{s}" for t in TERTS for s in SEXES]
           + [f"{t}_{s}_{v}" for t in TERTS for s in SEXES for v in VIEWS])
# "viewsex": sex x view subsets only (species with both sexes), saved under another name
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
if MODE == "viewsex":
    SUBSETS = [f"{s}_{v}" for s in SEXES for v in VIEWS]
OUTN = "ablation_subsets_groups" if MODE == "all" else f"ablation_subsets_groups_{MODE}"

# ── Dimorphism tertiles ──────────────────────────────────────────────────────
dim = pd.concat([pd.read_csv(DIM.format(v)).set_index("species").dimorphism.rename(v) for v in ("back", "belly")], axis=1)
dim["index"] = dim.mean(1, skipna=False)
dim = dim.dropna()

# ── Load predictions ─────────────────────────────────────────────────────────
t0 = time.time()
parts = []
for k in range(3):
    df = pickle.load(open(f"{BASE}/main_fold{k}/eval/predictions.pkl", "rb"))
    parts.append(df[["stem", "family", "condition_name", "habitat", "reg_prob"]])
    del df
    gc.collect()
df = parse_stems(pd.concat(parts, ignore_index=True))
del parts
gc.collect()
fams = sorted(df.family.unique())
fam2i = {f: i for i, f in enumerate(fams)}
wauc = make_wauc(bootstrap_counts(fams))
r0 = df[df.condition_name == "reference"]
nsx = r0[r0.sex.isin(["M", "F"])].groupby("species").sex.nunique()
both = set(nsx.index[nsx == 2])
sp = sorted(both & set(dim.index))
q = np.quantile(dim.loc[sp, "index"], [1 / 3, 2 / 3])
tert = pd.cut(dim.loc[sp, "index"], [-np.inf, q[0], q[1], np.inf], labels=list(TERTS)).astype(str)
del r0
print(f"loaded ({time.time() - t0:.0f}s), cuts {np.round(q, 3)}", flush=True)

# ── Arrays per condition ─────────────────────────────────────────────────────
data = {}
for cond, sub in df.groupby("condition_name", sort=True):
    data[cond] = {"stem": sub.stem.to_numpy(), "fidx": sub.family.map(fam2i).to_numpy(np.int32),
                  "view": sub.view.to_numpy(), "sex": sub.sex.to_numpy(), "both": sub.species.isin(both).to_numpy(),
                  "tert": sub.species.map(tert).fillna("").to_numpy(),
                  "H": np.stack(sub.habitat.to_numpy()).astype(np.float32),
                  "S": np.stack(sub.reg_prob.to_numpy()).astype(np.float32)}
del df
gc.collect()


# ── Subsets ──────────────────────────────────────────────────────────────────
def mask(dd, s):
    if s in VIEWS:
        return dd["view"] == s
    if s in SEXES:
        return (dd["sex"] == s) & dd["both"]
    p = s.split("_")
    if p[0] in SEXES:
        return (dd["sex"] == p[0]) & dd["both"] & (dd["view"] == p[1])
    m = dd["tert"] == p[0]
    if len(p) > 1:
        m &= dd["sex"] == p[1]
    if len(p) > 2:
        m &= dd["view"] == p[2]
    return m


def run(dd, m):
    return group_ovr(dd["H"][m], dd["S"][m].astype(np.float64), dd["fidx"][m], wauc)


# ── Ablation cost per subset ─────────────────────────────────────────────────
ref = data["reference"]
ref_pos = {s: i for i, s in enumerate(ref["stem"])}
rows, boots, n_groups = {}, {}, {}
for s in SUBSETS:
    t1 = time.time()
    res = {"reference": run(ref, mask(ref, s))}
    for cond, dd in data.items():
        if cond == "reference":
            continue
        m = mask(dd, s)
        res[cond] = run(dd, m)
        if cond.startswith("A6"):
            keep = np.zeros(len(ref["stem"]), bool)
            keep[[ref_pos[x] for x in dd["stem"][m]]] = True
            res[(cond, "clean")] = run(ref, keep)
    P_ref, BT_ref = res["reference"]
    rows[s] = {"reference": {"auc": P_ref.tolist(), "mean": float(np.nanmean(P_ref)),
                             "n": int(mask(ref, s).sum())}}
    for row in ROWS:
        members = SEED_GROUPS.get(row, [row])
        with np.errstate(all="ignore"):
            P = np.nanmean([res[m][0] for m in members], axis=0)
            BT = np.nanmean([res[m][1] for m in members], axis=0)
        Pr, BTr = res[(row, "clean")] if row.startswith("A6") else (P_ref, BT_ref)
        ok = ~np.isnan(P - Pr)
        rows[s][row] = {"delta": (P - Pr).tolist(), "mean_delta": float(np.mean((P - Pr)[ok])) if ok.any() else None,
                        "n_groups": int(ok.sum())}
        boots[f"{s}|{row}"] = np.nanmean((BT - BTr)[ok], 0) if ok.any() else np.full(B, np.nan)
    del res
    gc.collect()
    print(f"  {s}: n={rows[s]['reference']['n']} clean {rows[s]['reference']['mean']:.3f}  A1 {rows[s]['A1']['mean_delta']:+.3f}  ({time.time() - t1:.0f}s)", flush=True)

# ── Write ────────────────────────────────────────────────────────────────────
np.savez_compressed(f"{SP}/{OUTN}.npz", **boots)
json.dump({"rows": rows, "subsets": SUBSETS, "ablations": ROWS, "groups": GN, "cuts": q.tolist(), "B": B,
           "n_species": {t: int((tert == t).sum()) for t in TERTS}}, open(f"{SP}/{OUTN}.json", "w"))
print(f"done ({time.time() - t0:.0f}s)", flush=True)
