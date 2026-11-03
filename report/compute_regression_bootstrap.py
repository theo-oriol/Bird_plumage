"""Family-level bootstrap for the regression-head (reg_prob) per-habitat AUC
benchmark: per-cell 95% CIs, paired delta vs reference (same images, same
draws), and paired main - sham difference. Families are the resampling unit
(same convention as src/evaluation/metrics.bootstrap_delta_auc - the folds
hold out whole families, so images within a family are not independent).

Weighted Mann-Whitney AUC: scores are sorted once per cell, and each
bootstrap draw is just a per-image weight vector (= how many times that
image's family was drawn), so a draw costs a few vectorised cumsums instead
of a re-sort. One model resident in memory at a time.
"""
import os
import gc
import json
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from paths import RUNS_DIR, LABELS_JSON
BASE = RUNS_DIR
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
B = 500
CHUNK = 50
SEED = 12345
hab_names = json.load(open(LABELS_JSON))
NH = len(hab_names)

SEED_GROUPS = {
    "A3_h56": ["A3_h56_r1", "A3_h56_r2", "A3_h56_r3"],
    "A3_h16": ["A3_h16_r1", "A3_h16_r2", "A3_h16_r3"],
    "A3_h4": ["A3_h4_r1", "A3_h4_r2", "A3_h4_r3"],
    "A3_h1": ["A3_h1_r1", "A3_h1_r2", "A3_h1_r3"],
}
SINGLE = ["reference", "A1", "A2", "A5", "A7",
          "A4_theta0_L17", "A4_theta0_L41", "A4_theta0_L101",
          "A4_theta90_L17", "A4_theta90_L41", "A4_theta90_L101",
          "A6_blue", "A6_purple", "A6_red", "A6_yellow", "A6_green", "A6_rufous", "A6_light_brown", "A6_dark_brown"]
ROWS = SINGLE + list(SEED_GROUPS)


def load(model):
    dfs = []
    for k in range(3):
        with open(f"{BASE}/{model}_fold{k}/eval/predictions.pkl", "rb") as f:
            dfs.append(pickle.load(f))
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    data = {}
    for cond, sub in df.groupby("condition_name", sort=True):
        hab = np.stack(sub["habitat"].to_numpy()).astype(np.float32)
        data[cond] = {
            "stem": sub["stem"].to_numpy(),
            "fam": sub["family"].to_numpy(),
            "y": hab > 0,
            "hab": hab if cond == "reference" else None,
            "s": np.stack(sub["reg_prob"].to_numpy()).astype(np.float64),
        }
    del df
    gc.collect()
    return data


def make_wauc(counts):
    def wauc(s, y, fidx):
        if y.all() or (~y).all():
            return np.nan, np.full(B, np.nan)
        order = np.argsort(s, kind="mergesort")
        s_s, y_s, f_s = s[order], y[order], fidx[order]
        starts = np.flatnonzero(np.r_[True, s_s[1:] != s_s[:-1]])

        def compute(w):
            wp = w * y_s
            wn = w * (~y_s)
            pg = np.add.reduceat(wp, starts, axis=1)
            ng = np.add.reduceat(wn, starts, axis=1)
            below = np.cumsum(ng, axis=1) - ng
            num = (pg * (below + 0.5 * ng)).sum(1)
            den = pg.sum(1) * ng.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return num / den

        point = compute(np.ones((1, len(s))))[0]
        boots = np.empty(B)
        for i in range(0, B, CHUNK):
            boots[i:i + CHUNK] = compute(counts[i:i + CHUNK][:, f_s].astype(np.float64))
        return point, boots
    return wauc


def per_cond(data, fam2i, wauc):
    """returns cond -> (P[NH], BT[NH,B]); plus A6 -> reference restricted to the same images"""
    ref = data["reference"]
    ref_idx = {s: i for i, s in enumerate(ref["stem"])}
    res, ref_sub = {}, {}
    for cond, d in data.items():
        t0 = time.time()
        fidx = np.array([fam2i[f] for f in d["fam"]], dtype=np.int32)
        P, BT = np.full(NH, np.nan), np.full((NH, B), np.nan)
        for h in range(NH):
            P[h], BT[h] = wauc(d["s"][:, h], d["y"][:, h], fidx)
        res[cond] = (P, BT)
        if cond.startswith("A6"):
            ii = np.array([ref_idx[s] for s in d["stem"]])
            Pr, BTr = np.full(NH, np.nan), np.full((NH, B), np.nan)
            for h in range(NH):
                Pr[h], BTr[h] = wauc(ref["s"][ii, h], ref["y"][ii, h], fidx)
            ref_sub[cond] = (Pr, BTr)
        print(f"  {cond:18s} n={len(d['s']):7d}  {time.time() - t0:5.1f}s", flush=True)
    return res, ref_sub


def ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5], axis=-1)
    return lo, hi


def summarize(res, ref_sub, n_per_cond):
    """23 rows: point/CI for AUC and for delta vs reference (paired, same images)."""
    out = {}
    P_ref, BT_ref = res["reference"]
    for row in ROWS:
        members = SEED_GROUPS.get(row, [row])
        P = np.nanmean([res[m][0] for m in members], axis=0)
        BT = np.nanmean([res[m][1] for m in members], axis=0)
        if row.startswith("A6"):
            Pr, BTr = ref_sub[row]
        else:
            Pr, BTr = P_ref, BT_ref
        dP, dBT = P - Pr, BT - BTr
        valid = ~np.isnan(P)
        mP, mBT = np.nanmean(P[valid]), np.nanmean(BT[valid], axis=0)
        mdP, mdBT = np.nanmean(dP[valid]), np.nanmean(dBT[valid], axis=0)
        lo, hi = ci(BT)
        dlo, dhi = ci(dBT)
        mlo, mhi = ci(mBT)
        mdlo, mdhi = ci(mdBT)
        out[row] = {
            "n": int(n_per_cond[members[0]]),
            "auc": P.tolist(), "auc_lo": lo.tolist(), "auc_hi": hi.tolist(),
            "delta": dP.tolist(), "delta_lo": dlo.tolist(), "delta_hi": dhi.tolist(),
            "macro": float(mP), "macro_lo": float(mlo), "macro_hi": float(mhi),
            "macro_delta": float(mdP), "macro_delta_lo": float(mdlo), "macro_delta_hi": float(mdhi),
            "_BT": BT, "_mBT": mBT,
        }
    return out


t_start = time.time()

# ---- main ----
print("=== main: loading ===", flush=True)
data = load("main")
fams = sorted(set(data["reference"]["fam"]))
fam2i = {f: i for i, f in enumerate(fams)}
rng = np.random.default_rng(SEED)
counts = np.stack([np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)) for _ in range(B)]).astype(np.float32)
wauc = make_wauc(counts)
print(f"  {len(fams)} families, B={B}", flush=True)

# sanity check against sklearn on one cell
d = data["reference"]
sk = roc_auc_score(d["y"][:, 0], d["s"][:, 0])
mine, _ = wauc(d["s"][:, 0], d["y"][:, 0], np.array([fam2i[f] for f in d["fam"]]))
print(f"  sanity: sklearn={sk:.6f}  weighted={mine:.6f}", flush=True)
assert abs(sk - mine) < 1e-9

hab_ref = data["reference"]["hab"]
y_ref = data["reference"]["y"]
prevalence = y_ref.mean(0).tolist()
mean_share = [float(hab_ref[y_ref[:, h], h].mean()) for h in range(NH)]
n_main = {c: len(v["s"]) for c, v in data.items()}

res, ref_sub = per_cond(data, fam2i, wauc)
del data
gc.collect()
main_rows = summarize(res, ref_sub, n_main)
del res, ref_sub
gc.collect()
print(f"=== main done ({time.time() - t_start:.0f}s) ===", flush=True)

# ---- sham ----
print("=== sham: loading ===", flush=True)
data = load("sham")
assert set(data["reference"]["fam"]) <= set(fams)
n_sham = {c: len(v["s"]) for c, v in data.items()}
res, ref_sub = per_cond(data, fam2i, wauc)
del data
gc.collect()
sham_rows = summarize(res, ref_sub, n_sham)
del res, ref_sub
gc.collect()
print(f"=== sham done ({time.time() - t_start:.0f}s) ===", flush=True)

# ---- paired main - sham ----
diff = {}
for row in ROWS:
    m, s = main_rows[row], sham_rows[row]
    dBT = m["_BT"] - s["_BT"]
    lo, hi = ci(dBT)
    mlo, mhi = ci(m["_mBT"] - s["_mBT"])
    diff[row] = {
        "diff": (np.array(m["auc"]) - np.array(s["auc"])).tolist(), "diff_lo": lo.tolist(), "diff_hi": hi.tolist(),
        "macro": m["macro"] - s["macro"], "macro_lo": float(mlo), "macro_hi": float(mhi),
    }

for r in (main_rows, sham_rows):
    for v in r.values():
        v.pop("_BT")
        v.pop("_mBT")

json.dump({
    "habitat_names": hab_names, "B": B, "n_families": len(fams),
    "prevalence": prevalence, "mean_share": mean_share,
    "rows": ROWS, "main": main_rows, "sham": sham_rows, "diff": diff,
}, open(f"{OUT}/regression_bootstrap.json", "w"))
print(f"wrote regression_bootstrap.json  total {time.time() - t_start:.0f}s")
