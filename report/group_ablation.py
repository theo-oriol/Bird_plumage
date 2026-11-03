"""Where does the group signal come from? Group-pair discriminability under each ablation.

Groups (from clustering the clean pairwise matrix):
  Forest | Savanna/Shrubland/Artificial | Grassland/Desert/Rocky | Sea (Neritic + Marine Intertidal) | Wetlands
Each image's main group = group with the largest summed label share. For every
pair of groups (G1, G2) and every condition: among images whose main group is G1
or G2, AUC of "main group = G1" ranked by log(sum of G1 outputs / sum of G2
outputs). Paired family bootstrap (B=500, same draws everywhere): CI of the AUC
and of the change vs. clean images (for A6, clean = same images). A3 seeds are
averaged. One model in memory at a time.
"""
import os
import gc
import json
import pickle
import time
import numpy as np

from paths import RUNS_DIR
BASE = RUNS_DIR
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
B, CHUNK, SEED, EPS = 500, 50, 12345, 1e-6
GROUPS = {"Forest": [0], "Savanna/Shrub/Artificial": [1, 2, 6], "Open/arid": [3, 4, 9], "Sea": [5, 7], "Wetlands": [8]}
GN = list(GROUPS)
PAIRS = [(i, j) for i in range(len(GN)) for j in range(i + 1, len(GN))]
SEED_GROUPS = {f"A3_h{h}": [f"A3_h{h}_r{r}" for r in (1, 2, 3)] for h in (56, 16, 4, 1)}
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
    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    data = {}
    for cond, sub in df.groupby("condition_name", sort=True):
        hab = np.stack(sub["habitat"].to_numpy()).astype(np.float32)
        s = np.stack(sub["reg_prob"].to_numpy()).astype(np.float64)
        gshare = np.stack([hab[:, GROUPS[g]].sum(1) for g in GN], 1)
        gout = np.log(np.stack([s[:, GROUPS[g]].sum(1) for g in GN], 1) + EPS)
        data[cond] = {"stem": sub["stem"].to_numpy(), "fam": sub["family"].to_numpy(),
                      "main_g": gshare.argmax(1), "gout": gout}
        del hab, s
    del df
    gc.collect()
    return data


def make_wauc(counts):
    def wauc(s, y, fidx):
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


def pair_aucs(d, fam2i, wauc, idx=None):
    """P[pair], BT[pair, B] for one condition (optionally restricted to rows idx)."""
    mg, go, fam = d["main_g"], d["gout"], d["fam"]
    if idx is not None:
        mg, go, fam = mg[idx], go[idx], fam[idx]
    fidx = np.array([fam2i[f] for f in fam], dtype=np.int32)
    P, BT = np.full(len(PAIRS), np.nan), np.full((len(PAIRS), B), np.nan)
    for p, (a, b) in enumerate(PAIRS):
        m = (mg == a) | (mg == b)
        y = mg[m] == a
        if y.sum() < 30 or (~y).sum() < 30:
            continue
        P[p], BT[p] = wauc(go[m, a] - go[m, b], y, fidx[m])
    return P, BT


def ci(x):
    return np.nanpercentile(x, [2.5, 97.5], axis=-1)


out = {"groups": GROUPS, "group_names": GN, "pairs": PAIRS, "rows": ROWS, "B": B}
fams = None
for model in ("main", "sham"):
    t0 = time.time()
    print(f"=== {model}: loading ===", flush=True)
    data = load(model)
    if fams is None:
        fams = sorted(set(data["reference"]["fam"]))
        rng = np.random.default_rng(SEED)
        counts = np.stack([np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)) for _ in range(B)]).astype(np.float32)
        fam2i = {f: i for i, f in enumerate(fams)}
        wauc = make_wauc(counts)
        ref = data["reference"]
        out["n_main_group"] = {GN[g]: int((ref["main_g"] == g).sum()) for g in range(len(GN))}
        out["n_fam_main_group"] = {GN[g]: int(len(set(ref["fam"][ref["main_g"] == g]))) for g in range(len(GN))}
    print(f"  loaded ({time.time() - t0:.0f}s)", flush=True)
    ref = data["reference"]
    ref_pos = {s: i for i, s in enumerate(ref["stem"])}
    res, ref_sub = {}, {}
    for cond, d in data.items():
        t1 = time.time()
        res[cond] = pair_aucs(d, fam2i, wauc)
        if cond.startswith("A6"):
            ii = np.array([ref_pos[s] for s in d["stem"]])
            ref_sub[cond] = pair_aucs(ref, fam2i, wauc, ii)
        print(f"  {cond:18s} n={len(d['stem']):7d}  {time.time() - t1:5.1f}s", flush=True)
    del data
    gc.collect()
    rows = {}
    P_ref, BT_ref = res["reference"]
    for row in ROWS:
        members = SEED_GROUPS.get(row, [row])
        P = np.nanmean([res[m][0] for m in members], axis=0)
        BT = np.nanmean([res[m][1] for m in members], axis=0)
        Pr, BTr = ref_sub[row] if row.startswith("A6") else (P_ref, BT_ref)
        lo, hi = ci(BT)
        dlo, dhi = ci(BT - BTr)
        rows[row] = {"auc": P.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                     "clean_same": Pr.tolist(), "delta": (P - Pr).tolist(), "dlo": dlo.tolist(), "dhi": dhi.tolist()}
    out[model] = rows
    del res, ref_sub
    gc.collect()
    print(f"=== {model} done ({time.time() - t0:.0f}s) ===", flush=True)

json.dump(out, open(f"{SP}/group_ablation.json", "w"))
print("wrote group_ablation.json")
