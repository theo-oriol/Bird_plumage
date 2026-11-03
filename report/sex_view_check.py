import os
import gc
import json
import pickle
import numpy as np
import pandas as pd

from paths import RUNS_DIR, LABELS_JSON

# ── Settings ─────────────────────────────────────────────────────────────────
BASE = RUNS_DIR
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
B, CHUNK, SEED, EPS, MIN_SIDE = 500, 50, 12345, 1e-6, 30
GROUPS = {"Forest": [0], "Savanna/Shrub/Artificial": [1, 2, 6], "Open/arid": [3, 4, 9], "Sea": [5, 7], "Wetlands": [8]}
GN = list(GROUPS)
PAIRS = [(i, j) for i in range(len(GN)) for j in range(i + 1, len(GN))]
COMPARE = [("Belly", "Back"), ("Side", "Back"), ("Side", "Belly"),
           ("3 views", "Back"), ("3 views", "Belly"), ("3 views", "Side"), ("F", "M")]


# ── Helpers ──────────────────────────────────────────────────────────────────
def load(model):
    parts = []
    for k in range(3):
        df = pickle.load(open(f"{BASE}/{model}_fold{k}/eval/predictions.pkl", "rb"))
        r = df.loc[df.condition_name == "reference", ["stem", "family", "habitat", "reg_prob"]].copy()
        del df
        gc.collect()
        parts.append(r)
    d = pd.concat(parts, ignore_index=True)
    p = d.stem.str.split("_", expand=True)
    d["species"], d["sex"], d["view"], d["cat"] = p[0] + "_" + p[1], p[3], p[4], p[6]
    return d


def make_wauc(counts):
    def wauc(s, y, fidx):
        order = np.argsort(s, kind="mergesort")
        s_s, y_s, f_s = s[order], y[order], fidx[order]
        starts = np.flatnonzero(np.r_[True, s_s[1:] != s_s[:-1]])

        def compute(w):
            pg = np.add.reduceat(w * y_s, starts, axis=1)
            ng = np.add.reduceat(w * (~y_s), starts, axis=1)
            below = np.cumsum(ng, axis=1) - ng
            with np.errstate(invalid="ignore", divide="ignore"):
                return (pg * (below + 0.5 * ng)).sum(1) / (pg.sum(1) * ng.sum(1))

        boots = np.empty(B)
        for i in range(0, B, CHUNK):
            boots[i:i + CHUNK] = compute(counts[i:i + CHUNK][:, f_s].astype(np.float64))
        return compute(np.ones((1, len(s))))[0], boots
    return wauc


# AUCs per habitat (10, if asked) then per group pair (10): point and bootstrap [., B]
def metrics(H, S, fidx, wauc):
    P, BT = np.full(20, np.nan), np.full((20, B), np.nan)
    for h in range(10):
        y = H[:, h] > 0
        if y.sum() >= MIN_SIDE and (~y).sum() >= MIN_SIDE:
            P[h], BT[h] = wauc(S[:, h], y, fidx)
    gs = np.stack([H[:, GROUPS[g]].sum(1) for g in GN], 1).argmax(1)
    go = np.log(np.stack([S[:, GROUPS[g]].sum(1) for g in GN], 1) + EPS)
    for p, (a, b) in enumerate(PAIRS):
        m = (gs == a) | (gs == b)
        y = gs[m] == a
        if y.sum() >= MIN_SIDE and (~y).sum() >= MIN_SIDE:
            P[10 + p], BT[10 + p] = wauc(go[m, a] - go[m, b], y, fidx[m])
    return P, BT


# ── Per model ────────────────────────────────────────────────────────────────
ci = lambda x: np.nanpercentile(x, [2.5, 97.5], axis=-1)
names = json.load(open(LABELS_JSON))
out = {"labels": list(names) + [f"{GN[a]} vs {GN[b]}" for a, b in PAIRS], "B": B}
counts = fam2i = None
for model in ("main", "sham"):
    d = load(model)
    if counts is None:
        fams = sorted(d.family.unique())
        fam2i = {f: i for i, f in enumerate(fams)}
        rng = np.random.default_rng(SEED)
        counts = np.stack([np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)) for _ in range(B)]).astype(np.float32)
        wauc = make_wauc(counts)
    d["fidx"] = d.family.map(fam2i).astype(np.int32)

    subsets = {}
    for v in ("Back", "Belly", "Side"):
        subsets[v] = d[d.view == v].sort_values("cat")
    # 3 views combined: mean output per specimen (same order as the per-view subsets)
    back = subsets["Back"]
    S3 = np.mean([np.stack(subsets[v].reg_prob.to_numpy()) for v in ("Back", "Belly", "Side")], 0)
    assert all((subsets[v].cat.to_numpy() == back.cat.to_numpy()).all() for v in ("Belly", "Side"))
    both = d[d.sex.isin(["M", "F"])].groupby("species").sex.nunique()
    both = set(both.index[both == 2])
    for sx in ("M", "F"):
        subsets[sx] = d[(d.sex == sx) & d.species.isin(both)]

    res, sizes = {}, {}
    for name, sub in subsets.items():
        H = np.stack(sub.habitat.to_numpy()).astype(np.float32)
        S = np.stack(sub.reg_prob.to_numpy()).astype(np.float64)
        res[name] = metrics(H, S, sub.fidx.to_numpy(), wauc)
        sizes[name] = {"n": int(len(sub)), "n_species": int(sub.species.nunique()), "n_fam": int(sub.family.nunique())}
        if name == "Back":
            res["3 views"] = metrics(H, S3, sub.fidx.to_numpy(), wauc)
            sizes["3 views"] = sizes["Back"]
        print(model, name, sizes[name], np.round(np.nanmean(res[name][0][:10]), 3), flush=True)
    del d, subsets
    gc.collect()

    rows = {}
    for name, (P, BT) in res.items():
        lo, hi = ci(BT)
        rows[name] = {"auc": P.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), **sizes[name]}
    diffs = {}
    for a, b in COMPARE:
        D = res[a][1] - res[b][1]
        lo, hi = ci(D)
        mh = np.nanmean(D[:10], 0)
        mg = np.nanmean(D[10:], 0)
        diffs[f"{a} - {b}"] = {"delta": (res[a][0] - res[b][0]).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                               "mean_hab": float(np.nanmean(res[a][0][:10] - res[b][0][:10])), "mean_hab_ci": ci(mh).tolist(),
                               "mean_pair": float(np.nanmean(res[a][0][10:] - res[b][0][10:])), "mean_pair_ci": ci(mg).tolist()}
    out[model] = {"rows": rows, "diffs": diffs}
    del res
    gc.collect()

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/sex_view_check.json", "w"), indent=1)
for model in ("main", "sham"):
    print(f"\n== {model} ==")
    for k, v in out[model]["diffs"].items():
        print(f"{k:18s} habitats {v['mean_hab']:+.3f} {np.round(v['mean_hab_ci'], 3)}   pairs {v['mean_pair']:+.3f} {np.round(v['mean_pair_ci'], 3)}")
