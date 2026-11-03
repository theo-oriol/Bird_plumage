"""Shared pieces for the sex / view / dimorphism analyses (same definitions as
sex_view_check.py and group_ablation.py)."""
import os
import gc
import pickle
import numpy as np
import pandas as pd

from paths import RUNS_DIR
BASE = RUNS_DIR
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
B, CHUNK, SEED, EPS, MIN_SIDE = 500, 50, 12345, 1e-6, 30
GROUPS = {"Forest": [0], "Savanna/Shrub/Artificial": [1, 2, 6], "Open/arid": [3, 4, 9], "Sea": [5, 7], "Wetlands": [8]}
GN = list(GROUPS)
PAIRS = [(i, j) for i in range(len(GN)) for j in range(i + 1, len(GN))]


def parse_stems(d):
    p = d.stem.str.split("_", expand=True)
    d["species"], d["sex"], d["view"], d["cat"] = p[0] + "_" + p[1], p[3], p[4], p[6]
    return d


def load(model, conds=("reference",), cols=("stem", "family", "habitat", "reg_prob", "condition_name")):
    parts = []
    for k in range(3):
        df = pickle.load(open(f"{BASE}/{model}_fold{k}/eval/predictions.pkl", "rb"))
        r = df.loc[df.condition_name.isin(conds), list(cols)].copy()
        del df
        gc.collect()
        parts.append(r)
    return parse_stems(pd.concat(parts, ignore_index=True))


def bootstrap_counts(fams):
    rng = np.random.default_rng(SEED)
    return np.stack([np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)) for _ in range(B)]).astype(np.float32)


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


def metrics(H, S, fidx, wauc, habitats=True):
    """per-habitat AUCs (10, if habitats) then group-pair AUCs (10): point, boots [., B]."""
    nh = 10 if habitats else 0
    P, BT = np.full(nh + 10, np.nan), np.full((nh + 10, B), np.nan)
    for h in range(nh):
        y = H[:, h] > 0
        if y.sum() >= MIN_SIDE and (~y).sum() >= MIN_SIDE:
            P[h], BT[h] = wauc(S[:, h], y, fidx)
    gs = np.stack([H[:, GROUPS[g]].sum(1) for g in GN], 1).argmax(1)
    go = np.log(np.stack([S[:, GROUPS[g]].sum(1) for g in GN], 1) + EPS)
    for p, (a, b) in enumerate(PAIRS):
        m = (gs == a) | (gs == b)
        y = gs[m] == a
        if y.sum() >= MIN_SIDE and (~y).sum() >= MIN_SIDE:
            P[nh + p], BT[nh + p] = wauc(go[m, a] - go[m, b], y, fidx[m])
    return P, BT


def ci(x):
    return np.nanpercentile(x, [2.5, 97.5], axis=-1)


def group_ovr(H, S, fidx, wauc, mode="main"):
    """5 one-vs-rest group AUCs; score = summed output of G's habitats.
    mode "main": truth = image's main group is G (largest summed label share).
    mode "presence": truth = summed label share of G > 0 (as per-habitat presence AUC).
    point [5], boots [5, B]."""
    gshare = np.stack([H[:, GROUPS[g]].sum(1) for g in GN], 1)
    gs = gshare.argmax(1)
    P, BT = np.full(len(GN), np.nan), np.full((len(GN), B), np.nan)
    for g in range(len(GN)):
        y = gs == g if mode == "main" else gshare[:, g] > 0
        if y.sum() >= MIN_SIDE and (~y).sum() >= MIN_SIDE:
            P[g], BT[g] = wauc(S[:, GROUPS[GN[g]]].sum(1), y, fidx)
    return P, BT
