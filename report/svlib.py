import gc
import json
import os
import pickle

import numpy as np
import pandas as pd

from paths import RUNS_DIR, LABELS_JSON, DIMORPHISM_CSV

# ── Settings ─────────────────────────────────────────────────────────────────
# Shared by every script: paths, bootstrap, habitat groups, ablation conditions
BASE = RUNS_DIR
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
NAMES = json.load(open(LABELS_JSON))
NH = len(NAMES)
B, CHUNK, SEED, EPS, MIN_SIDE = 500, 50, 12345, 1e-6, 30
GROUPS = {"Forest": [0], "Savanna/Shrub/Artificial": [1, 2, 6], "Open/arid": [3, 4, 9], "Sea": [5, 7], "Wetlands": [8]}
GN = list(GROUPS)
PAIRS = [(i, j) for i in range(len(GN)) for j in range(i + 1, len(GN))]
# ablation table rows: single conditions, then A3 averaged over its three random seeds
SEED_GROUPS = {f"A3_h{h}": [f"A3_h{h}_r{r}" for r in (1, 2, 3)] for h in (56, 16, 4, 1)}
ABLATIONS = (["A1", "A2", "A5", "A7"] + [f"A4_theta{t}_L{l}" for t in (0, 90) for l in (17, 41, 101)]
             + [f"A6_{c}" for c in ("blue", "purple", "red", "yellow", "green", "rufous", "light_brown", "dark_brown")]
             + list(SEED_GROUPS))
TERTS, VIEWS, SEXES = ("low", "mid", "high"), ("Back", "Belly", "Side"), ("M", "F")


# ── Loading ──────────────────────────────────────────────────────────────────
def parse_stems(d):
    # stem = Genus_epithet_replicate_sex_view_spectrum_catalog
    p = d.stem.str.split("_", expand=True)
    d["species"], d["sex"], d["view"], d["cat"] = p[0] + "_" + p[1], p[3], p[4], p[6]
    return d


def load(model, conds=("reference",), cols=("stem", "family", "habitat", "reg_prob", "condition_name")):
    parts = []
    for k in range(3):
        df = pickle.load(open(f"{BASE}/{model}_fold{k}/eval/predictions.pkl", "rb"))
        parts.append(df.loc[df.condition_name.isin(conds), list(cols)].copy())
        del df
        gc.collect()
    return parse_stems(pd.concat(parts, ignore_index=True))


def arrays(d):
    # labels H, outputs S, families F of a predictions table
    return (np.stack(d.habitat.to_numpy()).astype(np.float32), np.stack(d.reg_prob.to_numpy()).astype(np.float64),
            d.family.to_numpy())


def load_conditions(model, s_dtype=np.float64):
    # every evaluation condition of one model: {condition: stem, species, view, sex, F, fidx, H, S};
    # fidx indexes the sorted families of the clean images, as family_auc does
    parts = []
    for k in range(3):
        df = pickle.load(open(f"{BASE}/{model}_fold{k}/eval/predictions.pkl", "rb"))
        parts.append(df[["stem", "family", "condition_name", "habitat", "reg_prob"]])
        del df
        gc.collect()
    df = parse_stems(pd.concat(parts, ignore_index=True))
    del parts
    fam2i = {f: i for i, f in enumerate(sorted(set(df.family[df.condition_name == "reference"])))}
    data = {}
    for cond, sub in df.groupby("condition_name", sort=True):
        data[cond] = {"stem": sub.stem.to_numpy(), "species": sub.species.to_numpy(), "view": sub.view.to_numpy(),
                      "sex": sub.sex.to_numpy(), "F": sub.family.to_numpy(),
                      "fidx": sub.family.map(fam2i).to_numpy(np.int32),
                      "H": np.stack(sub.habitat.to_numpy()).astype(np.float32),
                      "S": np.stack(sub.reg_prob.to_numpy()).astype(s_dtype)}
    del df
    gc.collect()
    return data


def load_hist_images(models=("main",), conds=("reference",)):
    # images of the histogram models (hist_stems.npy order): labels H, family F, fold, network outputs net[(model, cond)]
    stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True)
    pos = {s: i for i, s in enumerate(stems)}
    N = len(stems)
    H, F, fold = np.zeros((N, NH), np.float32), np.empty(N, object), np.full(N, -1)
    net = {(m, c): np.full((N, NH), np.nan) for m in models for c in conds}
    for model in models:
        for k in range(3):
            df = pickle.load(open(f"{BASE}/{model}_fold{k}/eval/predictions.pkl", "rb"))
            df = df[df.condition_name.isin(set(conds) | {"reference"})]
            for c, sub in df.groupby("condition_name"):
                ii = np.array([pos[s] for s in sub.stem])
                if (model, c) in net:
                    net[(model, c)][ii] = np.stack(sub.reg_prob.to_numpy())
                if c == "reference":
                    H[ii], F[ii], fold[ii] = np.stack(sub.habitat.to_numpy()), sub.family.to_numpy(), k
            del df
            gc.collect()
    return H, F, fold, net


def hist_pred(name):
    # out-of-fold predictions of a histogram model ("" = colour + lightness, "_L", "_ab", "_A2", "_A2_L")
    return np.load(f"{SP}/hist_pred_crops{name}_mlp.npy")


# ── Family bootstrap and AUCs ────────────────────────────────────────────────
def bootstrap_counts(fams):
    # B resamples of whole families: how many times each family is drawn
    rng = np.random.default_rng(SEED)
    return np.stack([np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams))
                     for _ in range(B)]).astype(np.float32)


def family_auc(F):
    # families sorted, family index per image, and the bootstrap AUC function
    fams = sorted(set(F))
    fam2i = {f: i for i, f in enumerate(fams)}
    return fam2i, np.array([fam2i[f] for f in F], np.int32), make_wauc(bootstrap_counts(fams))


def make_wauc(counts):
    # AUC with family weights, vectorised over the B resamples; equal to
    # sklearn.metrics.roc_auc_score(y, s, sample_weight=w) for each resample (ties count half)
    def wauc(s, y, fidx):
        order = np.argsort(s, kind="mergesort")
        s_s, y_s, f_s = s[order], y[order], fidx[order]
        starts = np.flatnonzero(np.r_[True, s_s[1:] != s_s[:-1]])

        def compute(w):
            pg = np.add.reduceat(w * y_s, starts, axis=1)
            ng = np.add.reduceat(w * ~y_s, starts, axis=1)
            below = np.cumsum(ng, axis=1) - ng
            with np.errstate(invalid="ignore", divide="ignore"):
                return (pg * (below + 0.5 * ng)).sum(1) / (pg.sum(1) * ng.sum(1))

        boots = np.empty(B)
        for i in range(0, B, CHUNK):
            boots[i:i + CHUNK] = compute(counts[i:i + CHUNK][:, f_s].astype(np.float64))
        return compute(np.ones((1, len(s))))[0], boots
    return wauc


def make_sauc(counts):
    # same AUC, but only pairs of images from the same stratum (here: the same fold) are compared
    def sauc(s, y, fidx, strata):
        o = np.lexsort((s, strata))
        s_s, y_s, f_s, g_s = s[o], y[o], fidx[o], strata[o]
        starts = np.flatnonzero(np.r_[True, (s_s[1:] != s_s[:-1]) | (g_s[1:] != g_s[:-1])])
        gs = g_s[starts]
        first = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        sid = np.repeat(np.arange(len(first)), np.diff(np.r_[first, len(starts)]))

        def compute(w):
            pg = np.add.reduceat(w * y_s, starts, axis=1)
            ng = np.add.reduceat(w * ~y_s, starts, axis=1)
            below = np.cumsum(ng, axis=1) - ng
            below -= below[:, first][:, sid]
            num = (pg * (below + 0.5 * ng)).sum(1)
            den = (np.add.reduceat(pg, first, axis=1) * np.add.reduceat(ng, first, axis=1)).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return num / den

        boots = np.empty(B)
        for i in range(0, B, CHUNK):
            boots[i:i + CHUNK] = compute(counts[i:i + CHUNK][:, f_s].astype(np.float64))
        return compute(np.ones((1, len(s))))[0], boots
    return sauc


def ci(x):
    return np.nanpercentile(x, [2.5, 97.5], axis=-1)


def summary(r):
    # r = (point, bootstrap): AUCs with 95% CIs, and their mean with its CI
    P, BT = r
    lo, hi = ci(BT)
    return {"auc": P.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "mean": float(np.nanmean(P)),
            "mean_ci": ci(np.nanmean(BT, 0)).tolist()}


def compare(a, b):
    # paired difference a - b (same bootstrap resamples): per AUC and on the mean
    D = a[1] - b[1]
    lo, hi = ci(D)
    return {"delta": (a[0] - b[0]).tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "mean": float(np.nanmean(a[0] - b[0])),
            "mean_ci": ci(np.nanmean(D, 0)).tolist()}


def seed_mean(results):
    # mean of (point, bootstrap) over the A3 random seeds
    return np.mean([r[0] for r in results], 0), np.mean([r[1] for r in results], 0)


# ── AUC sets ─────────────────────────────────────────────────────────────────
def main_group(H):
    return np.stack([H[:, GROUPS[g]].sum(1) for g in GN], 1).argmax(1)


def group_log_outputs(S):
    return np.log(np.stack([S[:, GROUPS[g]].sum(1) for g in GN], 1) + EPS)


def pair_aucs(H, S, fidx, wauc):
    # group pairs (10): images of the two main groups, score = log(output G1) - log(output G2)
    gs, go = main_group(H), group_log_outputs(S)
    P, BT = np.full(len(PAIRS), np.nan), np.full((len(PAIRS), B), np.nan)
    for p, (a, b) in enumerate(PAIRS):
        m = (gs == a) | (gs == b)
        y = gs[m] == a
        if y.sum() >= MIN_SIDE and (~y).sum() >= MIN_SIDE:
            P[p], BT[p] = wauc(go[m, a] - go[m, b], y, fidx[m])
    return P, BT


def metrics(H, S, fidx, wauc, habitats=True):
    # AUCs per habitat (10, if asked) then per group pair (10): point and bootstrap [., B]
    P, BT = pair_aucs(H, S, fidx, wauc)
    if not habitats:
        return P, BT
    Ph, BTh = np.full(NH, np.nan), np.full((NH, B), np.nan)
    for h in range(NH):
        y = H[:, h] > 0
        if y.sum() >= MIN_SIDE and (~y).sum() >= MIN_SIDE:
            Ph[h], BTh[h] = wauc(S[:, h], y, fidx)
    return np.r_[Ph, P], np.r_[BTh, BT]


def three_levels(H, S, fidx, auc):
    # AUCs at the three levels of section 5.3: 10 habitats, 5 groups (one vs. rest), 10 group pairs
    hp, hb = metrics(H, S, fidx, auc)
    return {"habitats": (hp[:NH], hb[:NH]), "groups": group_ovr(H, S, fidx, auc, "main"), "pairs": (hp[NH:], hb[NH:])}


def group_ovr(H, S, fidx, wauc, mode="main"):
    # one-vs-rest AUC per group (5); mode 'main': largest label share, 'presence': share > 0
    gshare = np.stack([H[:, GROUPS[g]].sum(1) for g in GN], 1)
    gs = gshare.argmax(1)
    P, BT = np.full(len(GN), np.nan), np.full((len(GN), B), np.nan)
    for g in range(len(GN)):
        y = gs == g if mode == "main" else gshare[:, g] > 0
        if y.sum() >= MIN_SIDE and (~y).sum() >= MIN_SIDE:
            P[g], BT[g] = wauc(S[:, GROUPS[GN[g]]].sum(1), y, fidx)
    return P, BT


# ── Ablation table ───────────────────────────────────────────────────────────
def per_condition(data, fn):
    # fn(H, S, fidx) -> (point, bootstrap) for every condition; A6 (one colour) is also scored on the
    # clean images of the same birds, its paired reference
    ref = data["reference"]
    pos = {s: i for i, s in enumerate(ref["stem"])}
    res, clean = {}, {}
    for cond, d in data.items():
        res[cond] = fn(d["H"], d["S"], d["fidx"])
        if cond.startswith("A6"):
            ii = np.array([pos[s] for s in d["stem"]])
            clean[cond] = fn(ref["H"][ii], ref["S"][ii], ref["fidx"][ii])
    return res, clean


def ablation_row(res, clean, row):
    # one table row: AUCs (mean over the A3 seeds) and the clean AUCs on the same images
    members = SEED_GROUPS.get(row, [row])
    with np.errstate(all="ignore"):
        P = np.nanmean([res[m][0] for m in members], axis=0)
        BT = np.nanmean([res[m][1] for m in members], axis=0)
    Pr, BTr = clean.get(row, res["reference"])
    return P, BT, Pr, BTr


# ── Section 6 subsets ────────────────────────────────────────────────────────
def species_with_both_sexes(species, sex):
    d = pd.DataFrame({"species": species, "sex": sex})
    n = d[d.sex.isin(SEXES)].groupby("species").sex.nunique()
    return set(n.index[n == 2])


def dimorphism_tertiles(both):
    # dimorphism index = mean of back and belly; tertiles over the species with both sexes and an index
    dim = pd.concat([pd.read_csv(DIMORPHISM_CSV.format(v)).set_index("species").dimorphism.rename(v)
                     for v in ("back", "belly")], axis=1)
    dim["index"] = dim.mean(1, skipna=False)
    dim = dim.dropna()  # species with fewer than 2 images in a sex have no index
    sp = sorted(both & set(dim.index))
    q = np.quantile(dim.loc[sp, "index"], [1 / 3, 2 / 3])
    tert = pd.cut(dim.loc[sp, "index"], [-np.inf, q[0], q[1], np.inf], labels=list(TERTS)).astype(str)
    return tert, q, dim


def subset_mask(s, view, sex, both, tert):
    # subset names: a view, a sex, sex_view, tertile, tertile_sex, tertile_sex_view
    if s in VIEWS:
        return view == s
    if s in SEXES:
        return (sex == s) & both
    p = s.split("_")
    if p[0] in SEXES:
        return (sex == p[0]) & both & (view == p[1])
    m = tert == p[0]
    if len(p) > 1:
        m &= sex == p[1]
    if len(p) > 2:
        m &= view == p[2]
    return m
