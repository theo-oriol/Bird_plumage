import os
import gc
import json
import pickle
import time
import numpy as np
import pandas as pd

from paths import RUNS_DIR, PROJECT_ROOT, LABELS_JSON
from svlib import B, CHUNK, EPS, MIN_SIDE, GROUPS, GN, PAIRS, SP, bootstrap_counts, ci

# ── Settings ─────────────────────────────────────────────────────────────────
CROSSWALK = os.path.join(PROJECT_ROOT, "data", "raw", "crosswalk_image_habitat_v0.csv")
FAMILY_ORDER_JSON = os.path.join(PROJECT_ROOT, "data", "family_order.json")
SEED_GROUPS = {f"A3_h{h}": [f"A3_h{h}_r{r}" for r in (1, 2, 3)] for h in (56, 16, 4, 1)}
ABLATIONS = ["A1", "A2", "A5", "A7",
             "A4_theta0_L17", "A4_theta0_L41", "A4_theta0_L101",
             "A4_theta90_L17", "A4_theta90_L41", "A4_theta90_L101"] + list(SEED_GROUPS)
NAMES = json.load(open(LABELS_JSON))
METRIC_NAMES = list(NAMES) + [f"{GN[a]} vs {GN[b]}" for a, b in PAIRS]


# ── Family -> order dictionary ───────────────────────────────────────────────
def family_order():
    cw = pd.read_csv(CROSSWALK, sep=";")
    cw["family"] = cw.family.str.lower()
    n = cw.groupby("family").order.nunique()
    assert (n == 1).all(), f"families with several orders: {n[n > 1].to_dict()}"
    fo = dict(sorted(zip(cw.family, cw.order)))
    with open(FAMILY_ORDER_JSON, "w") as f:
        json.dump(fo, f, indent=1)
    return fo


# ── Loading ──────────────────────────────────────────────────────────────────
# Reference rows of every fold plus the asked conditions, aligned on the reference stems
def load(model, conds):
    stems, fams, folds, H, S = [], [], [], [], {c: [] for c in conds}
    for k in range(3):
        with open(f"{RUNS_DIR}/{model}_fold{k}/eval/predictions.pkl", "rb") as f:
            df = pickle.load(f)
        df = df[df.condition_name.isin(conds)]
        ref = df[df.condition_name == "reference"].set_index("stem")
        stems.append(ref.index.to_numpy())
        fams.append(ref.family.to_numpy())
        folds.append(np.full(len(ref), k))
        H.append(np.stack(ref.habitat.to_numpy()).astype(np.float32))
        for c, sub in df.groupby("condition_name"):
            sub = sub.set_index("stem").loc[ref.index]
            S[c].append(np.stack(sub.reg_prob.to_numpy()).astype(np.float32))
        del df, ref
        gc.collect()
    return {"stem": np.concatenate(stems), "fam": np.concatenate(fams), "fold": np.concatenate(folds),
            "H": np.concatenate(H), "S": {c: np.concatenate(v) for c, v in S.items()}}


# ── Order-only baseline ──────────────────────────────────────────────────────
# Mean training label of the bird's order (other folds = training); unseen order -> training mean
def order_baseline(H, order, fold):
    P = np.empty_like(H)
    for k in range(3):
        tr, va = fold != k, fold == k
        means = pd.DataFrame(H[tr]).groupby(order[tr]).mean()
        glob = H[tr].mean(0)
        P[va] = np.stack([means.loc[o].to_numpy() if o in means.index else glob for o in order[va]])
    return P


# ── Stratified family-bootstrap AUC ──────────────────────────────────────────
# AUC over positive/negative pairs of the same stratum only (one stratum = the pooled AUC)
def make_sauc(counts):
    def sauc(s, y, fidx, strata, boot=True):
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

        point = compute(np.ones((1, len(s))))[0]
        if not boot:
            return point, None
        boots = np.empty(B)
        for i in range(0, B, CHUNK):
            boots[i:i + CHUNK] = compute(counts[i:i + CHUNK][:, f_s].astype(np.float64))
        return point, boots
    return sauc


# Per habitat (10) then per group pair (10), as in svlib.metrics, within the given strata
def metrics(H, S, fidx, strata, sauc, boot=True):
    P, BT = np.full(20, np.nan), np.full((20, B), np.nan)
    gs = np.stack([H[:, GROUPS[g]].sum(1) for g in GN], 1).argmax(1)
    go = np.log(np.stack([S[:, GROUPS[g]].sum(1) for g in GN], 1) + EPS)
    tasks = [(H[:, h] > 0, S[:, h], np.ones(len(H), bool)) for h in range(10)]
    for a, b in PAIRS:
        m = (gs == a) | (gs == b)
        tasks.append((gs[m] == a, go[m, a] - go[m, b], m))
    for t, (y, s, m) in enumerate(tasks):
        if y.sum() >= MIN_SIDE and (~y).sum() >= MIN_SIDE:
            p, bt = sauc(s.astype(np.float64), y, fidx[m], strata[m], boot)
            P[t] = p
            if boot:
                BT[t] = bt
    return P, BT


def avg_seeds(res, c):
    if c not in SEED_GROUPS:
        return res[c]
    ps, bs = zip(*[res[r] for r in SEED_GROUPS[c]])
    return np.mean(ps, 0), np.mean(bs, 0)


def summary(p, bt):
    lo, hi = ci(bt)
    return {"auc": p.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    fo = family_order()
    print(f"family_order.json: {len(fo)} families, {len(set(fo.values()))} orders")
    out = {"metrics": METRIC_NAMES}

    # Sham, clean images: order-only baseline, pooled vs. within-order AUC
    d = load("sham", ["reference"])
    order = np.array([fo[f] for f in d["fam"]])
    fams = np.unique(d["fam"])
    fidx = np.searchsorted(fams, d["fam"]).astype(np.int32)
    sauc = make_sauc(bootstrap_counts(fams))
    one = np.zeros(len(order), np.int32)
    oidx = np.unique(order, return_inverse=True)[1].astype(np.int32)
    base = order_baseline(d["H"], order, d["fold"])
    vals, counts = np.unique(order, return_counts=True)
    n_fam = pd.Series(d["fam"]).groupby(order).nunique()
    out["orders"] = {o: {"images": int(c), "families": int(n_fam[o])} for o, c in zip(vals, counts)}
    pas = order == "PASSERIFORMES"
    rows = {
        "order_only": metrics(d["H"], base, fidx, one, sauc),
        "order_only_within": metrics(d["H"], base, fidx, oidx, sauc),
        "sham": metrics(d["H"], d["S"]["reference"], fidx, one, sauc),
        "sham_within": metrics(d["H"], d["S"]["reference"], fidx, oidx, sauc),
        "sham_passeriformes": metrics(d["H"][pas], d["S"]["reference"][pas], fidx[pas], one[pas], sauc),
    }
    out["clean"] = {k: summary(*v) for k, v in rows.items()}
    print(f"sham done ({time.time() - t0:.0f} s)")
    del d
    gc.collect()

    # Main, ablations: Δ vs. clean, pooled and within order, paired on the same resamples
    conds = ["reference"] + [c for a in ABLATIONS for c in SEED_GROUPS.get(a, [a])]
    d = load("main", conds)
    assert (np.unique(d["fam"]) == fams).all()
    order = np.array([fo[f] for f in d["fam"]])
    oidx = np.unique(order, return_inverse=True)[1].astype(np.int32)
    res = {"pooled": {}, "within": {}}
    for c in conds:
        res["pooled"][c] = metrics(d["H"], d["S"][c], fidx, one, sauc)
        res["within"][c] = metrics(d["H"], d["S"][c], fidx, oidx, sauc)
        print(f"  main {c} ({time.time() - t0:.0f} s)")
    out["ablation"] = {}
    for mode in ("pooled", "within"):
        rp, rb = res[mode]["reference"]
        out["ablation"][mode] = {"reference": summary(rp, rb)}
        for a in ABLATIONS:
            p, bt = avg_seeds(res[mode], a)
            out["ablation"][mode][a] = {**summary(p, bt), "delta": (p - rp).tolist(),
                                        "delta_lo": ci(bt - rb)[0].tolist(), "delta_hi": ci(bt - rb)[1].tolist()}
    with open(os.path.join(SP, "order_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    report(out)
    print(f"done ({time.time() - t0:.0f} s): results/order_check.json")


# ── Printed summary ──────────────────────────────────────────────────────────
def report(out):
    c = out["clean"]
    print(f"\n{'':38s}{'order only':>12s}{'sham':>8s}{'sham within':>13s}{'within lo-hi':>15s}{'passerif.':>11s}")
    for i, n in enumerate(out["metrics"]):
        w = c["sham_within"]
        print(f"{n[:37]:38s}{c['order_only']['auc'][i]:12.3f}{c['sham']['auc'][i]:8.3f}"
              f"{w['auc'][i]:13.3f}{w['lo'][i]:8.3f}-{w['hi'][i]:.3f}{c['sham_passeriformes']['auc'][i]:11.3f}")
    a = out["ablation"]
    dp = np.array([a["pooled"][k]["delta"] for k in ABLATIONS])
    dw = np.array([a["within"][k]["delta"] for k in ABLATIONS])
    sp = np.array([(np.array(a["pooled"][k]["delta_hi"]) < 0) | (np.array(a["pooled"][k]["delta_lo"]) > 0) for k in ABLATIONS])
    sw = np.array([(np.array(a["within"][k]["delta_hi"]) < 0) | (np.array(a["within"][k]["delta_lo"]) > 0) for k in ABLATIONS])
    ok = ~np.isnan(dp) & ~np.isnan(dw)
    print(f"\nmain, Δ vs. clean{'':21s}{'pooled':>8s}{'within':>8s}  (mean over the 20 metrics)")
    for k, x, y in zip(ABLATIONS, np.nanmean(dp, 1), np.nanmean(dw, 1)):
        print(f"{k:38s}{x:8.3f}{y:8.3f}")
    print(f"correlation pooled vs. within Δ: r = {np.corrcoef(dp[ok], dw[ok])[0, 1]:.3f}; "
          f"same sign {np.mean(np.sign(dp[ok]) == np.sign(dw[ok])):.0%}; "
          f"significant pooled {sp[ok].sum()}, still significant within {(sp & sw)[ok].sum()}")


if __name__ == "__main__":
    main()
