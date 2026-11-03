"""Histogram-only baseline vs. the network on pixel-shuffled images (A3_h1).

A3_h1 keeps only the distribution of pixel values. If a model that sees only
that distribution (L*a*b* histograms, hist_features.py) separates the habitat
groups as well as the network does on A3_h1 images, the network's A3_h1 drop is
a genuine loss of spatial information, not the backbone being confused by
unnatural images.

Baseline: predicts the 10-habitat share distribution (soft-label cross-entropy),
a 1-hidden-layer MLP, trained on the network's own 3 folds
(train on two test folds, predict the third; 10% of training families held out
for early stopping). Then the same group-pair AUCs as group_ablation.py, family
bootstrap B=500 with the same draws, paired CIs for network minus baseline.
"""
import os
os.environ.setdefault("KMP_AFFINITY", "disabled")
import gc
import glob
import json
import pickle
import sys
import numpy as np
import torch

torch.set_num_threads(4)
from paths import PROJECT_ROOT, RUNS_DIR
ROOT = PROJECT_ROOT
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
B, CHUNK, SEED, EPS = 500, 50, 12345, 1e-6
GROUPS = {"Forest": [0], "Savanna/Shrub/Artificial": [1, 2, 6], "Open/arid": [3, 4, 9], "Sea": [5, 7], "Wetlands": [8]}
GN = list(GROUPS)
PAIRS = [(i, j) for i in range(len(GN)) for j in range(i + 1, len(GN))]
CONDS = ["reference", "A3_h1_r1", "A3_h1_r2", "A3_h1_r3"]
# "crops" = the 16 eval crops (fair comparison with A3_h1); "whole" = whole bird
FEAT = sys.argv[1] if len(sys.argv) > 1 else "crops"
CHUNKS = {"crops": "hist_crop_chunks", "whole": "hist_chunks"}[FEAT]

stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True)
X = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{SP}/{CHUNKS}/*.npy"))])
assert len(X) == len(stems), (len(X), len(stems))
pos = {s: i for i, s in enumerate(stems)}
N = len(stems)

# network outputs (reference + A3_h1 seeds), labels, family, fold - aligned to stems
net = {}
H = np.zeros((N, 10), np.float32)
fam = np.empty(N, object)
fold = np.full(N, -1)
for model in ("main", "sham"):
    out = {c: np.full((N, 10), np.nan) for c in CONDS}
    for k in range(3):
        df = pickle.load(open(f"{RUNS_DIR}/{model}_fold{k}/eval/predictions.pkl", "rb"))
        df = df[df.condition_name.isin(CONDS)]
        for c, sub in df.groupby("condition_name"):
            ii = np.array([pos[s] for s in sub.stem])
            out[c][ii] = np.stack(sub.reg_prob.to_numpy())
            if model == "main" and c == "reference":
                H[ii] = np.stack(sub.habitat.to_numpy())
                fam[ii] = sub.family.to_numpy()
                fold[ii] = k
        del df
        gc.collect()
    net[model] = out
assert (fold >= 0).all()

# ---- baseline fits ----
def fit_predict(kind, Xtr, Ytr, Xva, Yva, Xte, seed=0):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    if kind == "linear":
        m = torch.nn.Linear(d, 10)
    else:
        m = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 10))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr, Xva, Yva = map(torch.from_numpy, (Xtr, Ytr, Xva, Yva))
    best, best_state, bad = np.inf, None, 0
    for ep in range(200):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 512):
            j = perm[i:i + 512]
            loss = -(Ytr[j] * torch.log_softmax(m(Xtr[j]), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = -(Yva * torch.log_softmax(m(Xva), 1)).sum(1).mean().item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(torch.from_numpy(Xte)), 1).numpy().astype(np.float64), best, ep


# layout: 512 joint bins, then per channel (L, a, b, chroma) 32 bins + mean, std + 5 percentiles
is_bin = np.r_[np.ones(512, bool), np.tile(np.r_[np.ones(32, bool), np.zeros(7, bool)], 4)]
assert len(is_bin) == X.shape[1]
Xl = X.copy()
Xl[:, is_bin] = np.log(X[:, is_bin] + 1e-4)  # histograms are skewed; log-compress the bins only
Xl = Xl.astype(np.float32)
# feature subsets (argv[2]): "L" = lightness only, "ab" = colour only, default = all
SUBSET = sys.argv[2] if len(sys.argv) > 2 else "all"
if SUBSET == "L":
    Xl = Xl[:, 512:551]
elif SUBSET == "ab":
    ab = X[:, :512].reshape(-1, 8, 8, 8).sum(1).reshape(-1, 64)  # joint a*b*, summed over L*
    Xl = np.concatenate([np.log(ab + 1e-4), Xl[:, 551:]], 1).astype(np.float32)
if SUBSET != "all":
    FEAT = f"{FEAT}_{SUBSET}"
print("features", FEAT, Xl.shape, flush=True)
base = {"mlp": np.zeros((N, 10))}
fit_log = []
for k in range(3):
    tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    rng = np.random.default_rng(k)
    tf = np.unique(fam[tr])
    va_f = set(rng.choice(tf, int(0.1 * len(tf)), replace=False))
    va = tr[np.array([f in va_f for f in fam[tr]])]
    tr = tr[np.array([f not in va_f for f in fam[tr]])]
    mu, sd = Xl[tr].mean(0), Xl[tr].std(0) + 1e-6
    Z = ((Xl - mu) / sd).astype(np.float32)
    for kind in base:
        p, vl, ep = fit_predict(kind, Z[tr], H[tr], Z[va], H[va], Z[te])
        base[kind][te] = p
        fit_log.append({"fold": k, "kind": kind, "val_ce": vl, "epochs": ep})
        print(f"fold {k} {kind}: val CE {vl:.4f} ({ep} epochs)", flush=True)

# ---- group-pair AUCs ----
fams = sorted(set(fam))
fam2i = {f: i for i, f in enumerate(fams)}
fidx = np.array([fam2i[f] for f in fam], np.int32)
rng = np.random.default_rng(SEED)
counts = np.stack([np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)) for _ in range(B)]).astype(np.float32)
main_g = np.stack([H[:, GROUPS[g]].sum(1) for g in GN], 1).argmax(1)


def wauc(s, y, f):
    order = np.argsort(s, kind="mergesort")
    s_s, y_s, f_s = s[order], y[order], f[order]
    starts = np.flatnonzero(np.r_[True, s_s[1:] != s_s[:-1]])

    def compute(w):
        pg = np.add.reduceat(w * y_s, starts, axis=1)
        ng = np.add.reduceat(w * (~y_s), starts, axis=1)
        below = np.cumsum(ng, axis=1) - ng
        return (pg * (below + 0.5 * ng)).sum(1) / (pg.sum(1) * ng.sum(1))

    boots = np.empty(B)
    for i in range(0, B, CHUNK):
        boots[i:i + CHUNK] = compute(counts[i:i + CHUNK][:, f_s].astype(np.float64))
    return compute(np.ones((1, len(s))))[0], boots


def pair_aucs(prob):
    go = np.log(np.stack([prob[:, GROUPS[g]].sum(1) for g in GN], 1) + EPS)
    P, BT = np.full(len(PAIRS), np.nan), np.full((len(PAIRS), B), np.nan)
    for p, (a, b) in enumerate(PAIRS):
        m = (main_g == a) | (main_g == b)
        y = main_g[m] == a
        P[p], BT[p] = wauc(go[m, a] - go[m, b], y, fidx[m])
    return P, BT


res = {"base_mlp": pair_aucs(base["mlp"])}
for model in ("main", "sham"):
    res[f"{model}_clean"] = pair_aucs(net[model]["reference"])
    seeds = [pair_aucs(net[model][f"A3_h1_r{r}"]) for r in (1, 2, 3)]
    res[f"{model}_h1"] = (np.mean([s[0] for s in seeds], 0), np.mean([s[1] for s in seeds], 0))

ci = lambda x: np.percentile(x, [2.5, 97.5], axis=-1)
out = {"pairs": [[GN[a], GN[b]] for a, b in PAIRS], "fit_log": fit_log, "n": int(N), "rows": {}, "diffs": {}}
for name, (P, BT) in res.items():
    lo, hi = ci(BT)
    mlo, mhi = ci(BT.mean(0))
    out["rows"][name] = {"auc": P.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                         "mean": float(P.mean()), "mean_lo": float(mlo), "mean_hi": float(mhi)}
best = "base_mlp"
out["best_baseline"] = best
for name in ("main_h1", "sham_h1", "main_clean"):
    D = res[name][1] - res[best][1]
    lo, hi = ci(D)
    mlo, mhi = ci(D.mean(0))
    out["diffs"][f"{name} - {best}"] = {"delta": (res[name][0] - res[best][0]).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                                       "mean": float((res[name][0] - res[best][0]).mean()), "mean_lo": float(mlo), "mean_hi": float(mhi)}
out["features"] = FEAT
np.save(f"{SP}/hist_pred_{FEAT}_mlp.npy", base["mlp"])
json.dump(out, open(f"{SP}/hist_baseline_{FEAT}.json", "w"), indent=1)

print(f"\n{'pair':42s}" + "".join(f"{r:>12s}" for r in res))
for p, (a, b) in enumerate(PAIRS):
    print(f"{GN[a] + ' vs ' + GN[b]:42s}" + "".join(f"{res[r][0][p]:12.3f}" for r in res))
print(f"{'MEAN':42s}" + "".join(f"{res[r][0].mean():12.3f}" for r in res))
for k, v in out["diffs"].items():
    print(f"{k}: mean {v['mean']:+.3f} [{v['mean_lo']:+.3f}, {v['mean_hi']:+.3f}]")
