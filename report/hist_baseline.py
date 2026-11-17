import os
os.environ.setdefault("KMP_AFFINITY", "disabled")
import glob
import json
import sys
import numpy as np
import torch
from svlib import SP, GN, PAIRS, NH, load_hist_images, family_auc, pair_aucs, seed_mean, ci

torch.set_num_threads(4)

# ── Settings ─────────────────────────────────────────────────────────────────
# argv[1]: "crops" = histograms of the 16 evaluation crops, "crops_A2" = the same crops under A2
# argv[2]: feature subset, "L" = lightness only, "ab" = colour only, default = all
FEAT = sys.argv[1] if len(sys.argv) > 1 else "crops"
SUBSET = sys.argv[2] if len(sys.argv) > 2 else "all"
CHUNKS = {"crops": "hist_crop_chunks", "crops_A2": "hist_crop_chunks_A2"}[FEAT]
NAME = FEAT if SUBSET == "all" else f"{FEAT}_{SUBSET}"
CONDS = ["reference", "A3_h1_r1", "A3_h1_r2", "A3_h1_r3"]

# ── Data ─────────────────────────────────────────────────────────────────────
X = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{SP}/{CHUNKS}/*.npy"))])
H, F, fold, net = load_hist_images(("main", "sham"), CONDS)
assert len(X) == len(H) and (fold >= 0).all()

# ── Features ─────────────────────────────────────────────────────────────────
# layout: 512 joint L*a*b* bins, then per channel (L, a, b, chroma) 32 bins + mean, std + 5 percentiles
is_bin = np.r_[np.ones(512, bool), np.tile(np.r_[np.ones(32, bool), np.zeros(7, bool)], 4)]
Xl = X.copy()
Xl[:, is_bin] = np.log(X[:, is_bin] + 1e-4)  # histograms are skewed: log-compress the bins only
Xl = Xl.astype(np.float32)
if SUBSET == "L":
    Xl = Xl[:, 512:551]
elif SUBSET == "ab":
    ab = X[:, :512].reshape(-1, 8, 8, 8).sum(1).reshape(-1, 64)  # joint a*b*, summed over L*
    Xl = np.concatenate([np.log(ab + 1e-4), Xl[:, 551:]], 1).astype(np.float32)
print("features", NAME, Xl.shape, flush=True)


# ── Histogram model ──────────────────────────────────────────────────────────
def fit_predict(Xtr, Ytr, Xva, Yva, Xte, seed=0):
    # MLP on the label shares (soft cross-entropy), early stopping on the validation families
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, NH))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr, Xva, Yva = map(torch.from_numpy, (Xtr, Ytr, Xva, Yva))
    ce = lambda x, y: -(y * torch.log_softmax(m(x), 1)).sum(1).mean()
    best, best_state, bad = np.inf, None, 0
    for ep in range(200):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 512):
            j = perm[i:i + 512]
            loss = ce(Xtr[j], Ytr[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = ce(Xva, Yva).item()
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


# ── Fit per fold ─────────────────────────────────────────────────────────────
# trained on the two other folds (10% of their families for early stopping), predicts the held-out fold
pred, fit_log = np.zeros((len(H), NH)), []
for k in range(3):
    tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    va_f = set(np.random.default_rng(k).choice(np.unique(F[tr]), int(0.1 * len(np.unique(F[tr]))), replace=False))
    is_va = np.array([f in va_f for f in F[tr]])
    va, tr = tr[is_va], tr[~is_va]
    mu, sd = Xl[tr].mean(0), Xl[tr].std(0) + 1e-6
    Z = ((Xl - mu) / sd).astype(np.float32)
    pred[te], vl, ep = fit_predict(Z[tr], H[tr], Z[va], H[va], Z[te])
    fit_log.append({"fold": k, "kind": "mlp", "val_ce": vl, "epochs": ep})
    print(f"fold {k}: val CE {vl:.4f} ({ep} epochs)", flush=True)

# ── Group-pair AUCs ──────────────────────────────────────────────────────────
_, fidx, wauc = family_auc(F)
pa = lambda S: pair_aucs(H, S, fidx, wauc)
res = {"base_mlp": pa(pred)}
for model in ("main", "sham"):
    res[f"{model}_clean"] = pa(net[(model, "reference")])
    res[f"{model}_h1"] = seed_mean([pa(net[(model, f"A3_h1_r{r}")]) for r in (1, 2, 3)])


def row(P, BT, key="auc"):
    (lo, hi), (mlo, mhi) = ci(BT), ci(BT.mean(0))
    return {key: P.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "mean": float(P.mean()), "mean_lo": float(mlo), "mean_hi": float(mhi)}


# ── Write ────────────────────────────────────────────────────────────────────
# network minus the histogram model, paired
base = res["base_mlp"]
out = {"pairs": [[GN[a], GN[b]] for a, b in PAIRS], "fit_log": fit_log, "n": len(H), "best_baseline": "base_mlp",
       "features": NAME, "rows": {k: row(*v) for k, v in res.items()},
       "diffs": {f"{k} - base_mlp": row(res[k][0] - base[0], res[k][1] - base[1], "delta") for k in ("main_h1", "sham_h1", "main_clean")}}
np.save(f"{SP}/hist_pred_{NAME}_mlp.npy", pred)
json.dump(out, open(f"{SP}/hist_baseline_{NAME}.json", "w"), indent=1)
for k, v in out["diffs"].items():
    print(f"{k}: mean {v['mean']:+.3f} [{v['mean_lo']:+.3f}, {v['mean_hi']:+.3f}]")
