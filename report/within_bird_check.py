import os
import gc
import json
import pickle
import time
import numpy as np
from sklearn.metrics import roc_auc_score

from paths import RUNS_DIR, LABELS_JSON

# ── Settings ─────────────────────────────────────────────────────────────────
BASE = RUNS_DIR
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
names = json.load(open(LABELS_JSON))
NH, B, SEED, MIN_IMG, MIN_FAM, MIN_SIDE, EPS = 10, 500, 12345, 200, 5, 30, 1e-6


# ── Helpers ──────────────────────────────────────────────────────────────────
def load(model):
    S, H, F = [], [], []
    for k in range(3):
        df = pickle.load(open(f"{BASE}/{model}_fold{k}/eval/predictions.pkl", "rb"))
        r = df[df.condition_name == "reference"]
        S.append(np.stack(r.reg_prob.to_numpy()).astype(np.float64))
        H.append(np.stack(r.habitat.to_numpy()).astype(np.float32))
        F += list(r.family)
        del df, r
        gc.collect()
    return np.concatenate(S), np.concatenate(H), np.array(F)


# ── Per model ────────────────────────────────────────────────────────────────
out = {"habitat_names": names, "B": B, "min_img": MIN_IMG, "min_fam": MIN_FAM, "min_side": MIN_SIDE}
for model in ("main", "sham"):
    t0 = time.time()
    S, H, F = load(model)
    fams_all, fidx_all = np.unique(F, return_inverse=True)
    rng = np.random.default_rng(SEED)
    C = np.stack([np.bincount(rng.integers(0, len(fams_all), len(fams_all)), minlength=len(fams_all)) for _ in range(B)])
    L = np.log(S + EPS)
    pairs = {}
    for a in range(NH):
        for b in range(a + 1, NH):
            m = (H[:, a] > 0) & (H[:, b] > 0) & (H[:, a] != H[:, b])
            y = H[m, a] > H[m, b]
            n, nfam = int(m.sum()), int(len(np.unique(F[m])))
            rec = {"n": n, "n_fam": nfam, "n_a_larger": int(y.sum()), "n_b_larger": int((~y).sum())}
            if n < MIN_IMG or nfam < MIN_FAM or y.sum() < MIN_SIDE or (~y).sum() < MIN_SIDE:
                pairs[f"{a},{b}"] = rec
                continue
            s = L[m, a] - L[m, b]
            auc = roc_auc_score(y, s)
            fi = fidx_all[m]
            bs = []
            for c in C:
                w = c[fi]
                if w[y].sum() and w[~y].sum():
                    bs.append(roc_auc_score(y, s, sample_weight=w))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rec.update(auc=float(auc), lo=float(lo), hi=float(hi))
            pairs[f"{a},{b}"] = rec
            print(f"  {model} {names[a][:20]:20s} vs {names[b][:20]:20s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"n={n} fam={nfam} ({y.sum()} A>B / {(~y).sum()} B>A)  ({time.time() - t0:.0f}s)", flush=True)
    out[model] = pairs
    del S, H, F, L, C
    gc.collect()

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/within_bird_check.json", "w"))
print("wrote within_bird_check.json")
