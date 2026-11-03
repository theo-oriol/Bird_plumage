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
NH, B, SEED, MIN_IMG, MIN_FAM = 10, 500, 12345, 200, 5
EPS = 1e-6


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
out = {"habitat_names": names, "B": B, "min_img": MIN_IMG, "min_fam": MIN_FAM}
for model in ("main", "sham"):
    t0 = time.time()
    S, H, F = load(model)
    main_h = H.argmax(1)
    fams_all, fidx_all = np.unique(F, return_inverse=True)
    rng = np.random.default_rng(SEED)
    # one draw of family counts per bootstrap, shared by all pairs
    C = np.stack([np.bincount(rng.integers(0, len(fams_all), len(fams_all)), minlength=len(fams_all)) for _ in range(B)])
    counts = {int(h): {"n_img": int((main_h == h).sum()), "n_fam": int(len(np.unique(F[main_h == h])))} for h in range(NH)}
    ok = [h for h in range(NH) if counts[h]["n_img"] >= MIN_IMG and counts[h]["n_fam"] >= MIN_FAM]
    L = np.log(S + EPS)
    M = np.full((NH, NH), np.nan)
    LO = np.full((NH, NH), np.nan)
    HI = np.full((NH, NH), np.nan)
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            m = (main_h == a) | (main_h == b)
            y = main_h[m] == a
            s = L[m, a] - L[m, b]
            auc = roc_auc_score(y, s)
            fi = fidx_all[m]
            bs = []
            for c in C:
                w = c[fi]
                if w[y].sum() and w[~y].sum():
                    bs.append(roc_auc_score(y, s, sample_weight=w))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            M[a, b] = M[b, a] = auc
            LO[a, b] = LO[b, a] = lo
            HI[a, b] = HI[b, a] = hi
            print(f"  {model} {names[a][:22]:22s} vs {names[b][:22]:22s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  n={m.sum()}  ({time.time() - t0:.0f}s)", flush=True)
    out[model] = {"counts": counts, "ok": ok, "auc": M.tolist(), "lo": LO.tolist(), "hi": HI.tolist()}
    del S, H, F, L, C
    gc.collect()

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/pairwise_check.json", "w"))
print("wrote pairwise_check.json")
