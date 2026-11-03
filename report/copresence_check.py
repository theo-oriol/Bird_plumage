"""Co-presence check, clean images: among birds that use habitat A, can the model
tell whether they also use habitat B?

For each ordered pair (A, B), A != B: images with share_A > 0. Truth = share_B > 0;
score = output B. AUC with family bootstrap (B=500) 95% CI. Pairs with fewer than
MIN_SIDE images on either side of the truth or < MIN_FAM families are skipped.
One model in memory at a time.
"""
import os
import gc
import json
import pickle
import time
import numpy as np
from sklearn.metrics import roc_auc_score

from paths import RUNS_DIR, LABELS_JSON
BASE = RUNS_DIR
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
names = json.load(open(LABELS_JSON))
NH, B, SEED, MIN_SIDE, MIN_FAM = 10, 500, 12345, 30, 5


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


out = {"habitat_names": names, "B": B, "min_side": MIN_SIDE, "min_fam": MIN_FAM}
for model in os.environ.get("MODELS", "main,sham").split(","):
    t0 = time.time()
    S, H, F = load(model)
    fams_all, fidx_all = np.unique(F, return_inverse=True)
    rng = np.random.default_rng(SEED)
    C = np.stack([np.bincount(rng.integers(0, len(fams_all), len(fams_all)), minlength=len(fams_all)) for _ in range(B)])
    pairs = {}
    for a in range(NH):
        m = H[:, a] > 0
        fi = fidx_all[m]
        nfam = int(len(np.unique(fi)))
        for b in range(NH):
            if a == b:
                continue
            y = H[m, b] > 0
            rec = {"n": int(m.sum()), "n_fam": nfam, "n_with_b": int(y.sum()), "n_without_b": int((~y).sum()),
                   "n_fam_with_b": int(len(np.unique(fi[y]))), "n_fam_without_b": int(len(np.unique(fi[~y])))}
            if y.sum() < MIN_SIDE or (~y).sum() < MIN_SIDE or rec["n_fam_with_b"] < MIN_FAM or rec["n_fam_without_b"] < MIN_FAM:
                pairs[f"{a},{b}"] = rec
                continue
            s = S[m, b]
            auc = roc_auc_score(y, s)
            bs = []
            for c in C:
                w = c[fi]
                if w[y].sum() and w[~y].sum():
                    bs.append(roc_auc_score(y, s, sample_weight=w))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rec.update(auc=float(auc), lo=float(lo), hi=float(hi))
            pairs[f"{a},{b}"] = rec
            print(f"  {model} uses {names[a][:20]:20s} -> also {names[b][:20]:20s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"with {y.sum()} / without {(~y).sum()}  ({time.time() - t0:.0f}s)", flush=True)
    out[model] = pairs
    del S, H, F, C
    gc.collect()

json.dump(out, open(f"{SP}/copresence_check.json", "w"))
print("wrote copresence_check.json")
