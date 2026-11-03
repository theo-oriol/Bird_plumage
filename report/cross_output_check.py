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
NH, B, SEED = 10, 500, 12345


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
out = {"habitat_names": names, "B": B}
for model in ("main", "sham"):
    t0 = time.time()
    S, H, F = load(model)
    fams, fidx = np.unique(F, return_inverse=True)
    rng = np.random.default_rng(SEED)
    W = [np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams))[fidx] for _ in range(B)]
    M = np.zeros((NH, NH))
    res = []
    for t in range(NH):
        y = H[:, t] > 0
        M[t] = [roc_auc_score(y, S[:, h]) for h in range(NH)]
        other = [h for h in range(NH) if h != t]
        best = other[int(np.argmax(M[t, other]))]
        d = np.array([roc_auc_score(y, S[:, t], sample_weight=w) - roc_auc_score(y, S[:, best], sample_weight=w) for w in W])
        lo, hi = np.percentile(d, [2.5, 97.5])
        res.append({"own": M[t, t], "best_other": best, "best_other_auc": M[t, best],
                    "diff": M[t, t] - M[t, best], "lo": float(lo), "hi": float(hi)})
        print(f"  {model} {names[t][:28]:28s} own {M[t, t]:.3f} | best other {names[best][:24]:24s} {M[t, best]:.3f} "
              f"| diff {M[t, t] - M[t, best]:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({time.time() - t0:.0f}s)", flush=True)
    out[model] = {"matrix": M.tolist(), "rows": res}
    del S, H, F, W
    gc.collect()

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/cross_output_check.json", "w"))
print("wrote cross_output_check.json")
