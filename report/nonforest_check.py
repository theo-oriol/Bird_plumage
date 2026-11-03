"""How much of each habitat's AUC is just 'not a forest bird'?

Clean (reference) images, regression outputs. For each habitat t != Forest:
  full        AUC of "t present" ranked by output t, all images (= report value)
  notforest   same labels, ranked by -Forest output (the 'least forest-like' baseline)
  nf_own      AUC of "t present" ranked by output t, among images with Forest share == 0
              (+ family bootstrap 95% CI)
  nf_best     best other output in that same subset, and own - best other (+ CI)
  nf05_own    same as nf_own but subset = Forest share < 0.5 (forest not the main habitat)
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
NH, FOREST, B, SEED = 10, 0, 500, 12345


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


def boot_weights(fam, n_boot):
    fams, fidx = np.unique(fam, return_inverse=True)
    rng = np.random.default_rng(SEED)
    return [np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams))[fidx] for _ in range(n_boot)], len(fams)


def wauc(y, s, w):
    if not (w[y].sum() and w[~y].sum()):
        return np.nan
    return roc_auc_score(y, s, sample_weight=w)


out = {"habitat_names": names, "B": B}
for model in ("main", "sham"):
    t0 = time.time()
    S, H, F = load(model)
    nf = H[:, FOREST] == 0
    nf05 = H[:, FOREST] < 0.5
    Wnf, nfam_nf = boot_weights(F[nf], B)
    rows = {}
    for t in range(NH):
        if t == FOREST:
            continue
        y = H[:, t] > 0
        full = roc_auc_score(y, S[:, t])
        notforest = roc_auc_score(y, -S[:, FOREST])
        yn, Sn = y[nf], S[nf]
        own = roc_auc_score(yn, Sn[:, t])
        others = {h: roc_auc_score(yn, Sn[:, h]) for h in range(NH) if h not in (t, FOREST)}
        best = max(others, key=others.get)
        own_b = np.array([wauc(yn, Sn[:, t], w) for w in Wnf])
        diff_b = own_b - np.array([wauc(yn, Sn[:, best], w) for w in Wnf])
        lo, hi = np.nanpercentile(own_b, [2.5, 97.5])
        dlo, dhi = np.nanpercentile(diff_b, [2.5, 97.5])
        own05 = roc_auc_score(y[nf05], S[nf05][:, t])
        rows[names[t]] = dict(
            prev=float(y.mean()), n_pos=int(y.sum()), n_pos_nf=int(yn.sum()), frac_pos_kept=float(yn.sum() / y.sum()),
            full=full, notforest=notforest, nf_own=own, nf_lo=float(lo), nf_hi=float(hi),
            nf_best=names[best], nf_best_auc=others[best], nf_diff=own - others[best], nf_dlo=float(dlo), nf_dhi=float(dhi),
            nf05_own=own05, n_pos_nf05=int(y[nf05].sum()))
        r = rows[names[t]]
        print(f"  {model} {names[t][:26]:26s} full {full:.3f} | not-forest base {notforest:.3f} | "
              f"non-forest own {own:.3f} [{lo:.3f},{hi:.3f}] (pos kept {r['n_pos_nf']}/{r['n_pos']}) | "
              f"best other {names[best][:14]:14s} {others[best]:.3f} diff {own - others[best]:+.3f} [{dlo:+.3f},{dhi:+.3f}] | "
              f"forest<0.5 {own05:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    out[model] = {"n_images": int(len(S)), "n_nonforest": int(nf.sum()), "n_nonforest_families": nfam_nf,
                  "n_forest_lt05": int(nf05.sum()), "rows": rows}
    print(f"  {model}: {nf.sum()} non-forest images ({nfam_nf} families), {nf05.sum()} with forest share < 0.5", flush=True)
    del S, H, F, Wnf
    gc.collect()

json.dump(out, open(f"{SP}/nonforest_check.json", "w"))
print("wrote nonforest_check.json")
