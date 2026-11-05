import io
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import zstandard as zstd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from paths import PROJECT_ROOT
from svlib import SP, bootstrap_counts, ci
from order_check import family_order

# ── Settings ─────────────────────────────────────────────────────────────────
FOLD_DIR = os.path.join(PROJECT_ROOT, "data", "cross", "cross_version_1_FAM")
FEAT_SUB = "Familly_split_dinov3s_features"
CACHE = os.path.join(SP, "dinov3s_ref_feats.npy")
CACHE_STEMS = os.path.join(SP, "dinov3s_ref_stems.npy")
N_CROPS = (1, 4, 16)
SEED = 12345
KEY_ORDERS = ["PASSERIFORMES", "CHARADRIIFORMES", "PROCELLARIIFORMES", "SULIFORMES", "SPHENISCIFORMES",
              "ANSERIFORMES", "PELECANIFORMES", "GRUIFORMES", "PICIFORMES", "ACCIPITRIFORMES"]


# ── Specimens and features ───────────────────────────────────────────────────
def specimens():
    d = pd.concat([pd.read_csv(f"{FOLD_DIR}/valid_fold_{k}.csv", usecols=["name_of_img", "family"]).assign(fold=k)
                   for k in range(3)], ignore_index=True)
    d["stem"] = d.name_of_img.str.replace(r"\.png$", "", regex=True)
    d["view"] = d.stem.str.split("_").str[4]
    return d.sort_values("stem", ignore_index=True)


def feature_root():
    for key in ("FEATURES_DIR", "DATASETS_DIR"):
        root = os.path.join(os.environ.get(key, ""), FEAT_SUB)
        if os.path.isdir(root):
            return root
    raise SystemExit(f"{FEAT_SUB} not found under FEATURES_DIR or DATASETS_DIR (scripts/extract_dinov3s_features.py)")


# One zstd decompressor per thread (a shared one is not thread-safe)
_local = threading.local()


def read_array_zst(path):
    if not hasattr(_local, "dz"):
        _local.dz = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        return np.load(io.BytesIO(_local.dz.decompress(f.read())))


# Frozen DINOv3-S features of the 16 reference crops, cached as float16 (N, 16, 384)
def load_features(d):
    if os.path.exists(CACHE) and (np.load(CACHE_STEMS, allow_pickle=True) == d.stem.to_numpy()).all():
        return np.load(CACHE, mmap_mode="r")
    root = feature_root()
    paths = [f"{root}/NEW_Segmented-Aves-{v}-NPY/{s}.dinov3s_features.zst" for s, v in zip(d.stem, d.view)]
    with ThreadPoolExecutor(16) as ex:
        X = np.stack(list(ex.map(read_array_zst, paths))).astype(np.float16)
    np.save(CACHE, X)
    np.save(CACHE_STEMS, d.stem.to_numpy())
    return X


# Mean of k crops drawn at random per specimen (same draw for every fold)
def pool(X, k, rng):
    if k == X.shape[1]:
        return X.astype(np.float32).mean(1)
    idx = np.argsort(rng.random((len(X), X.shape[1])), 1)[:, :k]
    return np.take_along_axis(X, idx[:, :, None], 1).astype(np.float32).mean(1)


# ── Probe ────────────────────────────────────────────────────────────────────
# Linear probe trained on the other folds' families, tested on held-out families
def probe(Z, y, fold):
    pred = np.empty(len(y), dtype=object)
    prob = pd.DataFrame(0.0, index=np.arange(len(y)), columns=sorted(set(y)))
    for k in range(3):
        tr, te = fold != k, fold == k
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Z[tr]), y[tr])
        p = clf.predict_proba(sc.transform(Z[te]))
        pred[te] = clf.classes_[p.argmax(1)]
        prob.loc[te, list(clf.classes_)] = p
    return pred, prob


def wacc(correct, fidx, counts, m):
    w = counts[:, fidx[m]]
    return float(correct[m].mean()), ci((w * correct[m]).sum(1) / w.sum(1))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    fo = family_order()
    d = specimens()
    y = d.family.map(fo).to_numpy()
    fold = d.fold.to_numpy()
    X = load_features(d)
    print(f"features {X.shape} ({time.time() - t0:.0f} s)")

    fams = np.unique(d.family)
    fidx = np.searchsorted(fams, d.family)
    counts = bootstrap_counts(fams)
    # An order is learnable only if another fold has one of its families
    seen = np.zeros(len(y), bool)
    for k in range(3):
        seen[fold == k] = np.isin(y[fold == k], y[fold != k])
    major = np.array([pd.Series(y[fold != k]).mode()[0] for k in range(3)])[fold]
    out = {"n_images": len(y), "n_seen_order": int(seen.sum()),
           "unseen_orders": sorted(set(y[~seen])),
           "majority": {"acc": float((major == y).mean()), "acc_seen": float((major == y)[seen].mean())}}

    rng = np.random.default_rng(SEED)
    for k in N_CROPS:
        pred, prob = probe(pool(X, k, rng), y, fold)
        ok = pred == y
        acc, acc_ci = wacc(ok, fidx, counts, np.ones(len(y), bool))
        acc_s, acc_s_ci = wacc(ok, fidx, counts, seen)
        orders_s = sorted(set(y[seen]))
        bal = float(np.mean([ok[seen & (y == o)].mean() for o in orders_s]))
        # One-vs-rest AUC (0.5 = no information), only on test folds whose training had this order
        per = {}
        for o in KEY_ORDERS:
            m = np.isin(fold, [k2 for k2 in range(3) if o in set(y[fold != k2])])
            auc = float(roc_auc_score(y[m] == o, prob[o][m])) if (y[m] == o).any() else float("nan")
            per[o] = {"images": int((y == o).sum()), "recall": float(ok[y == o].mean()), "auc": auc,
                      "seen": bool(seen[y == o].all())}
        out[f"crops_{k}"] = {"acc": acc, "acc_ci": acc_ci.tolist(), "acc_seen": acc_s,
                             "acc_seen_ci": acc_s_ci.tolist(), "balanced_acc_seen": bal, "per_order": per}
        print(f"{k:2d} crops: acc {acc:.3f} [{acc_ci[0]:.3f}-{acc_ci[1]:.3f}]  seen orders {acc_s:.3f}  "
              f"balanced {bal:.3f}  ({time.time() - t0:.0f} s)")
    with open(os.path.join(SP, "order_probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"majority order: acc {out['majority']['acc']:.3f}, seen orders {out['majority']['acc_seen']:.3f}")
    print(f"images whose order is in no training fold: {(~seen).sum()} ({len(out['unseen_orders'])} orders)")
    for o, v in out["crops_16"]["per_order"].items():
        print(f"  {o:20s} {v['images']:6d} images {'' if v['seen'] else '(unseen)':9s}recall 1/4/16 crops: "
              + " / ".join(f"{out[f'crops_{k}']['per_order'][o]['recall']:.2f}" for k in N_CROPS)
              + "   AUC: " + " / ".join(f"{out[f'crops_{k}']['per_order'][o]['auc']:.2f}" for k in N_CROPS))


if __name__ == "__main__":
    main()
