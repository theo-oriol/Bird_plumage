import io
import os
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import zstandard as zstd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from paths import RUNS_DIR, PROJECT_ROOT
from svlib import SP, bootstrap_counts, ci
from order_check import family_order, make_sauc, metrics, METRIC_NAMES

# ── Settings ─────────────────────────────────────────────────────────────────
MODEL = sys.argv[1] if len(sys.argv) > 1 else "sham"
FOLD_DIR = os.path.join(PROJECT_ROOT, "data", "cross", "cross_version_1_FAM")
HAB_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]
FEAT_SUB = "Familly_split_dinov3s_features"
INNER = 3
KEY_ORDERS = ["PASSERIFORMES", "CHARADRIIFORMES", "PROCELLARIIFORMES", "SULIFORMES", "PELECANIFORMES",
              "ANSERIFORMES", "GRUIFORMES", "PICIFORMES", "ACCIPITRIFORMES", "PSITTACIFORMES"]


# ── Data ─────────────────────────────────────────────────────────────────────
_local = threading.local()


def read_array_zst(path):
    if not hasattr(_local, "dz"):
        _local.dz = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        return np.load(io.BytesIO(_local.dz.decompress(f.read())))


def feature_root():
    for key in ("FEATURES_DIR", "DATASETS_DIR"):
        root = os.path.join(os.environ.get(key, ""), FEAT_SUB)
        if os.path.isdir(root):
            return root
    raise SystemExit(f"{FEAT_SUB} not found under FEATURES_DIR or DATASETS_DIR")


# Held-out images of one fold: MIL embedding, frozen DINOv3 mean feature, labels, model output
def load_fold(k, fo, root):
    e = np.load(f"{RUNS_DIR}/{MODEL}_fold{k}/eval/embeddings.npz")
    v = pd.read_csv(f"{FOLD_DIR}/valid_fold_{k}.csv")
    v["stem"] = v.name_of_img.str.replace(r"\.png$", "", regex=True)
    v = v.set_index("stem").loc[e["stem"]]
    views = [s.split("_")[4] for s in e["stem"]]
    paths = [f"{root}/NEW_Segmented-Aves-{w}-NPY/{s}.dinov3s_features.zst" for s, w in zip(e["stem"], views)]
    with ThreadPoolExecutor(16) as ex:
        frozen = np.stack([x.mean(0) for x in ex.map(read_array_zst, paths)]).astype(np.float32)
    return {"mil": e["embedding"], "frozen": frozen, "S": e["reg_prob"].astype(np.float64),
            "H": v[HAB_COLS].to_numpy(np.float32) / 100, "fam": v.family.to_numpy(),
            "order": v.family.map(fo).to_numpy()}


# ── Probe ────────────────────────────────────────────────────────────────────
# Order probe, families held out (inner 3-fold within the fold); returns P(order) and order-mean habitat
def probe(Z, order, fam, H, orders):
    prob = np.zeros((len(order), len(orders)))
    hab = np.zeros_like(H, dtype=np.float64)
    seen = np.zeros(len(order), bool)
    for tr, te in GroupKFold(INNER).split(Z, order, fam):
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(Z[tr]), order[tr])
        p = clf.predict_proba(sc.transform(Z[te]))
        cols = np.searchsorted(orders, clf.classes_)
        prob[np.ix_(te, cols)] = p
        means = pd.DataFrame(H[tr]).groupby(order[tr]).mean().loc[clf.classes_].to_numpy()
        hab[te] = p @ means
        seen[te] = np.isin(order[te], clf.classes_)
    return prob, hab, seen


def order_scores(prob, order, seen, orders, fidx, counts):
    pred = orders[prob.argmax(1)]
    ok = pred == order
    w = counts[:, fidx[seen]]
    acc_ci = ci((w * ok[seen]).sum(1) / w.sum(1))
    so = sorted(set(order[seen]))
    per = {}
    for o in KEY_ORDERS:
        y = order == o
        if (y & seen).sum() >= 30:
            per[o] = {"images": int(y.sum()), "auc": float(roc_auc_score(y[seen], prob[seen, list(orders).index(o)]))}
    return {"acc_seen": float(ok[seen].mean()), "acc_seen_ci": acc_ci.tolist(),
            "balanced_acc_seen": float(np.mean([ok[seen & (order == o)].mean() for o in so])), "per_order": per}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    fo = family_order()
    orders = np.array(sorted(set(fo.values())))
    root = feature_root()
    parts = {k: [] for k in ("H", "S", "fam", "order", "seen", "major")}
    parts.update({f"{f}_{x}": [] for f in ("mil", "frozen") for x in ("prob", "hab")})
    for k in range(3):
        d = load_fold(k, fo, root)
        print(f"fold {k}: {len(d['order'])} images ({time.time() - t0:.0f} s)")
        for feat in ("mil", "frozen"):
            prob, hab, seen = probe(d[feat], d["order"], d["fam"], d["H"], orders)
            parts[f"{feat}_prob"].append(prob)
            parts[f"{feat}_hab"].append(hab)
        maj = np.zeros(len(d["order"]), object)
        for tr, te in GroupKFold(INNER).split(d["mil"], d["order"], d["fam"]):
            maj[te] = pd.Series(d["order"][tr]).mode()[0]
        for key in ("H", "S", "fam", "order"):
            parts[key].append(d[key])
        parts["seen"].append(seen)
        parts["major"].append(maj)
    A = {k: np.concatenate(v) for k, v in parts.items()}

    fams = np.unique(A["fam"])
    fidx = np.searchsorted(fams, A["fam"]).astype(np.int32)
    counts = bootstrap_counts(fams)
    sauc = make_sauc(counts)
    one = np.zeros(len(fidx), np.int32)
    oidx = np.searchsorted(orders, A["order"]).astype(np.int32)
    seen = A["seen"]

    out = {"model": MODEL, "metrics": METRIC_NAMES, "n_images": len(fidx), "n_seen": int(seen.sum()),
           "majority_acc_seen": float((A["major"] == A["order"])[seen].mean())}
    for feat in ("mil", "frozen"):
        out[f"order_{feat}"] = order_scores(A[f"{feat}_prob"], A["order"], seen, orders, fidx, counts)
    # Habitat predicted from the order the probe sees, scored like the PDF (pooled and within order)
    out["habitat"] = {}
    for name, S in (("model", A["S"]), ("order_from_mil", A["mil_hab"]), ("order_from_frozen", A["frozen_hab"])):
        for mode, strata in (("pooled", one), ("within", oidx)):
            p, bt = metrics(A["H"], S, fidx, strata, sauc)
            lo, hi = ci(bt)
            out["habitat"][f"{name}_{mode}"] = {"auc": p.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
        print(f"  {name} ({time.time() - t0:.0f} s)")
    with open(os.path.join(SP, f"order_probe_mil_{MODEL}.json"), "w") as f:
        json.dump(out, f, indent=1)
    report(out)


# ── Printed summary ──────────────────────────────────────────────────────────
def report(out):
    print(f"\norder probe, {out['n_seen']} images whose order is in the probe's training "
          f"(majority-order accuracy {out['majority_acc_seen']:.3f})")
    for feat in ("frozen", "mil"):
        r = out[f"order_{feat}"]
        print(f"  {feat:7s} acc {r['acc_seen']:.3f} [{r['acc_seen_ci'][0]:.3f}-{r['acc_seen_ci'][1]:.3f}]  "
              f"balanced {r['balanced_acc_seen']:.3f}")
    print(f"  one-vs-rest AUC per order{'':6s}frozen     mil")
    for o in KEY_ORDERS:
        if o in out["order_mil"]["per_order"]:
            print(f"  {o:28s}{out['order_frozen']['per_order'][o]['auc']:8.3f}{out['order_mil']['per_order'][o]['auc']:8.3f}")
    h = out["habitat"]
    print(f"\nhabitat AUC{'':27s}{'model':>8s}{'order<-mil':>12s}{'order<-frozen':>15s}{'model within':>14s}")
    for i, n in enumerate(out["metrics"]):
        print(f"{n[:37]:38s}{h['model_pooled']['auc'][i]:8.3f}{h['order_from_mil_pooled']['auc'][i]:12.3f}"
              f"{h['order_from_frozen_pooled']['auc'][i]:15.3f}{h['model_within']['auc'][i]:14.3f}")


if __name__ == "__main__":
    main()
