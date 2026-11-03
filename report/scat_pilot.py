"""Pilot: at which spatial scale is the habitat signal? Wavelet scattering on the evaluation crops.

The histogram model of 5.3 sees no arrangement at all. Here the same model also
gets wavelet scattering coefficients (Kymatio, order 1 + 2, 8 orientations) of
the same 16 crops, averaged over each crop and over the 16 crops, so it sees
texture up to a chosen scale but no arrangement beyond it. Crops are taken
every second pixel (112 px), like the histograms; wavelet scale 2^j at 112 px =
2^(j+1) px at the network's 224 px. Cumulative feature sets:
  hist        : histogram only (no arrangement)          ~ A3 h1
  hist + j<1  : + structure up to ~4 px  (224 scale)      ~ A3 h4
  hist + j<3  : + structure up to ~16 px                  ~ A3 h16
  hist + j<5  : + structure up to ~64 px                  ~ A3 h56
Channels L*, a*, b*. Pilot on 10% of the species (all views), network's own
3 family folds, same MLP and training as hist_baseline.py. Reported: 5 group
AUCs (5.1) and 10 group-pair AUCs, vs. the main network on the same images.
"""
import os
os.environ.setdefault("KMP_AFFINITY", "disabled")
import sys
import gc
import glob
import json
import pickle
import time
import numpy as np
import torch
from multiprocessing import get_context

SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylib"))  # kymatio: pip install --target report/pylib kymatio==0.3.0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svlib import GN, GROUPS, PAIRS, bootstrap_counts, make_wauc, group_ovr, ci  # noqa: E402
import hist_features_crops as hfc  # noqa: E402  (crop rebuild, same as the histograms)
from hist_features import _LUT, Q  # noqa: E402

from paths import PROJECT_ROOT, RUNS_DIR
ROOT = PROJECT_ROOT
J, L, FRAC, SEED, EPS = 5, 8, float(os.environ.get("FRAC", 0.10)), 0, 1e-6
TAG = os.environ.get("TAG", "pilot")
torch.set_num_threads(4)


def lab_crops(stem):
    crops, _ = hfc.chain.build_bag(*hfc.dirs(hfc.view_of(stem)), stem, hfc.Condition(),
                                   hfc.derive_rng("eval-condition", stem, "reference"),
                                   positions_rng=hfc.derive_rng("eval-geometry", stem))
    px = crops[:, ::2, ::2] >> 2
    lab = _LUT[(px[..., 0].astype(np.int32) << 12) | (px[..., 1].astype(np.int32) << 6) | px[..., 2]]
    return lab.transpose(0, 3, 1, 2).astype(np.float32)  # 16 x 3 x 112 x 112


def crops_and_hist(stem):
    """the 16 evaluation crops in L*a*b* (for scattering) and their histogram features (as hist_features_crops.feats)."""
    lab = lab_crops(stem)
    v = lab.transpose(0, 2, 3, 1).reshape(-1, 3)
    L, a, b = v[:, 0], v[:, 1], v[:, 2]
    C = np.hypot(a, b)
    j, _ = np.histogramdd(v, bins=(8, 8, 8), range=((0, 100), (-60, 60), (-60, 80)))
    out = [j.ravel() / len(v)]
    for x, r in ((L, (0, 100)), (a, (-60, 60)), (b, (-60, 80)), (C, (0, 90))):
        h, _ = np.histogram(np.clip(x, *r), bins=32, range=r)
        out += [h / len(v), [x.mean(), x.std()], np.percentile(x, Q)]
    return lab, np.concatenate(out).astype(np.float32)


# coefficient -> largest wavelet scale it uses (order 0: -1), following Kymatio 0.3's output order
maxj = [-1] + [n // L for n in range(J * L)]
for j1 in range(J):
    for l1 in range(L):
        for j2 in range(j1 + 1, J):
            for l2 in range(L):
                maxj.append(j2)
maxj = np.array(maxj)
order = np.r_[0, np.ones(J * L, int), 2 * np.ones(len(maxj) - 1 - J * L, int)]

if __name__ == "__main__":
    t0 = time.time()
    if not os.path.exists(f"{SP}/hist_stems.npy"):  # same image list as hist_features.py
        st = []
        for k in range(3):
            df = pickle.load(open(f"{RUNS_DIR}/main_fold{k}/eval/predictions.pkl", "rb"))
            st += df.loc[df.condition_name == "reference", "stem"].tolist()
            del df
        np.save(f"{SP}/hist_stems.npy", np.array(st))
    stems_all = np.load(f"{SP}/hist_stems.npy", allow_pickle=True)
    pos = {s: i for i, s in enumerate(stems_all)}
    hist_chunks = sorted(glob.glob(f"{SP}/hist_crop_chunks/*.npy"))
    Xh_all = np.concatenate([np.load(p) for p in hist_chunks]) if hist_chunks else None
    N_all = len(stems_all)
    H = np.zeros((N_all, 10), np.float32)
    S_net = np.zeros((N_all, 10))
    fam = np.empty(N_all, object)
    fold = np.full(N_all, -1)
    for k in range(3):
        df = pickle.load(open(f"{RUNS_DIR}/main_fold{k}/eval/predictions.pkl", "rb"))
        r = df[df.condition_name == "reference"]
        ii = np.array([pos[s] for s in r.stem])
        H[ii], S_net[ii] = np.stack(r.habitat.to_numpy()), np.stack(r.reg_prob.to_numpy())
        fam[ii], fold[ii] = r.family.to_numpy(), k
        del df, r
        gc.collect()
    species = np.array(["_".join(s.split("_")[:2]) for s in stems_all])
    rng = np.random.default_rng(SEED)
    usp = np.unique(species)
    pick = set(rng.choice(usp, int(FRAC * len(usp)), replace=False))
    sel = np.flatnonzero([s in pick for s in species])
    stems = stems_all[sel]
    print(f"{len(pick)} species, {len(sel)} images, {len(set(fam[sel]))} families ({time.time() - t0:.0f}s)", flush=True)

    out_f = f"{SP}/scat_{TAG}_feats.npy"
    outh_f = f"{SP}/scat_{TAG}_hist.npy"
    part_f = f"{SP}/scat_{TAG}_feats_partial.npz"
    if os.path.exists(out_f):
        F = np.load(out_f)
        Xh = np.load(outh_f)
    else:
        F = np.zeros((len(stems), 3, len(maxj)), np.float32)
        Xh = np.zeros((len(stems), 668), np.float32)
        start0 = 0
        if os.path.exists(part_f):  # resume after a crash
            z = np.load(part_f)
            start0 = int(z["start"])
            F[:start0] = z["F"][:start0]
            Xh[:start0] = z["Xh"][:start0]
            print(f"resuming at {start0}", flush=True)
        BATCH = 10
        with get_context("fork").Pool(6) as pool:  # workers forked before CUDA is initialised
            scat = __import__("kymatio.torch", fromlist=["Scattering2D"]).Scattering2D(J=J, shape=(112, 112), L=L).cuda()
            it = pool.imap(crops_and_hist, stems[start0:].tolist(), chunksize=4)
            buf, start = [], start0
            for i, (x, hv) in enumerate(it):
                Xh[start0 + i] = hv
                buf.append(x)
                if len(buf) == BATCH or start0 + i == len(stems) - 1:
                    xb = torch.from_numpy(np.stack(buf)).cuda()          # b x 16 x 3 x 112 x 112
                    b = xb.shape[0]
                    with torch.no_grad():
                        y = scat(xb.reshape(-1, 112, 112))                # (b*48) x C x 3 x 3
                    y = y.mean((-1, -2)).reshape(b, 16, 3, -1).mean(1)   # b x 3 x C
                    F[start:start + b] = y.cpu().numpy()
                    start += b
                    buf = []
                    if start % 1000 < BATCH:
                        print(f"  {start}/{len(stems)} ({time.time() - t0:.0f}s)", flush=True)
                    if start % 10000 < BATCH:
                        np.savez(part_f, F=F[:start], Xh=Xh[:start], start=start)
        np.save(out_f, F)
        np.save(outh_f, Xh)
        if TAG == "full" and Xh_all is None:  # also the 5.3 histogram features, in hist_features_crops.py's format
            os.makedirs(f"{SP}/hist_crop_chunks", exist_ok=True)
            np.save(f"{SP}/hist_crop_chunks/000000.npy", Xh)
        if os.path.exists(part_f):
            os.remove(part_f)
    print(f"features done ({time.time() - t0:.0f}s)", flush=True)

    # ---- feature sets ----
    is_bin = np.r_[np.ones(512, bool), np.tile(np.r_[np.ones(32, bool), np.zeros(7, bool)], 4)]
    Xh = Xh.copy()
    Xh[:, is_bin] = np.log(Xh[:, is_bin] + 1e-4)
    Fl = F.copy()
    Fl[:, :, order > 0] = np.log(np.maximum(Fl[:, :, order > 0], 0) + EPS)
    STEPS = [(1, "hist + scat <=4px"), (2, "hist + scat <=8px"), (3, "hist + scat <=16px"), (4, "hist + scat <=32px"), (5, "hist + scat <=64px")]
    SETS = {"hist": lambda: Xh}
    for jm, name in STEPS:
        SETS[name] = (lambda jm=jm: np.concatenate([Xh, Fl[:, :, (maxj < jm) & (order > 0)].reshape(len(sel), -1)], 1))
    SETS["scat <=64px only"] = lambda: Fl[:, :, order > 0].reshape(len(sel), -1)

    from hist_baseline_fit import fit_predict
    Hs, fs, folds = H[sel], fam[sel], fold[sel]
    preds = {}
    for name, make in SETS.items():
        X = make().astype(np.float32)
        P = np.zeros((len(sel), 10))
        for k in range(3):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            r2 = np.random.default_rng(k)
            tf = np.unique(fs[tr])
            va_f = set(r2.choice(tf, int(0.1 * len(tf)), replace=False))
            va = tr[np.array([f in va_f for f in fs[tr]])]
            tr = tr[np.array([f not in va_f for f in fs[tr]])]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Z = ((X - mu) / sd).astype(np.float32)
            P[te], _, _ = fit_predict("mlp", Z[tr], Hs[tr], Z[va], Hs[va], Z[te])
        preds[name] = P
        print(f"  fitted {name} ({X.shape[1]} features, {time.time() - t0:.0f}s)", flush=True)
        del X, Z
        gc.collect()
    np.savez_compressed(f"{SP}/scat_{TAG}_preds.npz", sel=sel, **{k.replace(" ", "_").replace("<=", "le").replace("+", "p"): v for k, v in preds.items()})
    preds["network (clean)"] = S_net[sel]

    # ---- AUCs with family bootstrap ----
    fams = sorted(set(fs))
    fam2i = {f: i for i, f in enumerate(fams)}
    fidx = np.array([fam2i[f] for f in fs], np.int32)
    wauc = make_wauc(bootstrap_counts(fams))
    gs = np.stack([Hs[:, GROUPS[g]].sum(1) for g in GN], 1).argmax(1)

    def pairs(S):
        go = np.log(np.stack([S[:, GROUPS[g]].sum(1) for g in GN], 1) + EPS)
        P, BT = [], []
        for a, b in PAIRS:
            m = (gs == a) | (gs == b)
            y = gs[m] == a
            if y.sum() < 30 or (~y).sum() < 30:
                continue
            p, bt = wauc(go[m, a] - go[m, b], y, fidx[m])
            P.append(p)
            BT.append(bt)
        if not P:  # no pair with enough birds on both sides (tiny samples only)
            return np.array([np.nan]), np.full((1, len(bootstrap_counts(fams))), np.nan)
        return np.array(P), np.array(BT)

    res = {}
    for name, S in preds.items():
        Pg, Bg = group_ovr(Hs, S.astype(np.float64), fidx, wauc)
        Pp, Bp = pairs(S)
        res[name] = {"groups": (np.nanmean(Pg), np.nanmean(Bg, 0)), "pairs": (Pp.mean(), Bp.mean(0)), "per_group": Pg.tolist(), "per_pair": Pp.tolist()}
    base = "hist"
    out = {"n_images": int(len(sel)), "n_species": len(pick), "n_families": len(fams), "rows": {}}
    print(f"\n{'feature set':24s} {'groups':>18s} {'vs hist':>24s} {'pairs':>18s} {'vs hist':>24s}")
    for name in res:
        row = {}
        line = f"{name:24s}"
        for lv in ("groups", "pairs"):
            p, b = res[name][lv]
            d, db = p - res[base][lv][0], b - res[base][lv][1]
            row[lv] = {"auc": float(p), "ci": ci(b).tolist(), "vs_hist": float(d), "vs_hist_ci": ci(db).tolist()}
            line += f" {p:.3f} [{ci(b)[0]:.3f},{ci(b)[1]:.3f}] {d:+.3f} [{ci(db)[0]:+.3f},{ci(db)[1]:+.3f}]"
        row["per_group"], row["per_pair"] = res[name]["per_group"], res[name]["per_pair"]
        out["rows"][name] = row
        print(line)
    json.dump(out, open(f"{SP}/scat_{TAG}.json", "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")
