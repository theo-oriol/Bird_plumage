"""Histogram features from the exact 16 crops the network saw at evaluation.

Same call as EvalDataset.__getitem__ for the clean "reference" condition:
chain.build_bag with positions_rng = derive_rng("eval-geometry", stem), so the
flip, crop positions, 128->224 resize and median blur are identical. A3_h1 only
permutes pixels inside each crop, so the pixel set used here (all pixels of the
16 crops, background included, every 2nd pixel per direction) is exactly what
survives pixel shuffling. Same features as hist_features.py.
"""
import os
import sys
import glob
import numpy as np
from multiprocessing import Pool

from paths import PROJECT_ROOT, DATASET_DIR
ROOT = PROJECT_ROOT
sys.path.insert(0, ROOT)
from src import chain  # noqa: E402
from src.ablations.common import Condition  # noqa: E402
from src.rng import derive_rng  # noqa: E402
from hist_features import _LUT, Q, view_of  # noqa: E402

SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
DS = DATASET_DIR
OUT = f"{SP}/hist_crop_chunks"
CHUNK = 2000


def dirs(view):
    leaf = f"NEW_Segmented-Aves-{view}-NPY"
    return [f"{DS}/Familly_split_{s}/{leaf}" for s in ("no_ablation", "A1", "A2", "A7")]


def feats(stem):
    crops, _ = chain.build_bag(*dirs(view_of(stem)), stem, Condition(), derive_rng("eval-condition", stem, "reference"),
                               positions_rng=derive_rng("eval-geometry", stem))
    px = crops[:, ::2, ::2].reshape(-1, 3) >> 2
    lab = _LUT[(px[:, 0].astype(np.int32) << 12) | (px[:, 1].astype(np.int32) << 6) | px[:, 2]]
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    C = np.hypot(a, b)
    j, _ = np.histogramdd(lab, bins=(8, 8, 8), range=((0, 100), (-60, 60), (-60, 80)))
    out = [j.ravel() / len(px)]
    for x, r in ((L, (0, 100)), (a, (-60, 60)), (b, (-60, 80)), (C, (0, 90))):
        h, _ = np.histogram(np.clip(x, *r), bins=32, range=r)
        out += [h / len(px), [x.mean(), x.std()], np.percentile(x, Q)]
    return np.concatenate(out).astype(np.float32)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True).tolist()
    done = {os.path.basename(p) for p in glob.glob(f"{OUT}/*.npy")}
    with Pool(8) as pool:
        for i in range(0, len(stems), CHUNK):
            name = f"{i:06d}.npy"
            if name in done:
                continue
            X = np.stack(pool.map(feats, stems[i:i + CHUNK], chunksize=20))
            np.save(f"{OUT}/{name}", X)
            print(f"{i + len(X)}/{len(stems)}", flush=True)
    print("done", flush=True)
