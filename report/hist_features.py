import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
from skimage import color

from paths import PROJECT_ROOT, DATASET_DIR
from svlib import SP, load

sys.path.insert(0, PROJECT_ROOT)
from src import chain  # noqa: E402
from src.ablations.common import Condition  # noqa: E402
from src.rng import derive_rng  # noqa: E402

# ── Settings ─────────────────────────────────────────────────────────────────
# Colour histograms of the exact 16 evaluation crops of every clean image (input of hist_baseline.py)
# argv[1]: evaluation condition of the crops (default reference = clean images), e.g. A2
COND = sys.argv[1] if len(sys.argv) > 1 else "reference"
OUT = f"{SP}/hist_crop_chunks" + ("" if COND == "reference" else f"_{COND}")
CHUNK, Q = 2000, [5, 25, 50, 75, 95]
# RGB -> CIELAB lookup table on 6-bit channels (fast, and the features are binned anyway)
_g = (np.arange(64) * 4 + 2) / 255.0
LUT = color.rgb2lab(np.stack(np.meshgrid(_g, _g, _g, indexing="ij"), -1).reshape(1, -1, 3))[0].astype(np.float32)


# ── Features ─────────────────────────────────────────────────────────────────
def dirs(view):
    leaf = f"NEW_Segmented-Aves-{view}-NPY"
    return [f"{DATASET_DIR}/Familly_split_{s}/{leaf}" for s in ("no_ablation", "A1", "A2", "A7")]


def feats(stem):
    # 512 joint L*a*b* bins, then per channel (L, a, b, chroma) 32 bins + mean, std + 5 percentiles
    view = stem.split("_")[4]
    cond = Condition() if COND == "reference" else Condition(active=frozenset({COND}))
    crops, _ = chain.build_bag(*dirs(view), stem, cond, derive_rng("eval-condition", stem, COND),
                               positions_rng=derive_rng("eval-geometry", stem))
    px = crops[:, ::2, ::2].reshape(-1, 3) >> 2
    lab = LUT[(px[:, 0].astype(np.int32) << 12) | (px[:, 1].astype(np.int32) << 6) | px[:, 2]]
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    j, _ = np.histogramdd(lab, bins=(8, 8, 8), range=((0, 100), (-60, 60), (-60, 80)))
    out = [j.ravel() / len(px)]
    for x, r in ((L, (0, 100)), (a, (-60, 60)), (b, (-60, 80)), (np.hypot(a, b), (0, 90))):
        h, _ = np.histogram(np.clip(x, *r), bins=32, range=r)
        out += [h / len(px), [x.mean(), x.std()], np.percentile(x, Q)]
    return np.concatenate(out).astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if COND == "reference":  # image list shared by every histogram script
        np.save(f"{SP}/hist_stems.npy", load("main").stem.to_numpy())
    stems = np.load(f"{SP}/hist_stems.npy", allow_pickle=True).tolist()
    os.makedirs(OUT, exist_ok=True)
    done = {os.path.basename(p) for p in glob.glob(f"{OUT}/*.npy")}  # resumable: finished chunks are skipped
    with Pool(8) as pool:
        for i in range(0, len(stems), CHUNK):
            if f"{i:06d}.npy" not in done:
                X = np.stack(pool.map(feats, stems[i:i + CHUNK], chunksize=20))
                np.save(f"{OUT}/{i:06d}.npy", X)
                print(f"{i + len(X)}/{len(stems)}", flush=True)
    print("done", flush=True)
