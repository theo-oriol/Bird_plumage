import os
import sys
import glob
import pickle
import numpy as np
from multiprocessing import Pool

from paths import PROJECT_ROOT, RUNS_DIR, DATASET_DIR

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = PROJECT_ROOT
sys.path.insert(0, ROOT)
from src.npy_io import read_canvas  # noqa: E402
from skimage import color  # noqa: E402

DATA = os.path.join(DATASET_DIR, "Familly_split_no_ablation")
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
OUT = f"{SP}/hist_chunks"
CHUNK = 2000
Q = [5, 25, 50, 75, 95]


# ── Helpers ──────────────────────────────────────────────────────────────────
def view_of(stem):
    for v in ("Back", "Belly", "Side"):
        if f"_{v}_" in stem:
            return v


_g = (np.arange(64) * 4 + 2) / 255.0
_LUT = color.rgb2lab(np.stack(np.meshgrid(_g, _g, _g, indexing="ij"), -1).reshape(1, -1, 3))[0].astype(np.float32)


def feats(stem):
    c = read_canvas(f"{DATA}/NEW_Segmented-Aves-{view_of(stem)}-NPY", stem)[::2, ::2]
    px = c[c.any(-1)] >> 2
    lab = _LUT[(px[:, 0].astype(np.int32) << 12) | (px[:, 1].astype(np.int32) << 6) | px[:, 2]]
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    C = np.hypot(a, b)
    j, _ = np.histogramdd(lab, bins=(8, 8, 8), range=((0, 100), (-60, 60), (-60, 80)))
    out = [j.ravel() / len(px)]
    for x, r in ((L, (0, 100)), (a, (-60, 60)), (b, (-60, 80)), (C, (0, 90))):
        h, _ = np.histogram(np.clip(x, *r), bins=32, range=r)
        out += [h / len(px), [x.mean(), x.std()], np.percentile(x, Q)]
    return np.concatenate(out).astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    stems = []
    for k in range(3):
        df = pickle.load(open(f"{RUNS_DIR}/main_fold{k}/eval/predictions.pkl", "rb"))
        stems += df.loc[df.condition_name == "reference", "stem"].tolist()
        del df
    np.save(f"{SP}/hist_stems.npy", np.array(stems))
    print(len(stems), "images", flush=True)
    done = {os.path.basename(p) for p in glob.glob(f"{OUT}/*.npy")}
    with Pool(4) as pool:
        for i in range(0, len(stems), CHUNK):
            name = f"{i:06d}.npy"
            if name in done:
                continue
            X = np.stack(pool.map(feats, stems[i:i + CHUNK], chunksize=20))
            np.save(f"{OUT}/{name}", X)
            print(f"{i + len(X)}/{len(stems)}", flush=True)
    print("done", flush=True)
