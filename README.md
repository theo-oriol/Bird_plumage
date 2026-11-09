# Single multi-ablation training

Implementation of the protocol `protocole_ablations_uniques_v3.pdf`: a
DINOv3-S + LoRA model, MIL aggregation by *gated* attention, seven ablations
drawn online on the crops. See the PDF for the full specification; this
document explains how the code maps onto it, what comes from where, and what
is still open.

## Getting started

```bash
pip install -r requirements.txt   # tested with torch 2.5.1 + CUDA

# machine-specific paths (dataset, folds, DINOv3): set them once
cp .env.example .env              # then edit .env; configs/paths.yaml reads its ${VARIABLES}

python scripts/check_dataset_integrity.py      # checks folds/crosswalk (§7)

# A1, A2, A7 are deterministic (no drawn parameter): precomputed once
# offline instead of being recomputed for every bag that draws them, at
# every epoch. Required before any training (chain.py raises a clear
# error if one of these ablations is drawn and its cache does not exist).
python scripts/build_a1_cache.py --workers 8
python scripts/build_a2_cache.py --workers 8
python scripts/build_a7_cache.py --workers 8

python train.py --config configs/train_fold0.yaml
```

The notebook `notebooks/01_ablation_catalogue.ipynb` shows the effect of
each ablation on a real specimen (open it with the `torch` kernel).
`scripts/benchmark_ablations.py` measures what is slow.


## Structure

```
src/
  color.py                 provided for the project — CIELAB / colour naming
  npy_io.py                 reading/writing .rgb.zst files
  sampling.py                crop positions from the .cand_*.zst tables
  rng.py                      deterministic seeds (image, condition, ...)
  chain.py                    THE canonical chain (protocol §2) — orders the 4 insertion points
  ablations/
    a1_gray_lstar.py ... a7_chroma_only.py    one ablation, one file,
      each fully self-contained (numpy/scipy/skimage/zstandard only):
      copying a single one of these files elsewhere is enough to reuse it,
      nothing else from the project is needed
    common.py                  Condition, CIELAB exclusive group
    combinations.py             §4 draw (36 conditions)
  datasets/
    folds.py                    fold CSVs + crosswalk + class weights
    bag_dataset.py                training Dataset (online draw, §4)
    eval_dataset.py                evaluation Dataset (fixed conditions, §7)
  model/
    backbone.py, mil.py, heads.py, bagmodel.py, losses.py
  training/
    run.py                      the training loop + traceability
  evaluation/
    conditions.py, a6_presence.py, metrics.py, run_eval.py

configs/            one YAML file per run; paths.yaml/model.yaml/optim.yaml are included by the others
scripts/             build_a{1,2,7}_cache, build_a6_presence_manifest, benchmark_ablations, check_dataset_integrity, combine_fold_results,
                      build_grayscale_dataset, build_grayscale_norm_lum_dataset
notebooks/            visual catalogue of the ablations
train.py, evaluate.py  entry points
```

## Traceability of a run

`python train.py --config configs/train_fold0.yaml` creates
`runs/main_fold0/` containing:

- `config.yaml` — the resolved config (all `includes` merged), so the run
  can be reproduced from this file alone;
- `manifest.json` — timestamp, git commit (if in a repository),
  python/torch versions, GPU, computed BCE class weights;
- `metrics.csv` — one row per epoch (loss, KL, BCE, lr, duration);
- `checkpoints/{last,best}.pt` — `best` = lowest smoothed training loss
  (the only criterion used for early stopping, §1 — no validation metric
  enters this choice, as the protocol requires).
