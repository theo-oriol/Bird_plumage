# Report results: `auc_regression_head_report.pdf`

Scripts that reproduce every result of the report (regression head, main vs. sham models).
They read the evaluated predictions in `runs/` and write
all their outputs (JSON, CSV, NPY/NPZ and figures) to `results/`, which is created automatically.

**One command:** `bash run_all.sh` (runs the steps below in order, one job at a time, with `nice -n 10`).
Scripts that import torch need `KMP_AFFINITY=disabled` (`run_all.sh` sets it).
Steps 15–16 rebuild the exact evaluation crops from the image dataset and are the heaviest (hours).

## Setup: paths

Copy `.env.example` to `.env` and set the paths for your machine (`.env` is not shared, see `.gitignore`).
All scripts read their paths through `paths.py`; `run_all.sh` reads `PYTHON` from `.env`.

| Variable | What it points to | Used by |
|---|---|---|
| `PYTHON` | Python with torch, numpy, pandas, scikit-learn, scipy, pyarrow | `run_all.sh` |
| `DATASETS_DIR` | image dataset, folder containing `Familly_split_<condition>/` | `hist_features*.py` (crop rebuild for 5.3) |
| `PROJECT_ROOT`, `RUNS_DIR` | optional: default to this repository and its `runs/` | almost every script |
| `LABELS_JSON`, `DIMORPHISM_DIR` | optional: default to `configs/labeltoname.json` and `data/dimorphism/` in this repository | sections 1–6; 6.3, 6.5 |

The predictions are read from `runs/{main,sham}_fold{0,1,2}/eval/predictions.pkl` (plus `metrics.csv`,
`manifest.json`); `hist_features_crops.py` also imports the training/evaluation code in `src/`.

## Steps and where each result appears in the report

| # | Script | Writes (in `results/`) | Report part |
|---|---|---|---|
| 1 | `compute_regression_bootstrap.py` | `regression_bootstrap.json` | sections 1–3 tables (AUC, &Delta;, main &minus; sham, intervals) |
| 2 | `compute_regression_benchmark.py` | `summary_slope_heatmap_reg.png`, `reference_noncum_reg_sham.png`, … | sections 1–2 figures |
| 3 | `make_losses_json.py` | `losses.json` | loss values |
| 4 | `plot_loss_curves.py` | `loss_curves.png` | loss figure |
| 5 | `cross_output_check.py` | `cross_output_check.json` | 4.2 (Marine Intertidal), 4.4 |
| 6 | `nonforest_check.py` | `nonforest_check.json` | 4.1 "not forest" baseline |
| 7 | `pairwise_check.py` | `pairwise_check.json` | 4.2 pairwise matrix |
| 8 | `within_bird_check.py` | `within_bird_check.json` | 4.4 (forest use within a bird) |
| 9 | `copresence_check.py` | `copresence_check.json` | 4.3 second habitat (`MODELS=sham` to run one model) |
| 10 | `copres_baseline.py` | `copres_baseline.json` | 4.3 "is detection specific?" check |
| 11 | `pairwise_clustering.py` | `pairwise_clustering.json` | 4.4 clustering into 5 groups |
| 12 | `group_ablation.py` | `group_ablation.json` | 5.2 group pairs under every ablation, 5.1 summary over pairs |
| 13 | `group_ovr_ablation.py main` / `presence` | `group_ovr_ablation_{main,presence}.json` | 5.1 group AUCs |
| 14 | `hist_features.py` | `hist_stems.npy` (+ whole-bird histograms) | image list for 5.3 |
| 15 | `hist_features_crops.py` | `hist_crop_chunks/` | histograms of the exact 16 evaluation crops |
| 16 | `hist_baseline.py crops` / `crops L` / `crops ab` | `hist_baseline_crops*.json`, `hist_pred_crops*_mlp.npy` | 5.3 histogram models (colour + lightness, lightness only, colour only) |
| 17 | `hist_compare.py` | `hist_compare.json` | 5.3 colour vs. lightness |
| 18 | `hist_group_ovr.py` | `hist_group_ovr_main.json` | 5.3 group level |
| 19 | `hist_per_habitat.py` | `hist_per_habitat.json` | 5.3 habitat level |
| 20 | `pattern_only.py` | `pattern_only.json` | 5.3 sources on their own (A2 = pattern only) |
| 21 | `texture_check.py` | `texture_check.json` | 5.3 network vs. histogram within each channel |
| 22 | `sex_view_check.py` | `sex_view_check.json` | 6.1–6.2 subset sizes |
| 23 | `sex_dimorph.py` | `sex_dimorph.json` | 6.3 tertiles |
| 24 | `sex_groups.py` | `sex_groups.json` | 6.1–6.3 group AUCs (clean) |
| 25 | `viewsex_groups.py` | `viewsex_groups.json` | 6.2 view &times; sex (clean) |
| 26 | `ablation_subsets_groups.py` / `viewsex` | `ablation_subsets_groups{,_viewsex}.{json,npz}` | 6.4–6.5 ablation costs per subset (bootstrap draws) |
| 27 | `spatial_subsets.py` | `spatial_subsets.{json,npz}` | 6.4–6.5 spatial-structure upper bound |
| 28 | `sexview_contrasts.py` | `sec6_4_view_sex.csv`, `sec6_5_dimorphism_A.csv`, `sec6_5_dimorphism_B.csv` | 6.4 and 6.5 tables (costs and paired contrasts) |

Shared helpers (imported, not run on their own): `svlib.py` (loading, family bootstrap, group AUCs),
`hist_baseline_fit.py` (training of the histogram model, an MLP).

Statistics everywhere: 500 bootstrap resamples of whole families (seed 12345), 95% percentile intervals;
paired comparisons use the same resamples. In the report, an AUC whose interval includes 0.5 is called
*chance* if the interval is within &plusmn;0.08 and *uncertain* if it is wider.

## Other scripts (not in the report)

- `order_check.py`: writes the family &rarr; order dictionary from the crosswalk (`../data/family_order.json`),
  then checks that the results do not come from recognising the bird's order: order-only baseline,
  AUCs computed within each order (sham, clean images) and ablation &Delta; within order (main). Writes `order_check.json`.
- `order_probe.py`: linear probe predicting the order from the frozen DINOv3-S features of the 16 evaluation crops
  (`scripts/extract_dinov3s_features.py`), trained on the other folds' families; 1, 4 and 16 crops. Writes `order_probe.json`.
- `build_all_predictions.py`: merges every `runs/*/eval/predictions.pkl` into `../runs/all_predictions.parquet`
  (one row per run &times; image &times; condition; columns `hab_*`, `reg_*`, `cls_*` per habitat).
