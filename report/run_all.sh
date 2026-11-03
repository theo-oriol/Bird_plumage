#!/usr/bin/env bash
set -euo pipefail

# ── Settings ─────────────────────────────────────────────────────────────────
# One job at a time: the heavy ones load all predictions
cd "$(dirname "$0")"
[ -f ../.env ] || { echo "copy .env.example to .env at the repository root and set the paths"; exit 1; }
set -a; . ../.env; set +a
PY=${PYTHON:-python}
export KMP_AFFINITY=disabled
run() { echo "=== $* ($(date +%H:%M))"; nice -n 10 "$PY" "$@"; }

# ── Sections 1-3, loss table and figures ─────────────────────────────────────
run compute_regression_bootstrap.py
run compute_regression_benchmark.py
run make_losses_json.py
run plot_loss_curves.py

# ── Section 4: specificity ───────────────────────────────────────────────────
run cross_output_check.py
run nonforest_check.py
run pairwise_check.py
run within_bird_check.py
run copresence_check.py
run copres_baseline.py
run pairwise_clustering.py

# ── Section 5: where the signal comes from ───────────────────────────────────
run group_ablation.py
run group_ovr_ablation.py main
run group_ovr_ablation.py presence
run hist_features.py
run hist_features_crops.py
run hist_baseline.py crops
run hist_baseline.py crops L
run hist_baseline.py crops ab
run hist_compare.py
run hist_group_ovr.py
run hist_per_habitat.py
run pattern_only.py
run texture_check.py

# ── Section 6: view, sex, dimorphism ─────────────────────────────────────────
run sex_view_check.py
run sex_dimorph.py
run sex_groups.py
run viewsex_groups.py
run ablation_subsets_groups.py
run ablation_subsets_groups.py viewsex
run spatial_subsets.py
run sexview_contrasts.py

echo "done: results in results/"
