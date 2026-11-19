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
run crop_sampling.py  # needs the GPU predictions of variance_crop*.py (see README)
run attention_position.py

# ── Section 4: specificity ───────────────────────────────────────────────────
run cross_output_check.py
run nonforest_check.py
run pairwise_check.py
run habitat_gradient.py
run copresence_check.py
run copres_baseline.py
run copres_baseline.py main
run within_bird_check.py

# ── Section 5: where the signal comes from ───────────────────────────────────
run group_ablation.py
run group_ovr_ablation.py main
run group_ovr_ablation.py presence
run hist_features.py
run hist_baseline.py crops
run hist_baseline.py crops L
run hist_baseline.py crops ab
run hist_compare.py
run hist_group_ovr.py
run hist_per_habitat.py
run pattern_only.py
run hist_features.py A2
run hist_baseline.py crops_A2
run hist_baseline.py crops_A2 L
run hist_a2_check.py
run texture_check.py

# ── Section 6: view, sex, dimorphism ─────────────────────────────────────────
run sex_view_check.py
run sex_dimorph.py
run sex_groups.py
run viewsex_groups.py
run ablation_subsets_groups.py
run ablation_subsets_groups.py viewsex
run spatial_subsets.py

echo "done: results in results/"
