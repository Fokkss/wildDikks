#!/usr/bin/env bash
set -euo pipefail

FEATURES_PATH="${1:-../data/features_2027.csv}"
OUTPUT_PATH="${2:-submissions_beta04/submission_2027_beta04_production.csv}"
ENSEMBLE="${3:-blend_legacy1892_depth4_w80}"
RULES_STRENGTH="${4:-0.25}"
BIAS="${5:-0.0}"

python -m new_model_beta04.predict_production \
  --features_path "$FEATURES_PATH" \
  --model_path artifacts_beta04/beta04_production_ensemble.pkl \
  --output_path "$OUTPUT_PATH" \
  --ensemble "$ENSEMBLE" \
  --rules_strength "$RULES_STRENGTH" \
  --bias "$BIAS"
