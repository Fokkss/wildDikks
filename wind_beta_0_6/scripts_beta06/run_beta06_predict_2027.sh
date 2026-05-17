#!/usr/bin/env bash
set -euo pipefail

FEATURES_PATH="${1:-../data/features_2027.csv}"
OUTPUT_PATH="${2:-submissions_beta06/submission_2027_beta06_production.csv}"
ENSEMBLE="${3:-legacy_conservative_1892_d4_w80}"
RULES="${4:-0.45}"
BIAS="${5:-0.55}"

python -m new_model_beta06.predict_production \
  --features_path "$FEATURES_PATH" \
  --model_path artifacts_beta06/beta06_production_ensemble.pkl \
  --output_path "$OUTPUT_PATH" \
  --ensemble "$ENSEMBLE" \
  --rules_strength "$RULES" \
  --bias "$BIAS"
