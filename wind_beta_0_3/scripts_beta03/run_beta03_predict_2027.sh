#!/usr/bin/env bash
set -euo pipefail

FEATURES_PATH="${1:-../data/features_2027.csv}"
OUTPUT_PATH="${2:-submissions_beta03/submission_2027_beta03_production.csv}"

python -m new_model_beta03.predict_production \
  --features_path "$FEATURES_PATH" \
  --model_path artifacts_beta03/beta03_production_ensemble.pkl \
  --output_path "$OUTPUT_PATH" \
  --ensemble ensemble_legacy_weighted \
  --rules_strength 0.25
