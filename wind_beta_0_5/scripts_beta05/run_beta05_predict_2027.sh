#!/usr/bin/env bash
set -euo pipefail

FEATURES_PATH="${1:-../data/features_2027.csv}"
OUTPUT_PATH="${2:-submissions_beta05/submission_2027_beta05_production.csv}"

python -m new_model_beta05.predict_production \
  --features_path "$FEATURES_PATH" \
  --model_path artifacts_beta05/beta05_production_ensemble.pkl \
  --output_path "$OUTPUT_PATH" \
  --ensemble b03_legacy_conservative \
  --rules_strength 0.35 \
  --bias 0.0
