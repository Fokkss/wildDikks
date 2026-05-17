#!/usr/bin/env bash
set -euo pipefail
FEATURES_PATH="${1:-../data/features_2027.csv}"
OUTPUT_PATH="${2:-submissions_beta12/submission_2027_beta12_production.csv}"
PROFILE="${3:-anchor_1700_d4_w70}"
RULES="${4:-0.55}"
BIAS="${5:-0.75}"
EXTRA_PROFILE="${6:-physics_strong}"
python -m new_model_beta12.predict_2027 \
  --features_path "$FEATURES_PATH" \
  --artifact_dir artifacts_beta12 \
  --output_path "$OUTPUT_PATH" \
  --profile "$PROFILE" \
  --rules_strength "$RULES" \
  --bias "$BIAS" \
  --extra_profile "$EXTRA_PROFILE"
