#!/usr/bin/env bash
set -euo pipefail
FEATURES_PATH=${1:-../data/features_2027.csv}
OUTPUT_PATH=${2:-submissions_beta13/submission_2027_beta13_catblend.csv}
ANCHOR_WEIGHT=${3:-0.60}
RULES_STRENGTH=${4:-0.55}
BIAS=${5:-0.75}
PROFILE=${6:-physics_strong}

python -m new_model_beta13.predict_2027_catblend \
  --features_path "$FEATURES_PATH" \
  --artifact_dir artifacts_beta13 \
  --output_path "$OUTPUT_PATH" \
  --anchor_weight "$ANCHOR_WEIGHT" \
  --rules_strength "$RULES_STRENGTH" \
  --bias "$BIAS" \
  --extra_profile "$PROFILE"
