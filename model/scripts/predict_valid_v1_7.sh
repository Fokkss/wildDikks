#!/usr/bin/env bash
set -euo pipefail

FEATURES_PATH="${1:-../data/valid_features.csv}"
OUTPUT_PATH="${2:-submissions_v1_7/submission_valid.csv}"
ARTIFACT_DIR="${3:-artifacts_v1_7}"

python -m new_model_v1_0.predict_final_v1_7 \
  --features_path "${FEATURES_PATH}" \
  --artifact_dir "${ARTIFACT_DIR}" \
  --output_path "${OUTPUT_PATH}" \
  --mode normal \
  --output_col prediction \
  --header
