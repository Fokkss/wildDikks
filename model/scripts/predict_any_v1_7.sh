#!/usr/bin/env bash
set -euo pipefail

FEATURES_PATH="${1:?pass path to csv}"
OUTPUT_PATH="${2:-submissions_v1_7/prediction_any.csv}"
ARTIFACT_DIR="${3:-artifacts_v1_7}"
OUTPUT_COL="${4:-prediction}"

python -m new_model_v1_0.predict_final_v1_7 \
  --features_path "${FEATURES_PATH}" \
  --artifact_dir "${ARTIFACT_DIR}" \
  --output_path "${OUTPUT_PATH}" \
  --mode generic \
  --output_col "${OUTPUT_COL}" \
  --header
