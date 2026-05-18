#!/usr/bin/env bash
set -euo pipefail

FEATURES_PATH="${1:-../data/features_17_18.csv}"
OUTPUT_PATH="${2:-submissions_v1_7/prediction_18_05.csv}"
ARTIFACT_DIR="${3:-artifacts_v1_7}"
FILLED_TABLE_PATH="${4:-}"

if [[ -z "${FILLED_TABLE_PATH}" ]]; then
  python -m new_model_v1_0.predict_final_v1_7 \
    --features_path "${FEATURES_PATH}" \
    --artifact_dir "${ARTIFACT_DIR}" \
    --output_path "${OUTPUT_PATH}" \
    --mode day18 \
    --target_col "Выработка. Результирующий расчет" \
    --header
else
  python -m new_model_v1_0.predict_final_v1_7 \
    --features_path "${FEATURES_PATH}" \
    --artifact_dir "${ARTIFACT_DIR}" \
    --output_path "${OUTPUT_PATH}" \
    --mode day18 \
    --target_col "Выработка. Результирующий расчет" \
    --filled_table_path "${FILLED_TABLE_PATH}" \
    --header
fi
