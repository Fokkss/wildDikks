#!/usr/bin/env bash
set -euo pipefail

TRAIN_PATH="${1:-../data/train_dataset.csv}"
FEATURES_PATH="${2:-}"
OUTPUT_PATH="${3:-submissions_v1_7/submission.csv}"
ARTIFACT_DIR="${4:-artifacts_v1_7}"
MODE="${5:-normal}"

if [[ -z "${FEATURES_PATH}" ]]; then
  python -m new_model_v1_0.train_final_v1_7 \
    --train_path "${TRAIN_PATH}" \
    --artifact_dir "${ARTIFACT_DIR}" \
    --target "Выработка. Результирующий расчет" \
    --header
else
  python -m new_model_v1_0.train_final_v1_7 \
    --train_path "${TRAIN_PATH}" \
    --features_path "${FEATURES_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --artifact_dir "${ARTIFACT_DIR}" \
    --prediction_mode "${MODE}" \
    --target "Выработка. Результирующий расчет" \
    --header
fi
