#!/usr/bin/env bash
set -euo pipefail
TRAIN_PATH="${1:-../data/train_dataset.csv}"
VALID_PATH="${2:-../data/valid_features.csv}"
TARGET="${3:-Выработка. Результирующий расчет}"

python -m new_model_beta13.train_predict \
  --train_path "$TRAIN_PATH" \
  --valid_path "$VALID_PATH" \
  --target "$TARGET" \
  --artifact_dir artifacts_beta13 \
  --submission_dir submissions_beta13 \
  --report_dir reports_beta13

python -m new_model_beta13.catboost_companion \
  --train_path "$TRAIN_PATH" \
  --valid_path "$VALID_PATH" \
  --target "$TARGET" \
  --artifact_dir artifacts_beta13 \
  --submission_dir submissions_beta13_cat \
  --report_dir reports_beta13_cat
