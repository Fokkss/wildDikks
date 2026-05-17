#!/usr/bin/env bash
set -euo pipefail
TRAIN_PATH=${1:-../data/train_dataset.csv}
VALID_PATH=${2:-../data/valid_features.csv}
TARGET=${3:-"Выработка. Результирующий расчет"}
python -m new_model_beta11.catboost_companion \
  --train_path "$TRAIN_PATH" \
  --valid_path "$VALID_PATH" \
  --target "$TARGET" \
  --artifact_dir artifacts_beta11 \
  --submission_dir submissions_beta11_cat \
  --report_dir reports_beta11_cat
