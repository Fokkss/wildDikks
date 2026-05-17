#!/usr/bin/env bash
set -euo pipefail

TRAIN_PATH="${1:-../data/train_dataset.csv}"
VALID_PATH="${2:-../data/valid_features.csv}"
TARGET_COL="${3:-Выработка. Результирующий расчет}"

python -m new_model_beta05.train_production \
  --train_path "$TRAIN_PATH" \
  --valid_path "$VALID_PATH" \
  --target_col "$TARGET_COL" \
  --artifacts_dir artifacts_beta05 \
  --submissions_dir submissions_beta05 \
  --reports_dir reports_beta05

cat submissions_beta05/SUBMIT_FIRST.txt
