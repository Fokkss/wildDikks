#!/usr/bin/env bash
set -euo pipefail

# этот кусок задаёт входные пути; по умолчанию решение лежит в отдельной папке, поэтому используем ../data
TRAIN_PATH=${1:-../data/train_dataset.csv}
FEATURES_PATH=${2:-../data/valid_features.csv}
TARGET=${3:-"Выработка. Результирующий расчет"}

# эта зона создаёт стандартные папки артефактов и предсказаний
mkdir -p artifacts_v1_0 submissions_v1_0 reports_v1_0

# этот кусок обучает V1.0 и формирует один submission-файл
python -m new_model_v1_0.train \
  --train_path "$TRAIN_PATH" \
  --valid_path "$FEATURES_PATH" \
  --target "$TARGET" \
  --artifact_dir artifacts_v1_0 \
  --output_path submissions_v1_0/submission_v1_0.csv
