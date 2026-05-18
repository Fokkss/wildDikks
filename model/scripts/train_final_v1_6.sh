#!/usr/bin/env bash
set -euo pipefail

# переходим в папку model, чтобы python -m видел пакет new_model_v1_0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MODEL_DIR"

TRAIN_PATH="${1:-../data/train_dataset.csv}"
FEATURES_PATH="${2:-../data/features_18_05.csv}"
OUTPUT_PATH="${3:-submissions_v1_6/prediction_18_05.csv}"
ARTIFACT_DIR="${4:-artifacts_v1_6}"

# обучаем модели и сразу выдаем 24 значения для пустой целевой колонки 18.05
python -m new_model_v1_0.train_final_v1_6 \
  --train_path "$TRAIN_PATH" \
  --features_path "$FEATURES_PATH" \
  --target "Выработка. Результирующий расчет" \
  --artifact_dir "$ARTIFACT_DIR" \
  --output_path "$OUTPUT_PATH" \
  --prediction_mode day_target \
  --expected_rows 24 \
  --header
