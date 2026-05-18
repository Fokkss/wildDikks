#!/usr/bin/env bash
set -euo pipefail

# predict запускается после обучения, когда artifacts_v1_6 уже создан
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MODEL_DIR"

FEATURES_PATH="${1:-../data/features_18_05.csv}"
OUTPUT_PATH="${2:-submissions_v1_6/prediction_18_05.csv}"
ARTIFACT_DIR="${3:-artifacts_v1_6}"

# прогноз считаем на всей таблице, а на выход пишем только 24 пустые строки 18.05
python -m new_model_v1_0.predict_final_v1_6 \
  --features_path "$FEATURES_PATH" \
  --artifact_dir "$ARTIFACT_DIR" \
  --output_path "$OUTPUT_PATH" \
  --mode day_target \
  --target_col "Выработка. Результирующий расчет" \
  --expected_rows 24 \
  --header
