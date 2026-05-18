#!/usr/bin/env bash
set -euo pipefail

# этот кусок задаёт путь к новым признакам и итоговому файлу прогноза
FEATURES_PATH=${1:-../data/features_2027.csv}
OUTPUT_PATH=${2:-submissions_v1_0/submission_2027_v1_0.csv}

# эта зона запускает инференс по уже обученным весам из artifacts_v1_0
python -m new_model_v1_0.predict \
  --features_path "$FEATURES_PATH" \
  --artifact_dir artifacts_v1_0 \
  --output_path "$OUTPUT_PATH" \
  --header
