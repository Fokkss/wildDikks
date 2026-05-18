#!/usr/bin/env bash
set -euo pipefail

# собираем zip для платформы, включая веса и train data
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$MODEL_DIR/.." && pwd)"
cd "$MODEL_DIR"

PREDICTION_CSV="${1:-submissions_v1_6/prediction_18_05.csv}"
ARTIFACT_DIR="${2:-artifacts_v1_6}"
TRAIN_DATA="${3:-../data/train_dataset.csv}"
NOTE_DOCX="${4:-../explanatory_note.docx}"
OUTPUT_ZIP="${5:-../final_delivery_v1_6.zip}"

python -m new_model_v1_0.make_delivery_zip_v1_6 \
  --project_model_dir . \
  --prediction_csv "$PREDICTION_CSV" \
  --artifact_dir "$ARTIFACT_DIR" \
  --train_data "$TRAIN_DATA" \
  --note_docx "$NOTE_DOCX" \
  --output_zip "$OUTPUT_ZIP"
