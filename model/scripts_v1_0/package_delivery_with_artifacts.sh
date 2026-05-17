#!/usr/bin/env bash
set -euo pipefail

# этот кусок собирает финальный ZIP: код + конфиг + веса + готовый submission
OUT=${1:-wind_v1_0_delivery_with_artifacts.zip}
mkdir -p dist_v1_0
rm -f "dist_v1_0/$OUT"

# эта зона проверяет, что веса уже лежат в artifacts_v1_0
if [ ! -d artifacts_v1_0 ]; then
  echo "artifacts_v1_0 not found. Run scripts_v1_0/run_train_predict.sh first or copy trained weights there." >&2
  exit 1
fi

# этот кусок упаковывает решение без исходных данных
zip -r "dist_v1_0/$OUT"   README.md requirements.txt   new_model_v1_0 scripts_v1_0 configs_v1_0 artifacts_v1_0 submissions_v1_0   -x "*/__pycache__/*"   -x "*.DS_Store"   -x "data/*"   -x "../data/*"

echo "Created: dist_v1_0/$OUT"
