#!/usr/bin/env bash
set -euo pipefail

# Creates a delivery ZIP that includes code + trained model artifacts.
# Run from wind_beta_0_13/ after training.
OUT=${1:-wind_beta_0_13_delivery_with_artifacts.zip}

mkdir -p dist_beta13
rm -f "dist_beta13/$OUT"

zip -r "dist_beta13/$OUT" \
  README.md README_BETA_0_13.md README_APPENDIX_BETA13.md WORK_LOG.md FOR_VAAS.md REPO_CLEANUP_CHECKLIST.md \
  new_model_beta13 scripts_beta13 configs_beta13 \
  artifacts_beta13 \
  -x "*/__pycache__/*" \
  -x "*.DS_Store" \
  -x "submissions_beta13*/*" \
  -x "reports_beta13*/*" \
  -x "data/*"

echo "Created: dist_beta13/$OUT"
