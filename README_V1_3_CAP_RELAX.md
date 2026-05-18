# V1.3 cap-relax continuation

This patch does not retrain models. It generates candidates around the current best V1.1/V1.2 pipeline.

Why: the current summary shows max prediction around 79.695 MW. This is likely caused by row-wise repair/availability cap. Since the leaderboard improved after feature engineering but did not improve after small config-grid changes, the next reasonable no-retrain experiment is to relax only the upper tail while keeping the global physical cap 90.09 MW.

## Install

Unzip from repository root:

```bash
unzip v1_3_cap_relax_patch.zip -d .
```

## Generate candidates

Run from `model/`:

```bash
python -m new_model_v1_0.predict_relax_cap_v1_3 \
  --features_path ../data/valid_features.csv \
  --artifact_dir artifacts_v1_0 \
  --output_dir submissions_v1_3 \
  --report_dir reports_v1_3 \
  --header
```

## Submit first

Open:

```text
model/submissions_v1_3/SUBMIT_FIRST.txt
```

Recommended first submissions:

```text
submissions_v1_3/v1_3_caprelax_uncap10.csv
submissions_v1_3/v1_3_caprelax_uncap15.csv
submissions_v1_3/v1_3_capzone_lift1p0.csv
submissions_v1_3/v1_3_tail_lift0p6.csv
submissions_v1_3/v1_3_caprelax_uncap20.csv
```

## If one candidate improves

Send the score table and this file back:

```text
model/reports_v1_3/v1_3_cap_relax_report.json
```

## If no candidate improves

Blend the current best `submission_v1_1.csv` with the least bad candidate:

```bash
python -m new_model_v1_0.blend_grid_v1_3 \
  --base ../submission_v1_1.csv \
  --other submissions_v1_3/v1_3_caprelax_uncap10.csv \
  --output_dir submissions_v1_3_blends \
  --prefix v11_caprelax10 \
  --weights 0.97,0.95,0.93,0.90,0.85 \
  --header
```

Submit first:

```text
submissions_v1_3_blends/v11_caprelax10_base095_other005.csv
submissions_v1_3_blends/v11_caprelax10_base093_other007.csv
submissions_v1_3_blends/v11_caprelax10_base090_other010.csv
```
