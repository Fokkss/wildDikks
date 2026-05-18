# V1.2 next patch

This patch is for the current `solvrx_final` layout where the main production code is in:

```text
model/new_model_v1_0/
```

The current accepted V1.1 result is strong (`8.3473`), so this patch does not replace the model architecture. It adds two safe tools:

1. `grid_predict_v1_2.py` — no-retrain grid around the current best config.
2. `blend_grid_v1_2.py` — blend grid against the current best submission.
3. `apply_feature_patch_v1_2.py` — optional retrain-only feature patch: EMA inertia + density-corrected wind speed.

## 1. Immediate no-retrain grid

Run from repo root:

```bash
cd model
python -m new_model_v1_0.grid_predict_v1_2 \
  --features_path ../data/valid_features.csv \
  --artifact_dir artifacts_v1_0 \
  --output_dir submissions_v1_2 \
  --report_dir reports_v1_2 \
  --header
```

Submit first:

```text
submissions_v1_2/v1_2_bias0p92_density.csv
submissions_v1_2/v1_2_xgb72_cat40_rules55_bias85_density.csv
submissions_v1_2/v1_2_xgb70_cat43_rules55_bias85_density.csv
submissions_v1_2/v1_2_rules0p58_bias0p85_density.csv
submissions_v1_2/v1_2_profile_physics_strong.csv
```

The script also writes:

```text
submissions_v1_2/SUBMIT_FIRST.txt
submissions_v1_2/SUBMIT_ALL.txt
reports_v1_2/candidate_distance_report.json
```

## 2. Blend against the accepted V1.1 submission

If the current accepted file is available as `submission_v1_1.csv`, run:

```bash
cd model
python -m new_model_v1_0.blend_grid_v1_2 \
  --base ../submission_v1_1.csv \
  --other submissions_v1_2/v1_2_bias0p92_density.csv \
  --output_dir submissions_v1_2_blends \
  --prefix v11_vs_bias092 \
  --weights 0.95,0.90,0.85,0.80,0.75 \
  --header
```

Submit first:

```text
submissions_v1_2_blends/v11_vs_bias092_base090_other010.csv
submissions_v1_2_blends/v11_vs_bias092_base085_other015.csv
submissions_v1_2_blends/v11_vs_bias092_base080_other020.csv
```

## 3. Optional retrain feature patch

Only do this after no-retrain grid/blends. From repo root:

```bash
python model/new_model_v1_0/apply_feature_patch_v1_2.py
```

Then retrain the standard production model:

```bash
cd model
python -m new_model_v1_0.train \
  --train_path ../data/train_dataset.csv \
  --valid_path ../data/valid_features.csv \
  --target "Выработка. Результирующий расчет" \
  --artifact_dir artifacts_v1_2_ema_density \
  --output_path submissions_v1_2_ema_density/submission_v1_2_ema_density.csv \
  --header
```

Send the generated submission, plus the `.summary.json`, back for analysis.

## What to send back to ChatGPT

For every accepted/rejected platform submission, send:

```text
filename
platform score
```

Also send these generated files:

```text
reports_v1_2/candidate_distance_report.json
submissions_v1_2/<best_file>.summary.json
```

If you run the optional retrain patch, send:

```text
submissions_v1_2_ema_density/submission_v1_2_ema_density.summary.json
model/artifacts_v1_2_ema_density/model_v1_0_card.json
```
