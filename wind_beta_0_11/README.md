# Wind farm generation forecast — beta 0.11

Self-contained production pipeline for hourly wind-farm generation forecasting.

The current production family is based on a compact XGBoost ensemble with physics-aware feature engineering and deterministic calibration rules. The validation-only teacher/distillation experiments are intentionally excluded from the default production path.

## Quick start

From this folder:

```bash
bash scripts_beta11/run_beta11_train_predict.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

The script creates:

```text
artifacts_beta11/      trained models and metadata
submissions_beta11/    candidate submissions with `prediction` header
reports_beta11/        summary JSONs for every candidate
```

Submit only the files listed in:

```bash
cat submissions_beta11/SUBMIT_FIRST.txt
```

If one of the first-wave files beats the current best, submit:

```bash
cat submissions_beta11/SUBMIT_SECOND.txt
```

## Current production anchor

The best production-only candidate before beta 0.11:

```text
beta10_anchor_1700_d4_w70_rules55_bias0p75_physics_strong.csv — 8.3960
```

Beta 0.11 performs a very narrow coordinate search around this profile:

```text
ensemble       = anchor_1700_d4_w70
rules_strength = 0.50 / 0.55 / 0.60 / 0.65
bias           = 0.72 / 0.75 / 0.78 / 0.82
profile        = physics_strong / physics_xstrong / density_boost / physics_plus_dir_tiny
```

## Predict 2027

After training:

```bash
bash scripts_beta11/run_beta11_predict_2027.sh \
  ../data/features_2027.csv \
  submissions_beta11/submission_2027_beta11_production.csv \
  anchor_1700_d4_w70 \
  0.55 \
  0.75 \
  physics_strong
```

The output format is:

```csv
prediction
...
```

## Optional CatBoost companion

Use only if XGBoost coordinate search stops improving and there is enough time:

```bash
bash scripts_beta11/run_beta11_catboost_companion.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

Then submit only:

```bash
cat submissions_beta11_cat/SUBMIT_CATBOOST.txt
```

This is a companion/blend test, not the default production path.

## Main modules

```text
new_model_beta11/feature_engineering.py  schema-tolerant physics features
new_model_beta11/train_predict.py        train XGBoost ensemble and candidates
new_model_beta11/predict_2027.py         inference from saved artifacts
new_model_beta11/catboost_companion.py   optional CatBoost companion model
```

