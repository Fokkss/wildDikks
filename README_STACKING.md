# Time-Series Stacking Ensemble for Wind Farm Forecasting

This patch adds a safer stacking ensemble for the wind-power task.

## What stacking does

Level 0 base models:

- several diverse XGBoost models;
- optionally CatBoost models.

Level 1 meta-model:

- `Ridge`;
- `LinearRegression`;
- small `XGBoost`;
- optional `LightGBM`.

The meta-model is trained on **out-of-fold predictions**, not on in-sample train predictions. This avoids leakage.

## Files

```text
models/base.py                      # richer wind-energy feature engineering
models/stacking_ensemble.py          # TimeSeriesStackingEnsemble
models/__init__.py                   # exports stacking class, does NOT import torch
new_model/outlier_cleaning.py        # target outlier cleaning utilities
new_model/cut_outliers.py            # CLI to create cleaned train CSV
new_model/tune_stacking_optuna.py    # Optuna tuning for stacking
new_model/train_stacking.py          # final stacking training
new_model/predict_stacking.py        # submission generation
```

## Features added in `ModelPreprocessor`

- `hour_sin`, `hour_cos`
- `month_sin`, `month_cos`
- `dayofyear_sin`, `dayofyear_cos`
- `available_turbines`
- `availability_ratio`
- `available_capacity_mw`
- `ws80_sq`
- `ws80_cube`
- `power_curve_proxy`
- `expected_power_proxy`
- `air_density`
- `wind_power_density`
- wind direction sin/cos
- wind shear features

## Recommended workflow

### 1. Create cleaned train dataset

```bash
python -m new_model.cut_outliers \
  --train_path data/train_dataset.csv \
  --output_path data/train_dataset_clean.csv \
  --mode clip
  
python -m new_model.cut_outliers `
  --train_path data/train_dataset.csv `
  --output_path data/train_dataset_clean.csv `
  --mode clip
```

`clip` is recommended first. It clips physically impossible target values to `[0, available_capacity]` and removes missing targets.

### 2. Smoke-test Optuna

```bash
python -m new_model.tune_stacking_optuna \
  --train_path data/train_dataset_clean.csv \
  --output_dir artifacts/stacking_optuna \
  --n_trials 5 \
  --max_models 3 \
  --outlier_mode none
  
python -m new_model.tune_stacking_optuna `
  --train_path data/train_dataset_clean.csv `
  --output_dir artifacts/stacking_optuna `
  --n_trials 5 `
  --max_models 3 `
  --outlier_mode none
```

### 3. Real Optuna search

```bash
python -m new_model.tune_stacking_optuna \
  --train_path data/train_dataset_clean.csv \
  --output_dir artifacts/stacking_optuna \
  --n_trials 30 \
  --max_models 5 \
  --outlier_mode none
  
python -m new_model.tune_stacking_optuna `
  --train_path data/train_dataset_clean.csv `
  --output_dir artifacts/stacking_optuna `
  --n_trials 30 `
  --max_models 5 `
  --outlier_mode none
```

If LightGBM is installed and you want to allow it as meta-model:

```bash
python -m new_model.tune_stacking_optuna \
  --train_path data/train_dataset_clean.csv \
  --output_dir artifacts/stacking_optuna \
  --n_trials 30 \
  --max_models 5 \
  --outlier_mode none \
  --include_lightgbm_meta
  
python -m new_model.tune_stacking_optuna `
  --train_path data/train_dataset_clean.csv `
  --output_dir artifacts/stacking_optuna `
  --n_trials 30 `
  --max_models 5 `
  --outlier_mode none `
  --include_lightgbm_meta
```

### 4. Train final stacking ensemble

```bash
python -m new_model.train_stacking \
  --train_path data/train_dataset_clean.csv \
  --model_path artifacts/stacking_ensemble.pkl \
  --config_path artifacts/stacking_optuna/best_stacking_config.json \
  --outlier_mode none
  
python -m new_model.train_stacking `
  --train_path data/train_dataset_clean.csv `
  --model_path artifacts/stacking_ensemble.pkl `
  --config_path artifacts/stacking_optuna/best_stacking_config.json `
  --outlier_mode none
```

Without Optuna config:

```bash
python -m new_model.train_stacking \
  --train_path data/train_dataset_clean.csv \
  --model_path artifacts/stacking_ensemble.pkl \
  --n_xgb 4 \
  --n_cat 1 \
  --meta_model ridge \
  --n_splits 5 \
  --outlier_mode none
```

### 5. Predict

No header:

```bash
python -m new_model.predict_stacking \
  --features_path data/valid_features.csv \
  --model_path artifacts/stacking_ensemble.pkl \
  --output_path submission.csv
  
python -m new_model.predict_stacking `
  --features_path data/valid_features.csv `
  --model_path artifacts/stacking_ensemble.pkl `
  --output_path submission.csv
```

With header:

```bash
python -m new_model.predict_stacking \
  --features_path data/valid_features.csv \
  --model_path artifacts/stacking_ensemble.pkl \
  --output_path submission.csv \
  --header
  
python -m new_model.predict_stacking `
  --features_path data/valid_features.csv `
  --model_path artifacts/stacking_ensemble.pkl `
  --output_path submission.csv `
  --header
```

## Notes

Start small. Stacking is expensive:

```text
n_trials * n_base_models * n_splits
```

For example, `30 trials * 5 models * 5 folds = 750 base model fits` plus full fits.

Recommended first run:

```bash
--n_trials 5 --max_models 3
```

Then scale up.
