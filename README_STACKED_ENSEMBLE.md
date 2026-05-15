# Stacked Wind Ensemble

This patch adds a safer N-model ensemble for wind farm power forecasting.

## Why not pseudo-label future weeks?

The proposed idea of predicting one future week, appending it to train, and then using it for the next model is risky because the appended rows do not have real target values. This can accumulate errors.

This implementation uses time-series stacking instead:

1. Train every base model on TimeSeriesSplit folds.
2. Use fold validation predictions as out-of-fold meta-features.
3. Train a Ridge meta-model on those out-of-fold predictions.
4. For valid/test, average fold predictions and pass them to the meta-model.

## Added features

The updated `models/base.py` adds:

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

## Workflow

### 1. Install extra dependency

```bash
pip install optuna
```

### 2. Optional: create cleaned train dataset

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

### 3. Tune stacked ensemble

```bash
python -m new_model.tune_stacked_ensemble_optuna \
  --train_path data/train_dataset_clean.csv \
  --output_dir artifacts/stacked_optuna \
  --n_trials 20 \
  --max_models 5 \
  --n_splits 3 \
  --outlier_mode none
  
python -m new_model.tune_stacked_ensemble_optuna `
  --train_path data/train_dataset_clean.csv `
  --output_dir artifacts/stacked_optuna `
  --n_trials 20 `
  --max_models 5 `
  --n_splits 3 `
  --outlier_mode none
```

Use `--outlier_mode none` here if you already cleaned the dataset in step 2.

### 4. Train final stacked ensemble

```bash
python -m new_model.train_stacked_ensemble \
  --train_path data/train_dataset_clean.csv \
  --model_path artifacts/stacked_ensemble.pkl \
  --config_path artifacts/stacked_optuna/best_stacked_config.json \
  --outlier_mode none
  
python -m new_model.train_stacked_ensemble `
  --train_path data/train_dataset_clean.csv `
  --model_path artifacts/stacked_ensemble.pkl `
  --config_path artifacts/stacked_optuna/best_stacked_config.json `
  --outlier_mode none
```

Or train default stacked ensemble without Optuna:

```bash
python -m new_model.train_stacked_ensemble \
  --train_path data/train_dataset.csv \
  --model_path artifacts/stacked_ensemble.pkl \
  --n_models 5 \
  --n_splits 5 \
  --outlier_mode clip
```

### 5. Predict

No header:

```bash
python -m new_model.predict_stacked_ensemble \
  --features_path data/valid_features.csv \
  --model_path artifacts/stacked_ensemble.pkl \
  --output_path submission.csv
  
python -m new_model.predict_stacked_ensemble `
  --features_path data/valid_features.csv `
  --model_path artifacts/stacked_ensemble.pkl `
  --output_path submission.csv
```

With header:

```bash
python -m new_model.predict_stacked_ensemble \
  --features_path data/valid_features.csv \
  --model_path artifacts/stacked_ensemble.pkl \
  --output_path submission.csv \
  --header
```

## Notes

- Start with `--n_trials 10`, `--max_models 3`, `--n_splits 3` to verify everything runs.
- Then use `--n_trials 30`, `--max_models 5`, `--n_splits 5` if you have time.
- This is heavier than the current CatBoost + XGBoost weighted ensemble.
