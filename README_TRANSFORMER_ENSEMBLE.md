# Transformer Ensemble for Wind Farm Forecasting

This is an experimental alternative to the CatBoost/XGBoost ensemble.

## What is `transform()` vs Transformer?

In the current project, `ModelPreprocessor.transform(df)` means: apply the already-fitted preprocessing pipeline to new data. It is not a model architecture.

A **Transformer** is a neural-network architecture based on self-attention. For this task it receives a sliding window of recent hourly feature rows, for example the last 24 hours, and predicts the current hour generation.

## Files

```text
models/transformer_model.py
new_model/outlier_cleaning.py
new_model/cut_outliers.py
new_model/tune_transformer_optuna.py
new_model/train_transformer_ensemble.py
new_model/predict_transformer_ensemble.py
requirements_transformer.txt
```

## Install

```bash
pip install -r requirements.txt
pip install -r requirements_transformer.txt
```

## Optional: clean outliers

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

## Tune with Optuna

Smoke test:

```bash
python -m new_model.tune_transformer_optuna \
  --train_path data/train_dataset_clean.csv \
  --output_dir artifacts/transformer_optuna \
  --n_trials 5 \
  --outlier_mode none
  
python -m new_model.tune_transformer_optuna `
  --train_path data/train_dataset_clean.csv `
  --output_dir artifacts/transformer_optuna `
  --n_trials 5 `
  --outlier_mode none
```

More serious run:

```bash
python -m new_model.tune_transformer_optuna \
  --train_path data/train_dataset_clean.csv \
  --output_dir artifacts/transformer_optuna \
  --n_trials 30 \
  --outlier_mode none
```

This saves:

```text
artifacts/transformer_optuna/best_transformer_config.json
artifacts/transformer_optuna/trials.csv
```

## Train final Transformer ensemble

With Optuna config:

```bash
python -m new_model.train_transformer_ensemble \
  --train_path data/train_dataset_clean.csv \
  --model_path artifacts/transformer_ensemble.pkl \
  --config_path artifacts/transformer_optuna/best_transformer_config.json \
  --outlier_mode none
```

Without Optuna:

```bash
python -m new_model.train_transformer_ensemble \
  --train_path data/train_dataset.csv \
  --model_path artifacts/transformer_ensemble.pkl \
  --n_models 3 \
  --outlier_mode clip
  
python -m new_model.train_transformer_ensemble `
  --train_path data/train_dataset.csv `
  --model_path artifacts/transformer_ensemble.pkl `
  --n_models 3 `
  --outlier_mode clip
```

## Predict

No header:

```bash
python -m new_model.predict_transformer_ensemble \
  --features_path data/valid_features.csv \
  --model_path artifacts/transformer_ensemble.pkl \
  --output_path submission.csv
  
python -m new_model.predict_transformer_ensemble `
  --features_path data/valid_features.csv `
  --model_path artifacts/transformer_ensemble.pkl `
  --output_path submission.csv
```

With header:

```bash
python -m new_model.predict_transformer_ensemble \
  --features_path data/valid_features.csv \
  --model_path artifacts/transformer_ensemble.pkl \
  --output_path submission.csv \
  --header
  
python -m new_model.predict_transformer_ensemble `
  --features_path data/valid_features.csv `
  --model_path artifacts/transformer_ensemble.pkl `
  --output_path submission.csv `
  --header
```

## Will it improve the score?

Maybe, but it is not guaranteed. XGBoost/CatBoost are usually stronger baselines on small-to-medium tabular datasets. Transformer can help if sequential patterns matter and the validation set is large enough. Treat this as an experimental second ensemble and compare local validation before trusting it.
