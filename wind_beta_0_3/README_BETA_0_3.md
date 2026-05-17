# Wind beta 0.3

Self-contained production-oriented pipeline for wind farm power forecasting.

## Main idea

Beta 0.2 showed that a teacher CSV can reproduce the best validation submission, but production-only mode was still weak. Beta 0.3 therefore adds a clean, reproducible production branch based on the recovered strong regressor parameters and the feature engineering file from the strong old solution.

The model no longer needs a validation submission CSV for normal inference. It trains on `train_dataset.csv`, saves `artifacts_beta03/beta03_production_ensemble.pkl`, and predicts any future feature table such as 2027 data.

## Train and predict validation

Run from the patch folder:

```bash
bash scripts_beta03/run_beta03_train_predict.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

Outputs:

```text
artifacts_beta03/
reports_beta03/
submissions_beta03/
```

First files to submit are listed in:

```bash
cat submissions_beta03/SUBMIT_FIRST.txt
```

## Predict 2027

```bash
bash scripts_beta03/run_beta03_predict_2027.sh \
  ../data/features_2027.csv \
  submissions_beta03/submission_2027_beta03_production.csv
```

The output has one column named `prediction`.

## What to submit first

1. `beta03_legacy_exact_1892.csv`
2. `beta03_legacy_exact_1892_rules25.csv`
3. `beta03_legacy_exact_2500.csv`
4. `beta03_ensemble_legacy_conservative.csv`
5. `beta03_ensemble_legacy_conservative_rules25.csv`
6. `beta03_ensemble_legacy_weighted.csv`
7. `beta03_ensemble_legacy_weighted_rules25.csv`
8. `beta03_ensemble_mean_all.csv`

## Why this is cleaner

Beta 0.3 separates two things:

- production model: trained only on train data;
- optional domain correction rules: deterministic physics/wake/stability corrections derived from diagnostics, not from a teacher CSV.

The old teacher approach is still useful for competition calibration, but beta 0.3 is the branch to show as a self-reproducible solution.
