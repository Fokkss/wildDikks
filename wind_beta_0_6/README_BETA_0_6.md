# Wind Beta 0.6 — fast production tuning

Goal: improve the fully reproducible production model without using a teacher submission CSV.

## What changed from beta 0.5

- Removed early-stopping probes from the default run. In beta 0.5 they underpredicted the last train holdout badly, so they are optional only.
- Focused on the best reproducible family: legacy XGBoost params + depth4 blend + deterministic physics rules + small global bias.
- Added a small bias grid around the current best (`rules45 + bias≈0.5`).
- Added a few deterministic profile variants: direction/cloud, pressure/temp/ws physics, low-wind relax.
- Kept outputs with `prediction` header.

## Run

```bash
bash scripts_beta06/run_beta06_train_predict.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

Then submit only files listed in:

```bash
cat submissions_beta06/SUBMIT_FIRST.txt
```

## 2027 prediction

Default profile is conservative but includes the currently confirmed production bias:

```bash
bash scripts_beta06/run_beta06_predict_2027.sh \
  ../data/features_2027.csv \
  submissions_beta06/submission_2027_beta06_production.csv
```

Equivalent explicit call:

```bash
python -m new_model_beta06.predict_production \
  --features_path ../data/features_2027.csv \
  --model_path artifacts_beta06/beta06_production_ensemble.pkl \
  --output_path submissions_beta06/submission_2027_beta06_production.csv \
  --ensemble legacy_conservative_1892_d4_w80 \
  --rules_strength 0.45 \
  --bias 0.55
```
