# Wind beta 0.5

Production-oriented pipeline. No teacher CSV is required for normal train/predict.

## What changed vs beta 0.4

- Reverted the main production family to beta 0.3, because it scored better than beta 0.4.
- Added focused rule/bias grid around the best production family.
- Added early-stopping route: `n_estimators=10000`, `early_stopping_rounds=300`; best iteration is estimated on the last chronological 20% of train, then the model is refit on all train with best_iter × factor.
- Keeps all paths clean: `artifacts_beta05`, `submissions_beta05`, `reports_beta05`.

## Run

```bash
bash scripts_beta05/run_beta05_train_predict.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

Submit only files listed in:

```bash
cat submissions_beta05/SUBMIT_FIRST.txt
```

## 2027 prediction

Default production-safe profile:

```bash
bash scripts_beta05/run_beta05_predict_2027.sh \
  ../data/features_2027.csv \
  submissions_beta05/submission_2027_beta05_production.csv
```

Manual profile example:

```bash
python -m new_model_beta05.predict_production \
  --features_path ../data/features_2027.csv \
  --model_path artifacts_beta05/beta05_production_ensemble.pkl \
  --output_path submissions_beta05/submission_2027_beta05_production.csv \
  --ensemble b03_legacy_conservative \
  --rules_strength 0.35 \
  --bias 0.0
```

## First submit list

1. `beta05_b03_legacy_conservative_rules35.csv`
2. `beta05_b03_legacy_conservative_rules45.csv`
3. `beta05_b03_legacy_conservative_rules60.csv`
4. `beta05_b03_legacy_conservative_rules35_bias0p5.csv`
5. `beta05_b03_legacy_conservative_rules35_bias0p75.csv`
6. `beta05_b03_legacy_conservative_rules45_bias0p5.csv`
7. `beta05_b03_legacy_weighted_rules25_bias0p5.csv`
8. `beta05_b03_legacy_weighted_rules35_bias0p5.csv`
9. `beta05_es_legacy_main_rules25.csv`
10. `beta05_es_mix_legacy_b03_rules25.csv`
11. `beta05_es_conservative_rules25.csv`
12. `beta05_es_bestmix_rules25.csv`
13. `beta05_es_bestmix_rules35.csv`
