# wind beta 0.8 — production line-search

Self-contained production pipeline. No teacher CSV is required. The current anchor is the beta07 production family:

- `anchor_1700_d4_w75`
- `rules_strength ≈ 0.45–0.60`
- `bias ≈ 0.70+`

Beta 0.8 runs a small, submission-budget-friendly search around that anchor.

## Train + predict valid

```bash
bash scripts_beta08/run_beta08_train_predict.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

Submit only the first wave first:

```bash
cat submissions_beta08/SUBMIT_FIRST.txt
```

If one of the first-wave files improves beta07 best, then submit the second wave:

```bash
cat submissions_beta08/SUBMIT_SECOND.txt
```

## Predict 2027

Default production candidate:

```bash
bash scripts_beta08/run_beta08_predict_2027.sh \
  ../data/features_2027.csv \
  submissions_beta08/submission_2027_beta08_production.csv \
  anchor_1700_d4_w75 \
  0.45 \
  0.70 \
  none
```

Arguments after output path: `profile`, `rules_strength`, `bias`, `extra_profile`.
