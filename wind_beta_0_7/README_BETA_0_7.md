# Wind forecast beta 0.7

Production-focused branch. No teacher CSV is required.

## Run

```bash
bash scripts_beta07/run_beta07_train_predict.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

Submit only files listed in:

```bash
cat submissions_beta07/SUBMIT_FIRST.txt
```

The list is generated after writing files, so it cannot contain missing paths.

## 2027 prediction

```bash
bash scripts_beta07/run_beta07_predict_2027.sh \
  ../data/features_2027.csv \
  submissions_beta07/submission_2027_beta07_production.csv \
  anchor_1700_d4_w75 \
  0.45 \
  0.50
```

Arguments: `features_path`, `output_path`, `profile`, `rules_strength`, `bias`.

## What changed

- Returned to the production family that worked best in beta03/beta05.
- Removed early-stopping branch from first-wave candidates because it underpredicted recent holdout.
- Added 1700-tree anchor because it was more stable than 1892/2100 in the last wave.
- Added direction-degrees/1000 handling.
- Uses 84 m hub-height proxy.
- Uses time-aware weather lags without sorting the output rows.
- Uses small production correction rules based on physical regimes found during validation diagnostics.

## Main first-wave candidates

- `anchor_1700_d4_w75`, rules 0.45, bias 0.45/0.55/0.70
- `anchor_1700_d4_w80`, rules 0.45/0.35, bias 0.50
- no-lag and Q1-weighted alternatives

The default production candidate for 2027 is intentionally conservative:

```text
profile = anchor_1700_d4_w75
rules_strength = 0.45
bias = 0.50
```
