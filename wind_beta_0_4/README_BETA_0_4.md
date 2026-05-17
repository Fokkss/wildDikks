# Wind farm forecasting — beta 0.4

Production-oriented branch. No teacher submission CSV is required for prediction.

## What beta 0.4 does

- Trains a focused XGBoost production ensemble around the recovered strong regressor:
  `learning_rate=0.03`, `max_depth=3`, `min_child_weight=5`, `subsample=0.85`,
  `colsample_bytree=0.85`, `reg_alpha=0.04`, `reg_lambda=5.0`.
- Uses 84 m hub-height correction and wind direction scale auto-detection (`degrees / 1000`).
- Keeps physically meaningful features: wind shear, direction sectors, air density,
  weather/cloud/pressure regimes, availability clipping.
- Generates a small focused set of production submissions for tuning weights/rules.
- Does not depend on a validation teacher file.

## Run from a fresh folder

```bash
bash scripts_beta04/run_beta04_train_predict.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

Submit files listed in:

```bash
cat submissions_beta04/SUBMIT_FIRST.txt
```

## Predict 2027

```bash
bash scripts_beta04/run_beta04_predict_2027.sh \
  ../data/features_2027.csv \
  submissions_beta04/submission_2027_beta04_production.csv \
  blend_legacy1892_depth4_w80 \
  0.25 \
  0.0
```

Arguments after output path are: `ensemble`, `rules_strength`, `bias`.
Keep `bias=0.0` for the most conservative production mode unless leaderboard/CV confirms otherwise.
