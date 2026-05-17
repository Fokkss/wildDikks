# README appendix — beta 0.13 final CatBoost blend search

## What changed from beta 0.12

Beta 0.12 showed that CatBoost is weaker as a standalone model but strongly complementary in a blend with the XGBoost anchor. The best observed region was around:

- `anchor_weight ≈ 0.60`, i.e. `60% XGB anchor + 40% CatBoost`;
- `rules_strength ≈ 0.55`;
- `bias ≈ 0.75`, possibly slightly higher;
- `profile = physics_strong`.

Beta 0.13 performs a narrow final search around that basin instead of generating a wide grid.

## Training and validation prediction

```bash
bash scripts_beta13/run_beta13_train_predict.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

Submit files listed in:

```bash
cat submissions_beta13_cat/SUBMIT_FIRST.txt
```

If first wave improves the current best, submit only selected files from:

```bash
cat submissions_beta13_cat/SUBMIT_SECOND.txt
```

## 2027 prediction

Use the best selected parameters. Current safe default before beta 0.13 scores:

```bash
bash scripts_beta13/run_beta13_predict_2027_catblend.sh \
  ../data/features_2027.csv \
  submissions_beta13/submission_2027_beta13_catblend.csv \
  0.60 \
  0.55 \
  0.75 \
  physics_strong
```

Parameters are:

```text
anchor_weight rules_strength bias profile
```

## Delivery ZIP with model weights

After training, package code and fitted model artifacts:

```bash
bash scripts_beta13/package_beta13_with_artifacts.sh
```

It creates:

```text
dist_beta13/wind_beta_0_13_delivery_with_artifacts.zip
```

The delivery archive intentionally excludes raw data, generated submissions and reports, but includes `artifacts_beta13/` with the trained model weights.
