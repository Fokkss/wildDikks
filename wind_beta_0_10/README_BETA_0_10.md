# Wind beta 0.10

Production-only iteration. No teacher CSV is required for training or prediction.

Current production anchor before this patch:
`beta09_anchor_1700_d4_w75_rules45_bias0p75_physics_strong.csv ≈ 8.41083`.

This patch does a very narrow search around that anchor:
- stronger physics profile (`physics_xstrong`, `physics_ultra`);
- micro-search around blend weights `w73/w75/w77`;
- micro-search around bias `0.70–0.80`;
- adds SG 3.4-132 / G132-IIA rated-speed features around `10.3 m/s` without replacing the empirically useful legacy `12 m/s` curve.

Run:

```bash
bash scripts_beta10/run_beta10_train_predict.sh ../data/train_dataset.csv ../data/valid_features.csv "Выработка. Результирующий расчет"
cat submissions_beta10/SUBMIT_FIRST.txt
```

Submit only `SUBMIT_FIRST.txt` first. Use `SUBMIT_SECOND.txt` only if first wave beats `8.41083`.

For 2027 production:

```bash
bash scripts_beta10/run_beta10_predict_2027.sh ../data/features_2027.csv submissions_beta10/submission_2027_beta10_production.csv anchor_1700_d4_w75 0.45 0.75 physics_strong
```
