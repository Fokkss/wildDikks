# Wind beta 0.9

Production-only iteration. No teacher CSV is required for training or prediction.

Focus: line-search around the best beta 0.8 production profile:
`anchor_1700_d4_w75 + rules45 + bias0.75 + physics`.

Run:

```bash
bash scripts_beta09/run_beta09_train_predict.sh ../data/train_dataset.csv ../data/valid_features.csv "Выработка. Результирующий расчет"
cat submissions_beta09/SUBMIT_FIRST.txt
```

Submit only `SUBMIT_FIRST.txt` first. Use `SUBMIT_SECOND.txt` only if first wave beats beta 0.8 best.

For 2027:

```bash
bash scripts_beta09/run_beta09_predict_2027.sh ../data/features_2027.csv submissions_beta09/submission_2027_beta09_production.csv anchor_1700_d4_w75 0.45 0.75 physics
```
