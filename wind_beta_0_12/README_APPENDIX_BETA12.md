# Beta 0.12 — CatBoost companion blend и production-пайплайн

## Зачем появилась beta 0.12

К beta 0.10 production-модель XGBoost вышла на плато около `8.396%`: лучший профиль был `anchor_1700_d4_w70 + rules55 + bias0.75 + physics_strong`. Узкий перебор `rules/bias/w` дальше давал только тысячные. В beta 0.11 была добавлена отдельная CatBoost-модель на том же feature set. Сам CatBoost хуже XGB-якоря, но blend `80% XGB anchor + 20% CatBoost` резко улучшил validation score до `8.3607%`. Это означает, что CatBoost даёт комплементарные ошибки, и его стоит оставить в production-ансамбле.

## Текущая архитектура

1. `legacy1700`: XGBoost с параметрами семейства старого сильного регрессора.
2. `depth4_1892`: более глубокий XGBoost-компаньон.
3. `anchor70 = 0.70 * legacy1700 + 0.30 * depth4_1892`.
4. `catboost_companion`: CatBoostRegressor на тех же признаках.
5. Финальный candidate:

```text
base = anchor_weight * anchor70 + (1 - anchor_weight) * catboost_companion
prediction = base + rule_delta(features, base) + profile_delta(features, base) + bias
```

## Что проверяет beta 0.12

Beta 0.11 показала монотонное улучшение при увеличении доли CatBoost:

```text
anchor_weight=0.95 -> 8.3853
anchor_weight=0.90 -> 8.3755
anchor_weight=0.85 -> 8.3672
anchor_weight=0.80 -> 8.3607
```

Поэтому beta 0.12 проверяет более сильную долю CatBoost: `anchor_weight = 0.70 / 0.72 / 0.75 / 0.78 / 0.80`, а также локально подбирает `rules_strength`, `bias` и `profile`.

## Запуск обучения и генерации submission

Из корня папки beta:

```bash
bash scripts_beta12/run_beta12_train_predict.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

Сначала отправлять только:

```bash
cat submissions_beta12_cat/SUBMIT_FIRST.txt
```

Если есть улучшение относительно `8.3607%`, отправить:

```bash
cat submissions_beta12_cat/SUBMIT_SECOND.txt
```

## Запуск для будущего 2027

После обучения все артефакты лежат в `artifacts_beta12/`, включая XGBoost-модели и `catboost_companion.joblib`.

```bash
bash scripts_beta12/run_beta12_predict_2027_catblend.sh \
  ../data/features_2027.csv \
  submissions_beta12/submission_2027_beta12_catblend.csv \
  0.75 \
  0.55 \
  0.75 \
  physics_strong
```

Аргументы после output path:

```text
anchor_weight rules_strength bias profile
```

## Что коммитить

Коммитить код и документацию. Не коммитить:

```text
artifacts_beta*/
submissions_beta*/
reports_beta*/
../data/
*.pkl
*.joblib
```

## Как быстро сделать zip для отправки

Из директории, где лежит `wind_beta_0_12`:

```bash
zip -r wind_beta_0_12_final.zip wind_beta_0_12 \
  -x "*/artifacts_beta12/*" \
  -x "*/submissions_beta12*/*" \
  -x "*/reports_beta12*/*" \
  -x "*.joblib" \
  -x "*.pkl"
```
