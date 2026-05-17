# Wind V1.0 — запуск решения

## 1. Установка

```bash
pip install -r requirements.txt
```

## 2. Обучить модель и сразу получить прогноз

Запускать из папки `wind_v1_0`:

```bash
bash scripts_v1_0/run_train_predict.sh   ../data/train_dataset.csv   ../data/valid_features.csv   "Выработка. Результирующий расчет"
```

Результат:

```text
artifacts_v1_0/legacy1700.joblib
artifacts_v1_0/depth4_1892.joblib
artifacts_v1_0/catboost_companion.joblib
artifacts_v1_0/model_v1_0_config.json
artifacts_v1_0/model_v1_0_card.json
submissions_v1_0/submission_v1_0.csv
```

`submission_v1_0.csv` создаётся в формате:

```csv
prediction
...
```

То есть одна строка header `prediction`, затем `N` строк предсказаний для входного датасета.

## 3. Получить прогноз по уже обученным весам

```bash
bash scripts_v1_0/run_predict.sh   ../data/valid_features.csv   submissions_v1_0/submission_v1_0.csv
```

Для нового датасета:

```bash
bash scripts_v1_0/run_predict.sh   ../data/features_2027.csv   submissions_v1_0/submission_2027_v1_0.csv
```

## 4. Финальные веса

Финальный ансамбль использует веса из папки `artifacts_v1_0`:

```text
legacy1700.joblib
xgboost, 1700 trees, max_depth=3

depth4_1892.joblib
xgboost, 1892 trees, max_depth=4

catboost_companion.joblib
catboost companion model
```

Финальный blend зафиксирован в:

```text
configs_v1_0/final_config.json
artifacts_v1_0/model_v1_0_config.json
```

## 5. Собрать ZIP с весами

```bash
bash scripts_v1_0/package_delivery_with_artifacts.sh
```

Архив появится здесь:

```text
dist_v1_0/wind_v1_0_delivery_with_artifacts.zip
```
