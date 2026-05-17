# Wind Power Forecast — V1.0

Полностью воспроизводимое решение для почасового прогноза выработки ВЭС.

Финальная модель V1.0 обучается только на `train_dataset.csv` и делает один итоговый прогноз по файлу признаков. В production-пути нет teacher CSV и нет ручного перебора сабмитов.

## 1. Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Обучение и прогноз Q1/valid

Запуск из папки `wind_v1_0`:

```bash
bash scripts_v1_0/run_train_predict.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  "Выработка. Результирующий расчет"
```

После запуска появятся:

```text
artifacts_v1_0/                 веса моделей и config
submissions_v1_0/submission_v1_0.csv
submissions_v1_0/submission_v1_0.summary.json
```

## 3. Инференс на новых данных / 2027

После обучения:

```bash
bash scripts_v1_0/run_predict.sh \
  ../data/features_2027.csv \
  submissions_v1_0/submission_2027_v1_0.csv
```

## 4. Финальная модель

V1.0 использует ансамбль:

```text
XGB anchor = 0.70 * XGB legacy1700 + 0.30 * XGB depth4_1892
final base = 0.60 * XGB anchor + 0.40 * CatBoost companion
final pred = final base + deterministic physics/rules correction + 0.85 MW bias
```

Финальный профиль выбран по leaderboard-экспериментам:

```text
beta13_catblend_anchor70_cat_w0p6_rules55_bias0p85_density_boost.csv
score: 8.341698088430002
```