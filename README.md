# wildDikks

# Прогнозирование выработки ВЭС — финальное решение V1.7

Решение предназначено для почасового прогноза выработки ветроэлектростанции по метеоданным и данным о ремонтах. Финальная версия поддерживает три режима работы:

1. обычный прогноз для `valid_features.csv` в формате лидерборда;
2. отдельный прогноз на 18.05.2026 по таблице, где за 17.05 выработка заполнена, а за 18.05 есть 24 строки с пустой колонкой выработки;
3. прогноз для любого другого CSV-файла с признаками.

Финальная модель воспроизводит лучшую найденную схему: production-ансамбль `legacy1700 + depth4_1892 + catboost_companion` и встроенный мягкий cap-relax слой. Важно: модель не читает старые submission-файлы, а полностью пересчитывает прогноз из обученных весов и входного CSV.

---

## 1. Структура проекта

Ожидаемая структура после распаковки патча:

```text
.
├── data/
│   ├── train_dataset.csv
│   ├── valid_features.csv
│   └── features_17_18.csv              # опционально, файл для прогноза на 18.05
├── model/
│   ├── configs_v1_7/
│   │   └── final_v1_7_weights.json
│   ├── new_model_v1_0/
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   ├── train_predict.py
│   │   ├── train_final_v1_7.py
│   │   ├── predict_final_v1_7.py
│   │   └── make_delivery_zip_v1_7.py
│   ├── scripts/
│   │   ├── train_final_v1_7.sh
│   │   ├── predict_valid_v1_7.sh
│   │   ├── predict_18_v1_7.sh
│   │   ├── predict_any_v1_7.sh
│   │   └── make_delivery_zip_v1_7.sh
│   └── requirements_final_v1_7.txt
├── README.md
└── explanatory_note.docx               # пояснительная записка для итогового zip
```

После обучения появятся папки:

```text
model/artifacts_v1_7/                  # веса модели, не загружать на GitHub
model/submissions_v1_7/                # созданные прогнозы, не загружать на GitHub
```

---

## 2. Что требуется установить

Рекомендуемая версия Python: **3.11**.

Из корня репозитория перейдите в папку `model`:

Установите пакет зависимостей:
```bash
pip install -r requirements.txt
```

```bash
cd model
```

На macOS для XGBoost может дополнительно потребоваться OpenMP:

```bash
brew install libomp
```

---

## 3. Входные данные

### 3.1. Обучающие данные

Файл:

```text
data/train_dataset.csv
```

Либо можете вставить любой другой датасет, но в эту директорию ($(pwd)/wildDikks/data/)

В нем должна быть целевая колонка:

```text
Выработка. Результирующий расчет
```

### 3.2. Обычный файл для лидерборда

Файл:

```text
data/valid_features.csv
```

Используется для обычного submission. На выходе создается один столбец с прогнозом.

### 3.3. Файл для прогноза на 18.05.2026

Файл, например:

```text
data/features_17_18.csv
```

Ожидаемая структура:

```text
строки за 17.05.2026: колонка "Выработка. Результирующий расчет" заполнена
строки за 18.05.2026: 24 строки, колонка "Выработка. Результирующий расчет" пустая
```

Модель использует весь файл для построения лаговых и сглаженных признаков, но на выход записывает только 24 прогноза для пустых строк 18.05.2026.

---

## 4. Обучение модели

Все команды ниже запускаются из папки `model`.

### 4.1. Только обучить модель

macOS, Linux:

```bash
bash scripts/train_final_v1_7.sh \
  ../data/train_dataset.csv \
  "" \
  submissions_v1_7/submission.csv \
  artifacts_v1_7
```

Windows:

```bash
bash scripts/train_final_v1_7.sh `
  ../data/train_dataset.csv `
  "" `
  submissions_v1_7/submission.csv `
  artifacts_v1_7
```

После выполнения появятся веса:

```text
artifacts_v1_7/legacy1700.joblib
artifacts_v1_7/depth4_1892.joblib
artifacts_v1_7/catboost_companion.joblib
artifacts_v1_7/final_v1_7_weights.json
artifacts_v1_7/model_v1_7_card.json
```

Файл `final_v1_7_weights.json` фиксирует веса ансамбля и параметры финального postprocessing-слоя.

---

## 5. Обычный прогноз для valid_features.csv

### 5.1. Обучить и сразу создать submission

macOS, Linux:

```bash
bash scripts/train_final_v1_7.sh \
  ../data/train_dataset.csv \
  ../data/valid_features.csv \
  submissions_v1_7/submission_valid.csv \
  artifacts_v1_7 \
  normal
```

Windows:

```bash
bash scripts/train_final_v1_7.sh `
  ../data/train_dataset.csv `
  ../data/valid_features.csv `
  submissions_v1_7/submission_valid.csv `
  artifacts_v1_7 `
  normal
```

Файл для отправки на обычную проверку:

```text
model/submissions_v1_7/submission_valid.csv
```

### 5.2. Создать обычный submission после уже выполненного обучения

macOS, Linux:

```bash
bash scripts/predict_valid_v1_7.sh \
  ../data/valid_features.csv \
  submissions_v1_7/submission_valid.csv \
  artifacts_v1_7
```

Windows:

```bash
bash scripts/predict_valid_v1_7.sh `
  ../data/valid_features.csv `
  submissions_v1_7/submission_valid.csv `
  artifacts_v1_7
```

По умолчанию bash-скрипт пишет файл с заголовком `prediction`, потому что такой формат использовался при последних успешных загрузках. Если нужен файл строго без заголовка, используйте прямой запуск Python:

macOS, Linux:

```bash
python -m new_model_v1_0.predict_final_v1_7 \
  --features_path ../data/valid_features.csv \
  --artifact_dir artifacts_v1_7 \
  --output_path submissions_v1_7/submission_valid_no_header.csv \
  --mode normal \
  --output_col prediction \
  --no_header
```

Windows:

```bash
python -m new_model_v1_0.predict_final_v1_7 `
  --features_path ../data/valid_features.csv `
  --artifact_dir artifacts_v1_7 `
  --output_path submissions_v1_7/submission_valid_no_header.csv `
  --mode normal `
  --output_col prediction `
  --no_header
```

---

## 6. Прогноз на 18.05.2026

Этот режим нужен для файла, где присутствуют строки за 17.05.2026 с заполненной выработкой и 24 строки за 18.05.2026 с пустой колонкой:

```text
Выработка. Результирующий расчет
```

### 6.1. Создать файл из 24 прогнозов

macOS, Linux:

```bash
bash scripts/predict_18_v1_7.sh \
  ../data/features_17_18.csv \
  submissions_v1_7/prediction_18_05.csv \
  artifacts_v1_7
```

Windows:

```bash
bash scripts/predict_18_v1_7.sh `
  ../data/features_17_18.csv `
  submissions_v1_7/prediction_18_05.csv `
  artifacts_v1_7
```

Выходной файл:

```text
model/submissions_v1_7/prediction_18_05.csv
```

Формат:

```text
24 строки
1 колонка
имя колонки: Выработка. Результирующий расчет
```

Проверить размер файла:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("submissions_v1_7/prediction_18_05.csv")
print(df.shape)
print(df.head())
PY
```

Ожидаемый результат:

```text
(24, 1)
```

### 6.2. Дополнительно сохранить полную таблицу с заполненными строками 18.05

macOS, Linux:

```bash
bash scripts/predict_18_v1_7.sh \
  ../data/features_17_18.csv \
  submissions_v1_7/prediction_18_05.csv \
  artifacts_v1_7 \
  submissions_v1_7/features_17_18_filled.csv
```

Windows:

```bash
bash scripts/predict_18_v1_7.sh `
  ../data/features_17_18.csv `
  submissions_v1_7/prediction_18_05.csv `
  artifacts_v1_7 `
  submissions_v1_7/features_17_18_filled.csv
```

В этом случае дополнительно появится:

```text
model/submissions_v1_7/features_17_18_filled.csv
```

Этот файл содержит исходную таблицу, где пустые значения выработки за 18.05 заполнены прогнозом.

---

## 7. Прогноз для любого другого CSV

Если нужно применить обученную модель к другому набору признаков:

macOS, Linux:

```bash
bash scripts/predict_any_v1_7.sh \
  ../data/any_features.csv \
  submissions_v1_7/prediction_any.csv \
  artifacts_v1_7 \
  prediction
```

Windows:

```bash
bash scripts/predict_any_v1_7.sh `
  ../data/any_features.csv `
  submissions_v1_7/prediction_any.csv `
  artifacts_v1_7 `
  prediction
```

Аргументы:

```text
1: путь к входному csv
2: путь к выходному csv
3: папка с обученными весами
4: название выходной колонки
```

Если нужен файл без заголовка:

macOS, Linux:

```bash
python -m new_model_v1_0.predict_final_v1_7 \
  --features_path ../data/any_features.csv \
  --artifact_dir artifacts_v1_7 \
  --output_path submissions_v1_7/prediction_any_no_header.csv \
  --mode generic \
  --output_col prediction \
  --no_header
```

Windows:

```bash
python -m new_model_v1_0.predict_final_v1_7 `
  --features_path ../data/any_features.csv `
  --artifact_dir artifacts_v1_7 `
  --output_path submissions_v1_7/prediction_any_no_header.csv `
  --mode generic `
  --output_col prediction `
  --no_header
```

---

---

---

---

---

---

---

---

## HELP

## 8. Частые проблемы

### XGBoost не запускается на macOS

Установите OpenMP:

```bash
brew install libomp
```

### Не найден модуль `new_model_v1_0`

Команды нужно запускать из папки `model`:

```bash
cd model
```

### Файл на 18.05 выдает не 24 строки

Проверьте, что в колонке:

```text
Выработка. Результирующий расчет
```

ровно 24 пустых значения для строк 18.05.2026. В режиме `day18` по умолчанию включена строгая проверка на 24 строки.

### Нужно получить CSV без заголовка

Используйте прямой запуск Python с флагом:

```bash
--no_header
```

Например:

macOS, Linux:

```bash
python -m new_model_v1_0.predict_final_v1_7 \
  --features_path ../data/valid_features.csv \
  --artifact_dir artifacts_v1_7 \
  --output_path submissions_v1_7/submission_valid_no_header.csv \
  --mode normal \
  --output_col prediction \
  --no_header
```

Windows:

```bash
python -m new_model_v1_0.predict_final_v1_7 `
  --features_path ../data/valid_features.csv `
  --artifact_dir artifacts_v1_7 `
  --output_path submissions_v1_7/submission_valid_no_header.csv `
  --mode normal `
  --output_col prediction `
  --no_header
```


