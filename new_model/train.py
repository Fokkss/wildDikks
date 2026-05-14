"""Train XGBoost model for wind farm hourly power forecasting.

Example:
    python train.py \
      --train_path ../data/train_dataset.csv \
      --model_dir output/artifacts \
      --target "Результирующий расчет" \
      --n_splits 2000 \
      --n_estimators 2000 \
      --early_stopping_rounds 10
"""

"""
Запуск: python train.py --train_path ../dataset/train.csv
python new_model/predict.py 
--features_path dataset/valid_features.csv 
--model_dir artifacts 
--output_path submission.csv
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# Импорты из локальных файлов (без префикса new_model)
from new_model.config import FARM_CAPACITY_MW
from new_model.feature_engineering import make_features, find_target_col, available_capacity_from_raw


def build_model(seed: int) -> XGBRegressor:
    """Инкапсуляция настроек модели. Возвращает готовый инстанс бустинга."""
    return XGBRegressor(
        n_estimators=1000,  # Уменьшено для скорости, на финале ставь 5000
        learning_rate=0.03,
        max_depth=4,
        objective="reg:absoluteerror",  # Оптимизируем напрямую под метрику хакатона (MAE)
        eval_metric="mae",
        random_state=seed,
        n_jobs=-1
    )
# previous params
# n_estimators = n_estimators,
# learning_rate = 0.03,
# max_depth = 3,
# min_child_weight = 5,
# subsample = 0.85,
# colsample_bytree = 0.85,
# reg_alpha = 0.04,
# reg_lambda = 5.0,
# objective = "reg:absoluteerror",  # good default when leaderboard is MAE
# eval_metric = "mae",
# tree_method = "hist",
# max_bin = 256,
# random_state = seed,
# n_jobs = -1,


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True, help="Путь к обучающим данным")
    parser.add_argument("--model_dir", default="artifacts", help="Куда сохранять веса")
    args = parser.parse_args()

    # 1. ПОДГОТОВКА И ДАННЫЕ
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.train_path)
    target_col = find_target_col(raw_df, "Результирующий расчет")

    # Извлекаем фичи через наш красивый движок
    print("Генерация признаков...")
    fe_df = make_features(raw_df, target_col=target_col)

    # Выделяем матрицу признаков (X) и ответы (y)
    # Исключаем целевую переменную и колонку с датой (коды категорий уже числа)
    feature_cols = [
        c
        for c in fe_df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(fe_df[c])]

    X = fe_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(raw_df[target_col], errors="coerce").fillna(0)

    # 2. ИМПУТАЦИЯ (Заполнение пропусков)
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    # 3. ОБУЧЕНИЕ (Упрощенная валидация)
    print(f"Старт обучения модели на {len(feature_cols)} признаках...")
    model = build_model(seed=42)
    model.fit(X_imp, y, verbose=False)

    # Оценка на тренировочной выборке (просто чтобы убедиться, что учится)
    pred = model.predict(X_imp)
    mae = mean_absolute_error(y, pred)
    print(f"[УСПЕХ] Внутренний MAE на трейне: {mae:.4f} MW")

    artifact = {
        "model": model,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "capacity_mw": FARM_CAPACITY_MW,
    }

    joblib.dump(artifact, model_dir / "wind_xgb_model.joblib")

    # Сохраняем топ фичей
    importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(model_dir / "importance.csv", index=False)

    print(f"Артефакты сохранены в: {model_dir}")


if __name__ == "__main__":
    main()