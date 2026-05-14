"""Example:
    python -m new_model.train \
      --train_path data/train_dataset.csv \
      --model_dir output/artifacts \
      --cv
    MyExample:
    1) python -m new_model/train.py --train_path dataset/train_dataset.csv
    2) python -m new_model/predict.py --features_path dataset/valid_features.csv --model_dir artifacts --output_path submission.csv
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
from new_model.config import FARM_CAPACITY_MW, COLUMN_MAP
from new_model.feature_engineering import (
    make_features,
    available_capacity_from_raw,
    sort_by_time_if_possible,
)


def build_model(seed: int) -> XGBRegressor:
    """Инкапсуляция настроек модели. Возвращает готовый инстанс бустинга."""
    return XGBRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=7,
        min_child_weight=5,
        gamma=0.2,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.10,
        reg_lambda=6.0,
        objective="reg:absoluteerror",  # good default when Leaderboard is MAE
        eval_metric="mae",
        tree_method="hist",
        max_bin=256,
        random_state=seed,
        n_jobs=-1,
        # early_stopping_rounds=50,
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

# ==========================================================
# VALIDATION OF DATA
# ==========================================================
def get_feature_columns(fe_df: pd.DataFrame, target_col: str) -> list[str]:
    """
    select numeric features only, excluding target.
    """
    return [
        c
        for c in fe_df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(fe_df[c])
    ]


def time_holdout_score(
        X: pd.DataFrame,
        y: pd.Series,
        raw_df: pd.DataFrame,
        seed: int,
        valid_size: float = 0.2,
) -> float:
    """
    simple chronological holdout validation.
    last valid_size fraction is used as validation set.
    """
    split_idx = int(len(X) * (1.0 - valid_size))

    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

    model = build_model(seed=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
    )

    pred = model.predict(X_valid)

    valid_cap = available_capacity_from_raw(raw_df.iloc[split_idx:]).to_numpy()
    pred = np.clip(pred, 0.0, valid_cap)

    mae = mean_absolute_error(y_valid, pred)
    return float(mae)


def timeseries_cv_score(
        X: pd.DataFrame,
        y: pd.Series,
        raw_df: pd.DataFrame,
        seed: int,
        n_splits: int = 5,
) -> list[float]:
    """
    TimeSeriesSplit validation.
    This is more reliable than random split for forecasting.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = build_model(seed=seed + fold)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        pred = model.predict(X_valid)

        valid_cap = available_capacity_from_raw(raw_df.iloc[valid_idx]).to_numpy()
        pred = np.clip(pred, 0.0, valid_cap)

        mae = mean_absolute_error(y_valid, pred)
        scores.append(float(mae))

        print(f"[CV] Fold {fold}: MAE = {mae:.4f} MW")

    print(f"[CV] Mean MAE: {np.mean(scores):.4f} MW")
    print(f"[CV] Std MAE:  {np.std(scores):.4f} MW")

    return scores


# ==========================================================
# BEGIN TRAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path",
                        required=True,
                        help="Путь к обучающим данным")
    parser.add_argument("--model_dir",
                        default="artifacts",
                        help="Куда сохранять веса")
    parser.add_argument("--seed",
                        type=int, default=42,
                        help="Seed")
    parser.add_argument("--cv",
                        action="store_true",
                        help="Run TimeSeriesSplit CV before final training")
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds",
    )
    args = parser.parse_args()

    # 1. ПОДГОТОВКА И ДАННЫЕ
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("loading data...")
    raw_df = pd.read_csv(args.train_path, encoding="utf-8-sig")

    print("sorting by time if datetime column exists...")
    raw_df = sort_by_time_if_possible(raw_df)

    # [FIX] Более гибкий поиск целевой колонки
    target_raw_name = next((k for k, v in COLUMN_MAP.items() if v == "target"), None)
    if target_raw_name not in raw_df.columns:
        # Если точного совпадения нет, ищем по ключевому слову
        potential = [c for c in raw_df.columns if "Результирующий" in c] #.расчет
        target_col = potential[0] if potential else target_raw_name
    else:
        target_col = target_raw_name

    print(f"Target column identified as: {target_col}")

    # Извлекаем фичи через наш красивый движок
    print("generating features...")
    fe_df = make_features(raw_df, target_col=target_col)

    # Выделяем матрицу признаков (X) и ответы (y)
    # Исключаем целевую переменную и колонку с датой (коды категорий уже числа)
    feature_cols = get_feature_columns(fe_df, target_col)

    X = fe_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # target cleaning and physical clipping
    row_cap = available_capacity_from_raw(raw_df)
    y = pd.to_numeric(raw_df[target_col], errors="coerce")
    y = y.clip(lower=0.0, upper=row_cap)

    # Remove rows without target
    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    raw_df = raw_df.loc[mask].reset_index(drop=True)

    # logger (preferred to move to other class and flag for log / e.g. --log /)
    print(f"rows: {len(X)}")
    print(f"features: {len(feature_cols)}")

    print("Fitting imputer...")
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X),
        columns=feature_cols,
    )

    # validation
    print("running chronological holdout validation...")
    holdout_mae = time_holdout_score(
        X=X_imp,
        y=y,
        raw_df=raw_df,
        seed=args.seed,
        valid_size=0.2,
    )
    print(f"[HOLDOUT] MAE: {holdout_mae:.4f} MW")

    cv_scores = None
    if args.cv:
        print("running TimeSeriesSplit CV...")
        cv_scores = timeseries_cv_score(
            X=X_imp,
            y=y,
            raw_df=raw_df,
            seed=args.seed,
            n_splits=args.n_splits,
        )

    # 3. ОБУЧЕНИЕ (Упрощенная валидация)
    print("training final model on all data...")
    model = build_model(seed=args.seed)
    split_idx = int(len(X_imp) * 0.9)

    X_train = X_imp.iloc[:split_idx]
    X_valid = X_imp.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_valid = y.iloc[split_idx:]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
    )

    # Оценка на тренировочной выборке (просто чтобы убедиться, что учится)
    pred = model.predict(X_imp)
    cap = available_capacity_from_raw(raw_df).to_numpy()
    pred = np.clip(pred, 0.0, cap)

    mae = mean_absolute_error(y, pred)
    print(f"[SUCCESS] Inner MAE on train: {mae:.4f} MW")

    artifact = {
        "model": model,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "capacity_mw": FARM_CAPACITY_MW,
        "target_col": target_col,
        "seed": args.seed,
        "holdout_mae": holdout_mae,
        "cv_scores": cv_scores,
    }

    joblib.dump(artifact, model_dir / "wind_xgb_model.joblib")

    # Сохраняем топ фичей
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(model_dir / "importance.csv", index=False)

    metrics = {
        "train_mae": mae,
        "holdout_mae": holdout_mae,
        "cv_scores": cv_scores,
        "feature_count": len(feature_cols),
        "rows": len(X),
    }

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"artifacts saved to: {model_dir}")
    print(f"model: {model_dir / 'wind_xgb_model.joblib'}")
    print(f"importance: {model_dir / 'importance.csv'}")
    print(f"metrics: {model_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
