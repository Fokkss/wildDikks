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

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


from config import FARM_CAPACITY_MW

from feature_engineering import (
    available_capacity_from_raw,
    find_datetime_col,
    find_target_col,
    make_features,
    sort_by_time_if_possible,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(
    seed: int, n_estimators: int = 5000, early_stopping_rounds: int | None = None
) -> XGBRegressor:
    params = dict(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.04,
        reg_lambda=5.0,
        objective="reg:absoluteerror",  # good default when leaderboard is MAE
        eval_metric="mae",
        tree_method="hist",
        max_bin=256,
        random_state=seed,
        n_jobs=-1,
    )
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
    return XGBRegressor(**params)


def select_feature_columns(
    fe: pd.DataFrame, target_col: str, drop_cols: list[str] | None = None
) -> list[str]:
    drop = set(drop_cols or []) | {target_col}
    cols: list[str] = []
    for col in fe.columns:
        if col in drop:
            continue
        if pd.api.types.is_numeric_dtype(fe[col]):
            # Drop fully empty columns; SimpleImputer cannot learn a median for them.
            if fe[col].notna().sum() > 0:
                cols.append(col)
    return cols


def load_csv(path: str) -> pd.DataFrame:
    # utf-8-sig handles Excel-exported CSV files with BOM.
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True, help="Path to train_dataset.csv")
    parser.add_argument(
        "--model_dir", default="artifacts", help="Where to save model artifacts"
    )
    parser.add_argument(
        "--target", default="Результирующий расчет", help="Target column name"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--capacity_mw", type=float, default=FARM_CAPACITY_MW)
    parser.add_argument("--n_estimators", type=int, default=6000)
    parser.add_argument("--early_stopping_rounds", type=int, default=150)

    args = parser.parse_args()

    set_seed(args.seed)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    raw = load_csv(args.train_path)
    target_col = find_target_col(raw, args.target)
    raw = sort_by_time_if_possible(raw)

    # Target sanity clipping: impossible values should not dominate training.
    y_raw = pd.to_numeric(raw[target_col], errors="coerce")
    row_cap = available_capacity_from_raw(raw).fillna(args.capacity_mw)
    y = y_raw.clip(lower=0.0, upper=row_cap)

    valid_rows = y.notna()
    raw = raw.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    fe = make_features(raw, target_col=target_col)
    dt_col = find_datetime_col(fe)
    drop_cols = [dt_col] if dt_col is not None else []
    feature_cols = select_feature_columns(
        fe, target_col=target_col, drop_cols=drop_cols
    )

    X = fe[feature_cols]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    n_splits = min(args.n_splits, max(2, len(X_imp) // 500))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_imp), start=1):
        X_train, X_val = X_imp.iloc[train_idx], X_imp.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model(
            seed=args.seed + fold,
            n_estimators=args.n_estimators,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        pred = model.predict(X_val)
        cap_val = available_capacity_from_raw(raw.iloc[val_idx]).to_numpy()
        pred = np.clip(pred, 0.0, cap_val)

        mae = mean_absolute_error(y_val, pred)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = model.get_params()["n_estimators"]
        else:
            best_iter += 1
        best_iterations.append(int(best_iter))
        fold_rows.append(
            {
                "fold": fold,
                "mae_mw": mae,
                "rmse_mw": rmse,
                "best_iteration": int(best_iter),
            }
        )
        print(
            f"Fold {fold}: MAE={mae:.4f} MW | RMSE={rmse:.4f} MW | best_iter={best_iter}"
        )

    cv = pd.DataFrame(fold_rows)
    print("\nCV summary")
    print(cv.to_string(index=False))
    print(f"Mean MAE:  {cv['mae_mw'].mean():.4f} MW")
    print(f"Mean RMSE: {cv['rmse_mw'].mean():.4f} MW")

    # Train final model on all data with number of trees inferred from time-series CV.
    final_estimators = int(
        np.clip(np.median(best_iterations) * 1.10, 30, args.n_estimators)
    )
    final_model = build_model(
        seed=args.seed, n_estimators=final_estimators, early_stopping_rounds=None
    )
    final_model.fit(X_imp, y, verbose=False)

    artifact = {
        "model": final_model,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "capacity_mw": args.capacity_mw,
        "seed": args.seed,
        "cv": fold_rows,
    }
    joblib.dump(artifact, model_dir / "wind_xgb_model.joblib")
    cv.to_csv(model_dir / "cv_metrics.csv", index=False)
    with open(model_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(model_dir / "feature_importance.csv", index=False)

    print(f"\nSaved artifacts to: {model_dir}")
    print(f"Final n_estimators: {final_estimators}")
    print("Top-20 features:")
    print(importance.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
