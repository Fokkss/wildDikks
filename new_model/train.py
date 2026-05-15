"""
python -m new_model.train \
      --train_path data/train_dataset.csv \
      --model_dir artifacts \
      --seed 42 \
      --valid_size 0.2 \
      --catboost_weight 0.5 \
      --xgboost_weight 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from new_model.config import (
    TARGET_COL,
    FARM_CAPACITY_MW,
    read_csv,
    normalize_columns,
    sort_by_datetime_if_possible,
    clip_predictions_to_available_capacity,
    competition_error_percent,
)

from models import WeightedEnsembleModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost + XGBoost ensemble.")

    parser.add_argument(
        "--train_path",
        required=True,
        help="Path to train_dataset.csv",
    )

    parser.add_argument(
        "--model_dir",
        default="artifacts",
        help="Directory where model and metrics will be saved.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--valid_size",
        type=float,
        default=0.2,
        help="Last fraction of chronological data used for local validation.",
    )

    parser.add_argument(
        "--catboost_weight",
        type=float,
        default=0.5,
        help="Weight of CatBoost predictions in ensemble.",
    )

    parser.add_argument(
        "--xgboost_weight",
        type=float,
        default=0.5,
        help="Weight of XGBoost predictions in ensemble.",
    )

    return parser.parse_args()


def ensure_target_exists(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column was not found after normalization. "
            f"Expected normalized target column: {TARGET_COL!r}. "
            f"Available columns: {list(df.columns)}"
        )


def split_chronological(
    df: pd.DataFrame,
    valid_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < valid_size < 0.9:
        raise ValueError("--valid_size must be between 0.0 and 0.9")

    split_idx = int(len(df) * (1.0 - valid_size))

    train_part = df.iloc[:split_idx].reset_index(drop=True)
    valid_part = df.iloc[split_idx:].reset_index(drop=True)

    return train_part, valid_part


def save_feature_importance(
    model: WeightedEnsembleModel,
    model_dir: Path,
) -> None:
    try:
        importances = model.get_feature_importance()

        for model_name, importance_df in importances.items():
            importance_df.to_csv(
                model_dir / f"importance_{model_name}.csv",
                index=False,
            )

    except Exception as exc:
        print(f"[WARN] Could not save feature importance: {exc}")


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading train data...")
    raw_df = read_csv(args.train_path)

    print("[2/6] Normalizing columns...")
    df = normalize_columns(raw_df)
    ensure_target_exists(df)

    print("[3/6] Sorting train data by datetime...")
    df = sort_by_datetime_if_possible(df)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Target: {TARGET_COL}")
    print(f"Farm capacity MW: {FARM_CAPACITY_MW}")

    train_part, valid_part = split_chronological(
        df=df,
        valid_size=args.valid_size,
    )

    print(f"Train rows: {len(train_part)}")
    print(f"Local valid rows: {len(valid_part)}")

    print("[4/6] Training temporary validation model...")
    validation_model = WeightedEnsembleModel(
        catboost_weight=args.catboost_weight,
        xgboost_weight=args.xgboost_weight,
    )

    validation_model.fit(
        train_df=train_part,
        valid_df=valid_part,
    )

    y_valid = pd.to_numeric(valid_part[TARGET_COL], errors="coerce")
    valid_pred = validation_model.predict(valid_part)
    valid_pred = clip_predictions_to_available_capacity(valid_pred, valid_part)

    valid_error_percent = competition_error_percent(y_valid, valid_pred)
    valid_mae_mw = float(np.mean(np.abs(y_valid.to_numpy() - valid_pred)))

    print(f"[LOCAL VALID] MAE: {valid_mae_mw:.4f} MW")
    print(f"[LOCAL VALID] Competition error: {valid_error_percent:.4f}%")

    pd.DataFrame(
        {
            "actual": y_valid,
            "prediction": valid_pred,
            "abs_error": np.abs(y_valid.to_numpy() - valid_pred),
        }
    ).to_csv(model_dir / "local_validation_predictions.csv", index=False)

    print("[5/6] Training final model on ALL available train data...")
    final_model = WeightedEnsembleModel(
        catboost_weight=args.catboost_weight,
        xgboost_weight=args.xgboost_weight,
    )

    final_model.fit(train_df=df)

    final_model_path = model_dir / "ensemble.pkl"
    final_model.save(final_model_path)

    print(f"Saved model: {final_model_path}")

    print("[6/6] Saving metrics and feature importances...")
    train_y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    train_pred = final_model.predict(df)
    train_pred = clip_predictions_to_available_capacity(train_pred, df)

    train_error_percent = competition_error_percent(train_y, train_pred)
    train_mae_mw = float(np.mean(np.abs(train_y.to_numpy() - train_pred)))

    metrics = {
        "seed": args.seed,
        "rows": int(len(df)),
        "valid_size": args.valid_size,
        "catboost_weight": args.catboost_weight,
        "xgboost_weight": args.xgboost_weight,
        "local_valid_mae_mw": valid_mae_mw,
        "local_valid_competition_error_percent": valid_error_percent,
        "train_mae_mw": train_mae_mw,
        "train_competition_error_percent": train_error_percent,
        "model_path": str(final_model_path),
    }

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_feature_importance(final_model, model_dir)

    print("Done.")
    print(f"Metrics: {model_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()