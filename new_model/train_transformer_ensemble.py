"""
Train Transformer ensemble for wind-farm generation forecasting.

Default run:
    python -m new_model.train_transformer_ensemble \
      --train_path data/train_dataset.csv \
      --model_path artifacts/transformer_ensemble.pkl \
      --outlier_mode clip

With Optuna config:
    python -m new_model.train_transformer_ensemble \
      --train_path data/train_dataset.csv \
      --model_path artifacts/transformer_ensemble.pkl \
      --config_path artifacts/transformer_optuna/best_transformer_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from new_model.config import (
    TARGET_COL,
    read_csv,
    normalize_columns,
    sort_by_datetime_if_possible,
    clip_predictions_to_available_capacity,
    competition_error_percent,
)
from new_model.outlier_cleaning import clean_target_outliers
from models.transformer_model import TransformerEnsembleModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer ensemble.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--model_path", default="artifacts/transformer_ensemble.pkl")
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--n_models", type=int, default=3)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--outlier_mode", choices=["clip", "drop", "none"], default="clip")
    parser.add_argument("--tolerance_mw", type=float, default=0.5)
    return parser.parse_args()


def split_chronological(df: pd.DataFrame, valid_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < valid_size < 0.9:
        raise ValueError("--valid_size must be between 0.0 and 0.9")
    split_idx = int(len(df) * (1.0 - valid_size))
    return (
        df.iloc[:split_idx].reset_index(drop=True),
        df.iloc[split_idx:].reset_index(drop=True),
    )


def load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)

    print("[1/7] Loading train data...")
    df = read_csv(args.train_path)

    print("[2/7] Normalizing and sorting...")
    df = normalize_columns(df)
    df = sort_by_datetime_if_possible(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    print("[3/7] Cleaning outliers...")
    df, outlier_report = clean_target_outliers(
        df=df,
        mode=args.outlier_mode,
        tolerance_mw=args.tolerance_mw,
    )
    print(json.dumps(outlier_report, ensure_ascii=False, indent=2))

    config = load_config(args.config_path)
    params = config.get("params", config)
    n_models = int(config.get("n_models", args.n_models))

    print(f"Rows after cleaning: {len(df)}")
    print(f"Transformer models: {n_models}")
    print(f"Params: {params}")

    train_part, valid_part = split_chronological(df, args.valid_size)

    print("[4/7] Training validation Transformer ensemble...")
    validation_model = TransformerEnsembleModel.from_config(
        n_models=n_models,
        seed=args.seed,
        params=params,
    )
    validation_model.fit(train_df=train_part, valid_df=valid_part)

    y_valid = pd.to_numeric(valid_part[TARGET_COL], errors="coerce").to_numpy()
    valid_pred = validation_model.predict(valid_part)
    valid_pred = clip_predictions_to_available_capacity(valid_pred, valid_part)

    valid_mae_mw = float(np.mean(np.abs(y_valid - valid_pred)))
    valid_error_percent = competition_error_percent(y_valid, valid_pred)

    print(f"[LOCAL VALID] MAE: {valid_mae_mw:.4f} MW")
    print(f"[LOCAL VALID] Competition error: {valid_error_percent:.4f}%")

    print("[5/7] Training final Transformer ensemble on ALL data...")
    final_model = TransformerEnsembleModel.from_config(
        n_models=n_models,
        seed=args.seed,
        params=params,
    )
    final_model.fit(train_df=df, valid_df=None)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(model_path)
    print(f"Saved model: {model_path}")

    print("[6/7] Computing train metrics...")
    y_train = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy()
    train_pred = final_model.predict(df)
    train_pred = clip_predictions_to_available_capacity(train_pred, df)

    train_mae_mw = float(np.mean(np.abs(y_train - train_pred)))
    train_error_percent = competition_error_percent(y_train, train_pred)

    metrics = {
        "seed": args.seed,
        "n_models": n_models,
        "params": params,
        "outlier_report": outlier_report,
        "valid_size": args.valid_size,
        "local_valid_mae_mw": valid_mae_mw,
        "local_valid_competition_error_percent": valid_error_percent,
        "train_mae_mw": train_mae_mw,
        "train_competition_error_percent": train_error_percent,
        "model_path": str(model_path),
    }

    print("[7/7] Saving metrics...")
    metrics_path = Path(args.metrics_path) if args.metrics_path else model_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics: {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
