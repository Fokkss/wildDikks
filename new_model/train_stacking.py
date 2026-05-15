from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from new_model.outlier_cleaning import clean_target_outliers
from models.stacking_ensemble import TimeSeriesStackingEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train time-series stacking ensemble.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--model_path", default="artifacts/stacking_ensemble.pkl")
    parser.add_argument("--config_path", default=None, help="Optional config from Optuna.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_xgb", type=int, default=4)
    parser.add_argument("--n_cat", type=int, default=1)
    parser.add_argument("--meta_model", choices=["ridge", "linear", "xgb", "lightgbm"], default="ridge")
    parser.add_argument("--prediction_mode", choices=["full", "fold_mean", "blend_full_fold"], default="full")
    parser.add_argument("--outlier_mode", choices=["none", "clip", "drop"], default="clip")
    parser.add_argument("--tolerance_mw", type=float, default=0.5)
    return parser.parse_args()


def split_chronological(df: pd.DataFrame, valid_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < valid_size < 0.9:
        raise ValueError("valid_size must be between 0 and 0.9")
    split_idx = int(len(df) * (1.0 - valid_size))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def load_config(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_from_args_or_config(args: argparse.Namespace, config: dict[str, Any] | None) -> TimeSeriesStackingEnsemble:
    if config is not None:
        return TimeSeriesStackingEnsemble.from_config(config)

    specs = TimeSeriesStackingEnsemble.default_specs(
        n_xgb=args.n_xgb,
        n_cat=args.n_cat,
        seed=args.seed,
    )

    return TimeSeriesStackingEnsemble(
        base_model_specs=specs,
        meta_model_type=args.meta_model,
        n_splits=args.n_splits,
        random_seed=args.seed,
        prediction_mode=args.prediction_mode,
        append_meta_stats=True,
    )


def evaluate(model: TimeSeriesStackingEnsemble, valid_df: pd.DataFrame) -> dict[str, float]:
    y_valid = pd.to_numeric(valid_df[TARGET_COL], errors="coerce")
    pred = model.predict(valid_df)
    pred = clip_predictions_to_available_capacity(pred, valid_df)

    mae_mw = float(np.mean(np.abs(y_valid.to_numpy(dtype=float) - pred)))
    error_percent = competition_error_percent(y_valid, pred)

    return {
        "mae_mw": mae_mw,
        "competition_error_percent": error_percent,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading data...")
    raw_df = read_csv(args.train_path)

    print("[2/7] Normalizing and sorting...")
    df = normalize_columns(raw_df)
    df = sort_by_datetime_if_possible(df)

    print("[3/7] Cleaning outliers...")
    df, outlier_report = clean_target_outliers(
        df,
        mode=args.outlier_mode,
        tolerance_mw=args.tolerance_mw,
        normalize=False,
        sort_by_time=False,
    )
    print(json.dumps(outlier_report.to_dict(), ensure_ascii=False, indent=2))

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Farm capacity MW: {FARM_CAPACITY_MW}")

    config = load_config(args.config_path)
    if config:
        print(f"Loaded config: {args.config_path}")

    train_part, valid_part = split_chronological(df, args.valid_size)
    print(f"Train rows: {len(train_part)}")
    print(f"Local valid rows: {len(valid_part)}")

    print("[4/7] Training local validation stacking model...")
    validation_model = build_model_from_args_or_config(args, config)
    validation_model.fit(train_part)

    local_metrics = evaluate(validation_model, valid_part)
    print(f"[LOCAL VALID] MAE: {local_metrics['mae_mw']:.4f} MW")
    print(f"[LOCAL VALID] Competition error: {local_metrics['competition_error_percent']:.4f}%")
    print(f"[OOF INSIDE TRAIN] MAE: {validation_model.oof_mae_mw_:.4f} MW")

    local_pred = validation_model.predict(valid_part)
    local_pred = clip_predictions_to_available_capacity(local_pred, valid_part)
    pd.DataFrame(
        {
            "actual": pd.to_numeric(valid_part[TARGET_COL], errors="coerce"),
            "prediction": local_pred,
            "abs_error": np.abs(pd.to_numeric(valid_part[TARGET_COL], errors="coerce").to_numpy(dtype=float) - local_pred),
        }
    ).to_csv(model_path.parent / "stacking_local_validation_predictions.csv", index=False)

    print("[5/7] Training final stacking model on ALL data...")
    final_model = build_model_from_args_or_config(args, config)
    final_model.fit(df)
    final_model.save(model_path)
    print(f"Saved model: {model_path}")

    print("[6/7] Evaluating train fit for diagnostics...")
    train_pred = final_model.predict(df)
    train_pred = clip_predictions_to_available_capacity(train_pred, df)
    train_y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    train_mae = float(np.mean(np.abs(train_y.to_numpy(dtype=float) - train_pred)))
    train_error = competition_error_percent(train_y, train_pred)

    print("[7/7] Saving metrics and config...")
    metrics = {
        "seed": args.seed,
        "train_path": args.train_path,
        "model_path": str(model_path),
        "rows": int(len(df)),
        "valid_size": args.valid_size,
        "outlier_report": outlier_report.to_dict(),
        "local_valid_mae_mw": local_metrics["mae_mw"],
        "local_valid_competition_error_percent": local_metrics["competition_error_percent"],
        "local_oof_mae_mw_inside_train_part": validation_model.oof_mae_mw_,
        "final_oof_mae_mw_all_data": final_model.oof_mae_mw_,
        "train_mae_mw": train_mae,
        "train_competition_error_percent": train_error,
        "final_model_diagnostics": final_model.diagnostics(),
    }

    with open(model_path.parent / "stacking_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(model_path.parent / "stacking_config.json", "w", encoding="utf-8") as f:
        json.dump(final_model.to_config(), f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Metrics: {model_path.parent / 'stacking_metrics.json'}")
    print(f"Config: {model_path.parent / 'stacking_config.json'}")


if __name__ == "__main__":
    main()
