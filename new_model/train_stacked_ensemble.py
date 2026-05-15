"""
Train final N-model time-series stacking ensemble.

Default diverse XGBoost ensemble:
    python -m new_model.train_stacked_ensemble \
      --train_path data/train_dataset.csv \
      --model_path artifacts/stacked_ensemble.pkl \
      --n_models 5 \
      --n_splits 5 \
      --outlier_mode clip

With Optuna config:
    python -m new_model.train_stacked_ensemble \
      --train_path data/train_dataset.csv \
      --model_path artifacts/stacked_ensemble.pkl \
      --config_path artifacts/stacked_optuna/best_stacked_config.json \
      --outlier_mode clip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from models.stacked_ensemble import TimeSeriesStackingEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stacked wind ensemble.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--model_path", default="artifacts/stacked_ensemble.pkl")
    parser.add_argument("--metrics_path", default="artifacts/stacked_metrics.json")
    parser.add_argument("--config_path", default=None, help="Path to best_stacked_config.json from Optuna.")
    parser.add_argument("--n_models", type=int, default=5)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--meta_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outlier_mode", choices=["clip", "drop", "none"], default="clip")
    parser.add_argument("--tolerance_mw", type=float, default=0.5)
    parser.add_argument("--prediction_mode", choices=["fold_mean", "full"], default="fold_mean")
    return parser.parse_args()


def load_config(path: str | None) -> dict | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    print("[1/5] Loading data...")
    df = read_csv(args.train_path)
    df = normalize_columns(df)
    df = sort_by_datetime_if_possible(df)

    print("[2/5] Cleaning target outliers...")
    df, cleaning_report = clean_target_outliers(
        df,
        mode=args.outlier_mode,
        tolerance_mw=args.tolerance_mw,
        sort_by_time=False,
    )
    print(json.dumps(cleaning_report.to_dict(), ensure_ascii=False, indent=2))

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    config = load_config(args.config_path)

    if config is None:
        print("[3/5] Building default diverse XGBoost stacked config...")
        model_specs = TimeSeriesStackingEnsemble.default_specs(
            n_models=args.n_models,
            seed=args.seed,
            feature_flags=None,
        )
        ensemble = TimeSeriesStackingEnsemble(
            model_specs=model_specs,
            n_splits=args.n_splits,
            meta_alpha=args.meta_alpha,
            random_seed=args.seed,
            prediction_mode=args.prediction_mode,
        )
    else:
        print(f"[3/5] Building ensemble from config: {args.config_path}")
        ensemble = TimeSeriesStackingEnsemble.from_config(config)

    print("[4/5] Fitting stacked ensemble...")
    ensemble.fit(df)

    print("[5/5] Saving model and metrics...")
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(model_path)

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    train_pred = ensemble.predict(df)
    train_pred = clip_predictions_to_available_capacity(train_pred, df)

    train_mae_mw = float(np.mean(np.abs(y.to_numpy() - train_pred)))
    train_error_percent = competition_error_percent(y, train_pred)

    metrics = {
        "model_path": str(model_path),
        "rows": int(len(df)),
        "n_base_models": int(len(ensemble.model_specs)),
        "n_splits": int(ensemble.n_splits),
        "meta_alpha": float(ensemble.meta_alpha),
        "prediction_mode": ensemble.prediction_mode,
        "oof_mae_mw": ensemble.oof_score_,
        "train_mae_mw": train_mae_mw,
        "train_competition_error_percent": train_error_percent,
        "cleaning_report": cleaning_report.to_dict(),
    }

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    ensemble.meta_weights().to_csv(model_path.parent / "stacked_meta_weights.csv", index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"OOF MAE: {ensemble.oof_score_:.5f} MW")
    print(f"Train MAE: {train_mae_mw:.5f} MW")
    print(f"Train competition error: {train_error_percent:.5f}%")


if __name__ == "__main__":
    main()
