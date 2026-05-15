"""
Optuna tuning for N-model time-series stacking ensemble.

Fast start:
    python -m new_model.tune_stacked_ensemble_optuna \
      --train_path data/train_dataset.csv \
      --output_dir artifacts/stacked_optuna \
      --n_trials 20 \
      --max_models 5 \
      --n_splits 3 \
      --outlier_mode clip

The script tunes:
- number of base XGBoost models;
- each base model hyperparameters;
- feature groups;
- Ridge meta-model alpha.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
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
from models.base import DEFAULT_FEATURE_FLAGS
from models.stacked_ensemble import TimeSeriesStackingEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune stacked ensemble with Optuna.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_dir", default="artifacts/stacked_optuna")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--max_models", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outlier_mode", choices=["clip", "drop", "none"], default="clip")
    parser.add_argument("--tolerance_mw", type=float, default=0.5)
    parser.add_argument(
        "--tune_core_features",
        action="store_true",
        help="If set, Optuna may disable time/availability/power_curve. Default keeps them enabled.",
    )
    return parser.parse_args()


def split_chronological(df: pd.DataFrame, valid_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < valid_size < 0.9:
        raise ValueError("valid_size must be between 0.0 and 0.9")
    split_idx = int(len(df) * (1.0 - valid_size))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def suggest_feature_flags(trial: optuna.Trial, tune_core_features: bool) -> dict[str, bool]:
    flags = DEFAULT_FEATURE_FLAGS.copy()

    if tune_core_features:
        flags["time"] = trial.suggest_categorical("use_time", [True, False])
        flags["availability"] = trial.suggest_categorical("use_availability", [True, False])
        flags["power_curve"] = trial.suggest_categorical("use_power_curve", [True, False])
    else:
        flags["time"] = True
        flags["availability"] = True
        flags["power_curve"] = True

    # These two sometimes help, sometimes add noise depending on weather data quality.
    flags["air_density"] = trial.suggest_categorical("use_air_density", [True, False])
    flags["wind_power_density"] = trial.suggest_categorical("use_wind_power_density", [True, False])

    return flags


def suggest_xgb_params(trial: optuna.Trial, model_idx: int, seed: int) -> dict[str, Any]:
    prefix = f"m{model_idx}_"
    return {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": trial.suggest_int(prefix + "n_estimators", 800, 3000),
        "learning_rate": trial.suggest_float(prefix + "learning_rate", 0.008, 0.05, log=True),
        "max_depth": trial.suggest_int(prefix + "max_depth", 3, 6),
        "min_child_weight": trial.suggest_int(prefix + "min_child_weight", 4, 24),
        "subsample": trial.suggest_float(prefix + "subsample", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float(prefix + "colsample_bytree", 0.55, 0.95),
        "reg_alpha": trial.suggest_float(prefix + "reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float(prefix + "reg_lambda", 1.0, 20.0, log=True),
        "tree_method": "hist",
        "max_bin": trial.suggest_categorical(prefix + "max_bin", [128, 256, 512]),
        "random_state": seed + model_idx,
        "n_jobs": -1,
        "early_stopping_rounds": 150,
    }


def build_model_specs(
    trial: optuna.Trial,
    max_models: int,
    seed: int,
    feature_flags: dict[str, bool],
) -> list[dict[str, Any]]:
    n_models = trial.suggest_int("n_models", 2, max_models)
    specs = []

    for i in range(n_models):
        specs.append(
            {
                "name": f"xgb_{i}",
                "model_type": "xgb",
                "seed_offset": i,
                "params": suggest_xgb_params(trial, model_idx=i, seed=seed),
                "feature_flags": feature_flags.copy(),
            }
        )

    return specs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_csv(args.train_path)
    df = normalize_columns(raw_df)
    df = sort_by_datetime_if_possible(df)

    df, cleaning_report = clean_target_outliers(
        df,
        mode=args.outlier_mode,
        tolerance_mw=args.tolerance_mw,
        sort_by_time=False,
    )

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    train_part, valid_part = split_chronological(df, args.valid_size)
    y_valid = pd.to_numeric(valid_part[TARGET_COL], errors="coerce")

    def objective(trial: optuna.Trial) -> float:
        feature_flags = suggest_feature_flags(trial, tune_core_features=args.tune_core_features)
        model_specs = build_model_specs(
            trial=trial,
            max_models=args.max_models,
            seed=args.seed,
            feature_flags=feature_flags,
        )

        meta_alpha = trial.suggest_float("meta_alpha", 0.01, 20.0, log=True)
        prediction_mode = trial.suggest_categorical("prediction_mode", ["fold_mean", "full"])

        ensemble = TimeSeriesStackingEnsemble(
            model_specs=model_specs,
            n_splits=args.n_splits,
            meta_alpha=meta_alpha,
            random_seed=args.seed,
            prediction_mode=prediction_mode,
        )

        ensemble.fit(train_part)
        pred = ensemble.predict(valid_part)
        pred = clip_predictions_to_available_capacity(pred, valid_part)

        score = competition_error_percent(y_valid, pred)
        mae_mw = float(np.mean(np.abs(y_valid.to_numpy() - pred)))

        trial.set_user_attr("feature_flags", feature_flags)
        trial.set_user_attr("model_specs", model_specs)
        trial.set_user_attr("mae_mw", mae_mw)
        trial.set_user_attr("oof_mae_mw", ensemble.oof_score_)

        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name="wind_stacked_ensemble",
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    best_config = {
        "model_specs": best.user_attrs["model_specs"],
        "n_splits": args.n_splits,
        "meta_alpha": best.params["meta_alpha"],
        "random_seed": args.seed,
        "clip_predictions": True,
        "prediction_mode": best.params["prediction_mode"],
        "feature_flags": best.user_attrs["feature_flags"],
        "best_score_percent": study.best_value,
        "best_mae_mw": best.user_attrs.get("mae_mw"),
        "best_oof_mae_mw": best.user_attrs.get("oof_mae_mw"),
        "outlier_mode": args.outlier_mode,
        "cleaning_report": cleaning_report.to_dict(),
    }

    with open(output_dir / "best_stacked_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, ensure_ascii=False, indent=2)

    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)

    print(f"Best score percent: {study.best_value:.5f}")
    print(f"Best MAE MW: {best_config['best_mae_mw']:.5f}")
    print(f"Saved config: {output_dir / 'best_stacked_config.json'}")
    print(f"Saved trials: {output_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()
