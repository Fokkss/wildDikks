"""
Optuna search for feature groups and model params.

Example:
    python -m new_model.tune_features_optuna \
      --train_path data/train_dataset.csv \
      --output_dir artifacts/optuna \
      --n_trials 30 \
      --valid_size 0.2 \
      --model xgb
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

from models.base import ModelPreprocessor, DEFAULT_FEATURE_FLAGS
from models.xgboost_model import XGBoostWindModel
from models.catboost_model import CatBoostWindModel
from models.ensemble import WeightedEnsembleModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune feature groups with Optuna.")

    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_dir", default="artifacts/optuna")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=["xgb", "ensemble"],
        default="xgb",
        help="Use xgb for faster feature search, ensemble for final heavier tuning.",
    )

    return parser.parse_args()


def split_chronological(
    df: pd.DataFrame,
    valid_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1.0 - valid_size))

    train_part = df.iloc[:split_idx].reset_index(drop=True)
    valid_part = df.iloc[split_idx:].reset_index(drop=True)

    return train_part, valid_part


def suggest_feature_flags(trial: optuna.Trial) -> dict[str, bool]:
    """
    Tune feature groups, not every single column.

    This is more stable and faster than brute-forcing all columns.
    """
    flags = {}

    for feature_group in DEFAULT_FEATURE_FLAGS:
        flags[feature_group] = trial.suggest_categorical(
            f"use_{feature_group}",
            [True, False],
        )

    # I recommend forcing time features on.
    # If you want Optuna to decide, remove this line.
    flags["time"] = True

    return flags


def suggest_xgb_params(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    return {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": trial.suggest_int("xgb_n_estimators", 800, 3500),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.05, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 6),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 4, 20),
        "subsample": trial.suggest_float("xgb_subsample", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.55, 0.95),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1.0, 15.0, log=True),
        "tree_method": "hist",
        "max_bin": trial.suggest_categorical("xgb_max_bin", [128, 256, 512]),
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": 150,
    }


def suggest_cat_params(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    return {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": trial.suggest_int("cat_iterations", 600, 2000),
        "learning_rate": trial.suggest_float("cat_learning_rate", 0.01, 0.06, log=True),
        "depth": trial.suggest_int("cat_depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1.0, 15.0, log=True),
        "random_strength": trial.suggest_float("cat_random_strength", 0.1, 5.0),
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 150,
    }


def build_model(
    trial: optuna.Trial,
    model_type: str,
    feature_flags: dict[str, bool],
    seed: int,
):
    xgb_preprocessor = ModelPreprocessor(feature_flags=feature_flags.copy())

    xgb_model = XGBoostWindModel(
        preprocessor=xgb_preprocessor,
        random_seed=seed,
        params=suggest_xgb_params(trial, seed),
    )

    if model_type == "xgb":
        return xgb_model

    cat_preprocessor = ModelPreprocessor(feature_flags=feature_flags.copy())

    cat_model = CatBoostWindModel(
        preprocessor=cat_preprocessor,
        random_seed=seed,
        params=suggest_cat_params(trial, seed),
    )

    cat_weight = trial.suggest_float("catboost_weight", 0.2, 0.8)
    xgb_weight = 1.0 - cat_weight

    return WeightedEnsembleModel(
        catboost_model=cat_model,
        xgboost_model=xgb_model,
        catboost_weight=cat_weight,
        xgboost_weight=xgb_weight,
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv(args.train_path)
    df = normalize_columns(df)
    df = sort_by_datetime_if_possible(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    train_part, valid_part = split_chronological(df, args.valid_size)

    y_valid = pd.to_numeric(valid_part[TARGET_COL], errors="coerce")

    def objective(trial: optuna.Trial) -> float:
        feature_flags = suggest_feature_flags(trial)

        model = build_model(
            trial=trial,
            model_type=args.model,
            feature_flags=feature_flags,
            seed=args.seed,
        )

        model.fit(train_df=train_part, valid_df=valid_part)

        pred = model.predict(valid_part)
        pred = clip_predictions_to_available_capacity(pred, valid_part)

        score = competition_error_percent(y_valid, pred)

        trial.set_user_attr("feature_flags", feature_flags)

        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="wind_feature_search",
    )

    study.optimize(objective, n_trials=args.n_trials)

    print("Best score:", study.best_value)
    print("Best params:", study.best_params)

    best_feature_flags = study.best_trial.user_attrs["feature_flags"]

    with open(output_dir / "best_feature_flags.json", "w", encoding="utf-8") as f:
        json.dump(best_feature_flags, f, ensure_ascii=False, indent=2)

    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

    study.trials_dataframe().to_csv(
        output_dir / "trials.csv",
        index=False,
    )

    print(f"Saved: {output_dir / 'best_feature_flags.json'}")
    print(f"Saved: {output_dir / 'best_params.json'}")
    print(f"Saved: {output_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()