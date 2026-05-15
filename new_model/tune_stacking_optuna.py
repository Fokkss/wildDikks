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
from models.stacking_ensemble import TimeSeriesStackingEnsemble

try:
    import lightgbm  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune stacking ensemble with Optuna.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_dir", default="artifacts/stacking_optuna")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_models", type=int, default=5)
    parser.add_argument("--outlier_mode", choices=["none", "clip", "drop"], default="clip")
    parser.add_argument("--tolerance_mw", type=float, default=0.5)
    parser.add_argument("--include_lightgbm_meta", action="store_true")
    return parser.parse_args()


def split_chronological(df: pd.DataFrame, valid_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1.0 - valid_size))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def suggest_feature_flags(trial: optuna.Trial) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for name in DEFAULT_FEATURE_FLAGS:
        flags[name] = trial.suggest_categorical(f"use_{name}", [True, False])

    # These are almost always useful and cheap. Keep them on unless you want pure ablation.
    flags["time"] = True
    flags["availability"] = True
    flags["power_curve"] = True
    return flags


def suggest_xgb_params(trial: optuna.Trial, seed: int, model_idx: int) -> dict[str, Any]:
    return {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": trial.suggest_int("xgb_n_estimators", 800, 3000),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.05, log=True),
        "max_depth": trial.suggest_int(f"xgb_{model_idx}_max_depth", 3, 6),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 4, 20),
        "subsample": trial.suggest_float("xgb_subsample", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.55, 0.95),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1.0, 15.0, log=True),
        "tree_method": "hist",
        "max_bin": trial.suggest_categorical("xgb_max_bin", [128, 256, 512]),
        "random_state": seed + model_idx,
        "n_jobs": -1,
        "early_stopping_rounds": 150,
    }


def suggest_cat_params(trial: optuna.Trial, seed: int, model_idx: int) -> dict[str, Any]:
    return {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": trial.suggest_int("cat_iterations", 600, 1800),
        "learning_rate": trial.suggest_float("cat_learning_rate", 0.01, 0.06, log=True),
        "depth": trial.suggest_int(f"cat_{model_idx}_depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1.0, 15.0, log=True),
        "random_strength": trial.suggest_float("cat_random_strength", 0.1, 5.0),
        "random_seed": seed + 100 + model_idx,
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 150,
    }


def suggest_meta_config(trial: optuna.Trial, include_lightgbm_meta: bool) -> tuple[str, dict[str, Any]]:
    choices = ["ridge", "linear", "xgb"]
    if include_lightgbm_meta and LIGHTGBM_AVAILABLE:
        choices.append("lightgbm")

    meta_type = trial.suggest_categorical("meta_model_type", choices)

    if meta_type == "ridge":
        return meta_type, {"alpha": trial.suggest_float("meta_ridge_alpha", 1e-3, 100.0, log=True)}

    if meta_type == "linear":
        return meta_type, {}

    if meta_type == "xgb":
        return meta_type, {
            "n_estimators": trial.suggest_int("meta_xgb_n_estimators", 100, 700),
            "learning_rate": trial.suggest_float("meta_xgb_learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("meta_xgb_max_depth", 1, 3),
            "min_child_weight": trial.suggest_int("meta_xgb_min_child_weight", 3, 30),
            "reg_lambda": trial.suggest_float("meta_xgb_reg_lambda", 1.0, 20.0, log=True),
        }

    if meta_type == "lightgbm":
        return meta_type, {
            "n_estimators": trial.suggest_int("meta_lgb_n_estimators", 100, 700),
            "learning_rate": trial.suggest_float("meta_lgb_learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("meta_lgb_num_leaves", 4, 16),
            "min_child_samples": trial.suggest_int("meta_lgb_min_child_samples", 10, 80),
            "reg_lambda": trial.suggest_float("meta_lgb_reg_lambda", 1.0, 20.0, log=True),
        }

    raise ValueError(meta_type)


def build_trial_config(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    feature_flags = suggest_feature_flags(trial)

    max_models = max(1, args.max_models)
    n_xgb = trial.suggest_int("n_xgb", 2, max_models)
    max_cat = max(0, min(2, max_models - n_xgb))
    n_cat = trial.suggest_int("n_cat", 0, max_cat) if max_cat > 0 else 0

    specs: list[dict[str, Any]] = []
    for i in range(n_xgb):
        specs.append(
            {
                "name": f"xgb_{i}",
                "model_type": "xgb",
                "seed_offset": i,
                "params": suggest_xgb_params(trial, args.seed, i),
                "feature_flags": feature_flags,
            }
        )

    for i in range(n_cat):
        specs.append(
            {
                "name": f"cat_{i}",
                "model_type": "catboost",
                "seed_offset": 100 + i,
                "params": suggest_cat_params(trial, args.seed, i),
                "feature_flags": feature_flags,
            }
        )

    meta_type, meta_params = suggest_meta_config(trial, args.include_lightgbm_meta)

    return {
        "base_model_specs": specs,
        "meta_model_type": meta_type,
        "meta_model_params": meta_params,
        "n_splits": trial.suggest_int("n_splits", 3, 5),
        "random_seed": args.seed,
        "clip_predictions": True,
        "prediction_mode": trial.suggest_categorical("prediction_mode", ["full", "blend_full_fold"]),
        "append_meta_stats": trial.suggest_categorical("append_meta_stats", [True, False]),
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = read_csv(args.train_path)
    df = normalize_columns(df)
    df = sort_by_datetime_if_possible(df)

    print("Cleaning outliers...")
    df, report = clean_target_outliers(
        df,
        mode=args.outlier_mode,
        tolerance_mw=args.tolerance_mw,
        normalize=False,
        sort_by_time=False,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    train_part, valid_part = split_chronological(df, args.valid_size)
    y_valid = pd.to_numeric(valid_part[TARGET_COL], errors="coerce")

    def objective(trial: optuna.Trial) -> float:
        config = build_trial_config(trial, args)
        model = TimeSeriesStackingEnsemble.from_config(config)
        model.fit(train_part)

        pred = model.predict(valid_part)
        pred = clip_predictions_to_available_capacity(pred, valid_part)
        score = competition_error_percent(y_valid, pred)

        trial.set_user_attr("config", config)
        trial.set_user_attr("oof_mae_mw", model.oof_mae_mw_)
        trial.set_user_attr("n_base_models", len(config["base_model_specs"]))
        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="wind_stacking_tuning")
    study.optimize(objective, n_trials=args.n_trials)

    best_config = study.best_trial.user_attrs["config"]

    print("Best score:", study.best_value)
    print("Best params:", study.best_params)

    with open(output_dir / "best_stacking_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, ensure_ascii=False, indent=2)

    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

    with open(output_dir / "outlier_report.json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)

    print(f"Saved: {output_dir / 'best_stacking_config.json'}")
    print(f"Saved: {output_dir / 'best_params.json'}")
    print(f"Saved: {output_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()
