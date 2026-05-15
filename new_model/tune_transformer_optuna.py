"""
Optuna tuning for TransformerWindModel.

Fast smoke test:
    python -m new_model.tune_transformer_optuna \
      --train_path data/train_dataset.csv \
      --output_dir artifacts/transformer_optuna \
      --n_trials 5 \
      --outlier_mode clip

Real run:
    python -m new_model.tune_transformer_optuna \
      --train_path data/train_dataset.csv \
      --output_dir artifacts/transformer_optuna \
      --n_trials 30 \
      --outlier_mode clip
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
from models.transformer_model import TransformerEnsembleModel, TransformerWindModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Transformer model with Optuna.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_dir", default="artifacts/transformer_optuna")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outlier_mode", choices=["clip", "drop", "none"], default="clip")
    parser.add_argument("--tolerance_mw", type=float, default=0.5)
    parser.add_argument(
        "--tune_ensemble_size",
        action="store_true",
        help="Also tune n_models. Otherwise Optuna tunes one base Transformer config.",
    )
    return parser.parse_args()


def split_chronological(df: pd.DataFrame, valid_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1.0 - valid_size))
    return (
        df.iloc[:split_idx].reset_index(drop=True),
        df.iloc[split_idx:].reset_index(drop=True),
    )


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])

    possible_heads = [h for h in [2, 4, 8] if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", possible_heads)

    return {
        "sequence_length": trial.suggest_categorical("sequence_length", [12, 24, 48, 72]),
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "epochs": trial.suggest_int("epochs", 15, 60),
        "patience": trial.suggest_int("patience", 4, 10),
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = read_csv(args.train_path)
    df = normalize_columns(df)
    df = sort_by_datetime_if_possible(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    print("Cleaning outliers...")
    df, outlier_report = clean_target_outliers(
        df=df,
        mode=args.outlier_mode,
        tolerance_mw=args.tolerance_mw,
    )
    print(json.dumps(outlier_report, ensure_ascii=False, indent=2))

    train_part, valid_part = split_chronological(df, args.valid_size)
    y_valid = pd.to_numeric(valid_part[TARGET_COL], errors="coerce").to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        if args.tune_ensemble_size:
            n_models = trial.suggest_int("n_models", 1, 3)
            model = TransformerEnsembleModel.from_config(
                n_models=n_models,
                seed=args.seed,
                params=params,
            )
        else:
            n_models = 1
            model = TransformerWindModel(
                random_seed=args.seed,
                params=params,
            )

        model.fit(train_df=train_part, valid_df=valid_part)

        pred = model.predict(valid_part)
        pred = clip_predictions_to_available_capacity(pred, valid_part)
        score = competition_error_percent(y_valid, pred)

        trial.set_user_attr("params", params)
        trial.set_user_attr("n_models", n_models)

        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="transformer_wind_tuning",
    )
    study.optimize(objective, n_trials=args.n_trials)

    best_params = study.best_trial.user_attrs["params"]
    best_n_models = study.best_trial.user_attrs["n_models"]

    best_config = {
        "n_models": int(best_n_models),
        "params": best_params,
        "best_score": float(study.best_value),
        "outlier_report": outlier_report,
    }

    with open(output_dir / "best_transformer_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, ensure_ascii=False, indent=2)

    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)

    print("Best score:", study.best_value)
    print("Best config:")
    print(json.dumps(best_config, ensure_ascii=False, indent=2))
    print(f"Saved: {output_dir / 'best_transformer_config.json'}")
    print(f"Saved: {output_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()
