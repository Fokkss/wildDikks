"""
python -m new_model.predict \
  --features_path data/valid_features.csv \
  --model_path artifacts/ensemble.pkl \
  --output_path submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from new_model.config import (
    read_csv,
    normalize_columns,
    clip_predictions_to_available_capacity,
)

from models import WeightedEnsembleModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission.csv.")

    parser.add_argument(
        "--features_path",
        required=True,
        help="Path to valid_features.csv",
    )

    parser.add_argument(
        "--model_path",
        default="artifacts/ensemble.pkl",
        help="Path to saved ensemble.pkl",
    )

    parser.add_argument(
        "--output_path",
        default="submission.csv",
        help="Path to output CSV file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/4] Loading features...")
    raw_features = read_csv(args.features_path)

    print("[2/4] Normalizing columns...")
    features = normalize_columns(raw_features)

    print("[3/4] Loading model...")
    model = WeightedEnsembleModel.load(args.model_path)

    print("[4/4] Predicting...")
    predictions = model.predict(features)
    predictions = clip_predictions_to_available_capacity(predictions, features)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ВАЖНО:
    # Формат платформы: один столбец, без индекса, без заголовка.
    pd.DataFrame(predictions).to_csv(
        output_path,
        index=False,
    )

    print(f"Saved {len(predictions)} predictions to: {output_path}")


if __name__ == "__main__":
    main()