"""
Predict with N-model stacked ensemble.

Example:
    python -m new_model.predict_stacked_ensemble \
      --features_path data/valid_features.csv \
      --model_path artifacts/stacked_ensemble.pkl \
      --output_path submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from new_model.config import read_csv, normalize_columns, clip_predictions_to_available_capacity
from models.stacked_ensemble import TimeSeriesStackingEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with stacked wind ensemble.")
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--model_path", default="artifacts/stacked_ensemble.pkl")
    parser.add_argument("--output_path", default="submission.csv")
    parser.add_argument(
        "--header",
        action="store_true",
        help="Write a CSV header named prediction. Default: no header.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/4] Loading features...")
    raw = read_csv(args.features_path)
    features = normalize_columns(raw)

    print("[2/4] Loading stacked model...")
    model = TimeSeriesStackingEnsemble.load(args.model_path)

    print("[3/4] Predicting...")
    pred = model.predict(features)
    pred = clip_predictions_to_available_capacity(pred, features)

    print("[4/4] Saving submission...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.header:
        pd.DataFrame({"prediction": pred}).to_csv(output_path, index=False)
    else:
        pd.DataFrame(pred).to_csv(output_path, index=False)

    print(f"Saved {len(pred)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
