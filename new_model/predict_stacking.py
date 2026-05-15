from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from new_model.config import read_csv, normalize_columns, clip_predictions_to_available_capacity
from models.stacking_ensemble import TimeSeriesStackingEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission using stacking ensemble.")
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--model_path", default="artifacts/stacking_ensemble.pkl")
    parser.add_argument("--output_path", default="submission.csv")
    parser.add_argument("--header", action="store_true", help="Write header row 'prediction'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/4] Loading features...")
    raw_features = read_csv(args.features_path)
    print(f"Rows in valid_features: {len(raw_features)}")

    print("[2/4] Normalizing columns...")
    features = normalize_columns(raw_features)

    print("[3/4] Loading stacking ensemble...")
    model = TimeSeriesStackingEnsemble.load(args.model_path)

    print("[4/4] Predicting...")
    predictions = model.predict(features)
    predictions = clip_predictions_to_available_capacity(predictions, features)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.header:
        pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)
    else:
        pd.DataFrame(predictions).to_csv(output_path, index=False, header=False)

    print(f"Saved {len(predictions)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
