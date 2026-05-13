"""Generate predictions for valid_features.csv.

Example:
    python predict.py \
      --features_path ../data/valid_features.csv \
      --model_dir output/artifacts \
      --output_path submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from new_model.feature_engineering import (
    FARM_CAPACITY_MW,
    available_capacity_from_raw,
    make_features,
    sort_by_time_if_possible,
)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", required=True, help="Path to valid_features.csv")
    parser.add_argument("--model_dir", default="artifacts", help="Directory with wind_xgb_model.joblib")
    parser.add_argument("--output_path", default="submission.csv")
    parser.add_argument("--prediction_col", default="prediction", help="Name of the single prediction column")
    args = parser.parse_args()

    artifact_path = Path(args.model_dir) / "wind_xgb_model.joblib"
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    imputer = artifact["imputer"]
    feature_cols = artifact["feature_cols"]
    capacity_mw = float(artifact.get("capacity_mw", FARM_CAPACITY_MW))

    raw = load_csv(args.features_path)
    # Keep original row order for output. Feature lags are computed on chronological order only if the
    # input itself is already chronological; most competition valid files are.
    fe = make_features(raw, target_col=None)

    # Align with training columns. Missing engineered features are filled with NaN and then imputed.
    for col in feature_cols:
        if col not in fe.columns:
            fe[col] = np.nan
    X = fe[feature_cols]
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_cols)

    pred = model.predict(X_imp)
    row_cap = available_capacity_from_raw(raw).fillna(capacity_mw).to_numpy()
    row_cap = np.minimum(row_cap, capacity_mw)
    pred = np.clip(pred, 0.0, row_cap)

    out = pd.DataFrame({args.prediction_col: pred})
    out.to_csv(args.output_path, index=False)
    print(f"Saved {len(out)} predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
