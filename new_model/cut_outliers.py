"""
Clean target outliers in train dataset.

Examples:

Physical clipping:
    python -m new_model.cut_outliers \
      --train_path data/train_dataset.csv \
      --output_path data/train_dataset_clean.csv \
      --mode clip

Drop physically impossible rows:
    python -m new_model.cut_outliers \
      --train_path data/train_dataset.csv \
      --output_path data/train_dataset_clean.csv \
      --mode drop
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
    available_capacity_from_df,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean train target outliers.")

    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_path", required=True)

    parser.add_argument(
        "--mode",
        choices=["clip", "drop"],
        default="clip",
        help="clip: clip target to physical bounds; drop: remove impossible rows.",
    )

    parser.add_argument(
        "--tolerance_mw",
        type=float,
        default=0.5,
        help="Allowed tolerance beyond physical capacity before row is considered outlier.",
    )

    parser.add_argument(
        "--report_path",
        default=None,
        help="Optional JSON report path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = read_csv(args.train_path)
    df = normalize_columns(df)
    df = sort_by_datetime_if_possible(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    capacity = available_capacity_from_df(df)

    missing_target_mask = y.isna()

    below_zero_mask = y < -args.tolerance_mw
    above_capacity_mask = y > capacity + args.tolerance_mw

    physical_outlier_mask = (
        missing_target_mask
        | below_zero_mask
        | above_capacity_mask
    )

    original_rows = len(df)

    if args.mode == "drop":
        cleaned = df.loc[~physical_outlier_mask].copy()
    else:
        cleaned = df.copy()
        clipped_y = y.clip(lower=0.0, upper=capacity)
        cleaned[TARGET_COL] = clipped_y
        cleaned = cleaned.loc[~missing_target_mask].copy()

    cleaned = cleaned.reset_index(drop=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    report = {
        "mode": args.mode,
        "input_path": args.train_path,
        "output_path": args.output_path,
        "original_rows": int(original_rows),
        "cleaned_rows": int(len(cleaned)),
        "removed_rows": int(original_rows - len(cleaned)),
        "missing_target_rows": int(missing_target_mask.sum()),
        "below_zero_rows": int(below_zero_mask.sum()),
        "above_capacity_rows": int(above_capacity_mask.sum()),
        "tolerance_mw": args.tolerance_mw,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    report_path = Path(args.report_path) if args.report_path else output_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved cleaned train: {output_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()