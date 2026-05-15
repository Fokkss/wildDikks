"""
Create a cleaned train csv.

Examples:
    python -m new_model.cut_outliers \
      --train_path data/train_dataset.csv \
      --output_path data/train_dataset_clean.csv \
      --mode clip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from new_model.config import read_csv, normalize_columns, sort_by_datetime_if_possible
from new_model.outlier_cleaning import clean_target_outliers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean target outliers.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mode", choices=["clip", "drop", "none"], default="clip")
    parser.add_argument("--tolerance_mw", type=float, default=0.5)
    parser.add_argument("--report_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = read_csv(args.train_path)
    df = normalize_columns(df)
    df = sort_by_datetime_if_possible(df)

    cleaned, report = clean_target_outliers(
        df=df,
        mode=args.mode,
        tolerance_mw=args.tolerance_mw,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    report_path = Path(args.report_path) if args.report_path else output_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved cleaned file: {output_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
