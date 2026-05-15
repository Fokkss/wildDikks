from __future__ import annotations

import argparse
import json
from pathlib import Path

from new_model.config import read_csv
from new_model.outlier_cleaning import clean_target_outliers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cleaned train dataset.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mode", choices=["none", "clip", "drop"], default="clip")
    parser.add_argument("--tolerance_mw", type=float, default=0.5)
    parser.add_argument("--report_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = read_csv(args.train_path)
    cleaned, report = clean_target_outliers(
        df,
        mode=args.mode,
        tolerance_mw=args.tolerance_mw,
        normalize=True,
        sort_by_time=True,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    report_path = Path(args.report_path) if args.report_path else output_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"Saved cleaned train: {output_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
