from __future__ import annotations

"""
Create a compact blend grid around two or more already-generated submissions.

Example:
    cd model
    python -m new_model_v1_0.blend_grid_v1_2 \
      --base submissions_v1_1/submission_v1_1.csv \
      --other submissions_v1_2/v1_2_bias0p92_density.csv \
      --output_dir submissions_v1_2_blends \
      --prefix old849_vs_v12_bias092 \
      --weights 0.95,0.90,0.85,0.80,0.75 \
      --header
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_pred(path: str | Path, *, has_header: bool | None = None) -> np.ndarray:
    path = Path(path)
    if has_header is True:
        df = pd.read_csv(path)
    elif has_header is False:
        df = pd.read_csv(path, header=None)
    else:
        # Try normal read first; if the first row is numeric-looking header, fallback.
        df = pd.read_csv(path)
        if len(df) == 0 or df.shape[1] != 1:
            df = pd.read_csv(path, header=None)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)


def write_pred(path: Path, pred: np.ndarray, *, header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if header:
        pd.DataFrame({"prediction": pred}).to_csv(path, index=False)
    else:
        pd.DataFrame(pred).to_csv(path, index=False, header=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Best/stable submission, usually current v1_1")
    p.add_argument("--other", required=True, help="New candidate submission to blend into base")
    p.add_argument("--output_dir", default="submissions_v1_2_blends")
    p.add_argument("--prefix", default="blend")
    p.add_argument("--weights", default="0.95,0.90,0.85,0.80,0.75,0.70", help="Base weights")
    p.add_argument("--clip_min", type=float, default=0.0)
    p.add_argument("--clip_max", type=float, default=90.09)
    p.add_argument("--header", action="store_true", default=True)
    p.add_argument("--no_header", dest="header", action="store_false")
    args = p.parse_args()

    base = read_pred(args.base)
    other = read_pred(args.other)

    if len(base) != len(other):
        raise SystemExit(f"Length mismatch: base={len(base)}, other={len(other)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    submit_list = []

    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    for w_base in weights:
        w_other = 1.0 - w_base
        pred = w_base * base + w_other * other
        pred = np.clip(pred, args.clip_min, args.clip_max)

        tag = f"{args.prefix}_base{int(round(w_base * 100)):03d}_other{int(round(w_other * 100)):03d}.csv"
        out_path = out_dir / tag
        write_pred(out_path, pred, header=args.header)
        submit_list.append(str(out_path))

        report.append(
            {
                "file": str(out_path),
                "base": str(args.base),
                "other": str(args.other),
                "w_base": w_base,
                "w_other": w_other,
                "n": int(len(pred)),
                "mean": float(np.mean(pred)),
                "std": float(np.std(pred)),
                "min": float(np.min(pred)),
                "max": float(np.max(pred)),
                "mae_to_base": float(np.mean(np.abs(pred - base))),
                "mae_to_other": float(np.mean(np.abs(pred - other))),
                "corr_base_other": float(np.corrcoef(base, other)[0, 1]) if len(base) > 1 else float("nan"),
            }
        )

    (out_dir / f"{args.prefix}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{args.prefix}_SUBMIT.txt").write_text("\n".join(submit_list) + "\n", encoding="utf-8")

    print(f"Generated {len(submit_list)} blends in {out_dir}")
    print(f"Submit list: {out_dir / (args.prefix + '_SUBMIT.txt')}")


if __name__ == "__main__":
    main()
