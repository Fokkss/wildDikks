from __future__ import annotations

"""
Simple blend grid for already generated submissions.

Run from model/:
    python -m new_model_v1_0.blend_grid_v1_3 \
      --base ../submission_v1_1.csv \
      --other submissions_v1_3/v1_3_caprelax_uncap10.csv \
      --output_dir submissions_v1_3_blends \
      --prefix v11_caprelax10 \
      --weights 0.95,0.90,0.85,0.80 \
      --header
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_pred(path: Path, *, has_header: bool | None) -> np.ndarray:
    if has_header is None:
        # Try headered first; fallback to headerless.
        df = pd.read_csv(path)
        if df.shape[1] == 1 and len(df) > 0:
            return pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        return pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=float)

    if has_header:
        return pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=float)


def write_pred(path: Path, pred: np.ndarray, *, header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 90.09)
    if header:
        pd.DataFrame({"prediction": pred}).to_csv(path, index=False)
    else:
        pd.DataFrame(pred).to_csv(path, index=False, header=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--output_dir", default="submissions_v1_3_blends")
    parser.add_argument("--prefix", default="blend")
    parser.add_argument("--weights", default="0.95,0.90,0.85,0.80,0.75")
    parser.add_argument("--base_has_header", action="store_true", default=None)
    parser.add_argument("--base_no_header", dest="base_has_header", action="store_false")
    parser.add_argument("--other_has_header", action="store_true", default=None)
    parser.add_argument("--other_no_header", dest="other_has_header", action="store_false")
    parser.add_argument("--header", action="store_true", default=True)
    parser.add_argument("--no_header", dest="header", action="store_false")
    args = parser.parse_args()

    base = read_pred(Path(args.base), has_header=args.base_has_header)
    other = read_pred(Path(args.other), has_header=args.other_has_header)

    if len(base) != len(other):
        raise ValueError(f"Length mismatch: base={len(base)}, other={len(other)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = []
    for w_base in [float(x) for x in args.weights.split(",") if x.strip()]:
        w_other = 1.0 - w_base
        pred = w_base * base + w_other * other
        tag = f"base{int(round(w_base*100)):03d}_other{int(round(w_other*100)):03d}"
        out_path = out_dir / f"{args.prefix}_{tag}.csv"
        write_pred(out_path, pred, header=args.header)
        made.append(str(out_path))

    (out_dir / f"{args.prefix}_SUBMIT_FIRST.txt").write_text("\n".join(made) + "\n", encoding="utf-8")

    for p in made:
        print(p)


if __name__ == "__main__":
    main()
