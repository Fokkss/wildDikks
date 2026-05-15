from __future__ import annotations

import numpy as np
import pandas as pd

from new_model.config import TARGET_COL, available_capacity_from_df


def clean_target_outliers(
    df: pd.DataFrame,
    mode: str = "clip",
    tolerance_mw: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """
    Clean physically impossible target values.

    mode="clip":
        - remove rows with missing target;
        - clip target to [0, available_capacity_mw].

    mode="drop":
        - remove missing target rows;
        - remove rows below 0 - tolerance;
        - remove rows above available_capacity_mw + tolerance.

    mode="none":
        - remove only missing target rows.
    """
    if mode not in {"clip", "drop", "none"}:
        raise ValueError("mode must be one of: clip, drop, none")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    out = df.copy()
    y = pd.to_numeric(out[TARGET_COL], errors="coerce")
    cap = available_capacity_from_df(out)

    missing_mask = y.isna()
    below_zero_mask = y < -tolerance_mw
    above_capacity_mask = y > cap + tolerance_mw

    original_rows = len(out)

    if mode == "drop":
        remove_mask = missing_mask | below_zero_mask | above_capacity_mask
        out = out.loc[~remove_mask].copy()
    elif mode == "clip":
        out = out.loc[~missing_mask].copy()
        cap_kept = cap.loc[~missing_mask]
        y_kept = y.loc[~missing_mask]
        out[TARGET_COL] = y_kept.clip(lower=0.0, upper=cap_kept)
    else:
        out = out.loc[~missing_mask].copy()

    out = out.reset_index(drop=True)

    report = {
        "mode": mode,
        "original_rows": int(original_rows),
        "cleaned_rows": int(len(out)),
        "removed_rows": int(original_rows - len(out)),
        "missing_target_rows": int(missing_mask.sum()),
        "below_zero_rows": int(below_zero_mask.sum()),
        "above_capacity_rows": int(above_capacity_mask.sum()),
        "tolerance_mw": float(tolerance_mw),
    }

    return out, report
