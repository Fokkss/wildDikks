from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from new_model.config import TARGET_COL, available_capacity_from_df, normalize_columns, sort_by_datetime_if_possible


@dataclass
class OutlierCleaningReport:
    mode: str
    original_rows: int
    cleaned_rows: int
    removed_rows: int
    missing_target_rows: int
    below_zero_rows: int
    above_capacity_rows: int
    tolerance_mw: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "mode": self.mode,
            "original_rows": self.original_rows,
            "cleaned_rows": self.cleaned_rows,
            "removed_rows": self.removed_rows,
            "missing_target_rows": self.missing_target_rows,
            "below_zero_rows": self.below_zero_rows,
            "above_capacity_rows": self.above_capacity_rows,
            "tolerance_mw": self.tolerance_mw,
        }


def clean_target_outliers(
    df: pd.DataFrame,
    mode: str = "clip",
    tolerance_mw: float = 0.5,
    sort_by_time: bool = True,
) -> tuple[pd.DataFrame, OutlierCleaningReport]:
    """
    Clean target outliers using physical limits.

    mode='clip':
        - drop rows with missing target;
        - clip target to [0, available_capacity_mw].

    mode='drop':
        - drop rows with missing target;
        - drop rows below 0 or above available capacity, with tolerance.

    Do not apply this to valid_features.csv because it has no target.
    """
    if mode not in {"clip", "drop", "none"}:
        raise ValueError("mode must be one of: clip, drop, none")

    out = normalize_columns(df)
    if sort_by_time:
        out = sort_by_datetime_if_possible(out)

    if TARGET_COL not in out.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    y = pd.to_numeric(out[TARGET_COL], errors="coerce")
    capacity = available_capacity_from_df(out)

    missing_target_mask = y.isna()
    below_zero_mask = y < -tolerance_mw
    above_capacity_mask = y > capacity + tolerance_mw
    impossible_mask = missing_target_mask | below_zero_mask | above_capacity_mask

    original_rows = len(out)

    if mode == "none":
        cleaned = out.loc[~missing_target_mask].copy()
    elif mode == "drop":
        cleaned = out.loc[~impossible_mask].copy()
    else:
        cleaned = out.loc[~missing_target_mask].copy()
        capacity_clean = capacity.loc[cleaned.index]
        y_clean = pd.to_numeric(cleaned[TARGET_COL], errors="coerce")
        cleaned[TARGET_COL] = y_clean.clip(lower=0.0, upper=capacity_clean)

    cleaned = cleaned.reset_index(drop=True)

    report = OutlierCleaningReport(
        mode=mode,
        original_rows=int(original_rows),
        cleaned_rows=int(len(cleaned)),
        removed_rows=int(original_rows - len(cleaned)),
        missing_target_rows=int(missing_target_mask.sum()),
        below_zero_rows=int(below_zero_mask.sum()),
        above_capacity_rows=int(above_capacity_mask.sum()),
        tolerance_mw=float(tolerance_mw),
    )

    return cleaned, report
