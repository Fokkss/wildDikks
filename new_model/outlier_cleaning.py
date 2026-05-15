from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from new_model.config import (
    TARGET_COL,
    available_capacity_from_df,
    normalize_columns,
    sort_by_datetime_if_possible,
)


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

    def to_dict(self) -> dict:
        return asdict(self)


def clean_target_outliers(
    df: pd.DataFrame,
    mode: str = "clip",
    tolerance_mw: float = 0.5,
    normalize: bool = True,
    sort_by_time: bool = True,
) -> tuple[pd.DataFrame, OutlierCleaningReport]:
    """
    Clean only train target outliers.

    mode:
        none - only normalize/sort and return data unchanged;
        clip - clip target to [0, available_capacity_mw];
        drop - drop rows outside physical range.
    """
    if mode not in {"none", "clip", "drop"}:
        raise ValueError("mode must be one of: none, clip, drop")

    out = df.copy()
    if normalize:
        out = normalize_columns(out)
    if sort_by_time:
        out = sort_by_datetime_if_possible(out)

    if TARGET_COL not in out.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found.")

    y = pd.to_numeric(out[TARGET_COL], errors="coerce")
    capacity = available_capacity_from_df(out)

    missing_target_mask = y.isna()
    below_zero_mask = y < -tolerance_mw
    above_capacity_mask = y > capacity + tolerance_mw
    physical_outlier_mask = missing_target_mask | below_zero_mask | above_capacity_mask

    original_rows = len(out)

    if mode == "none":
        cleaned = out.copy()
    elif mode == "drop":
        cleaned = out.loc[~physical_outlier_mask].copy()
    else:
        cleaned = out.copy()
        cleaned[TARGET_COL] = y.clip(lower=0.0, upper=capacity)
        cleaned = cleaned.loc[~missing_target_mask].copy()

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
