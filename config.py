from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Station constants
# =========================

FARM_CAPACITY_MW = 90.09
N_TURBINES = 26
TURBINE_CAPACITY_MW = FARM_CAPACITY_MW / N_TURBINES
HUB_HEIGHT_M = 80

TARGET_COL = "Выработка"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"


# =========================
# Column normalization
# =========================

TARGET_ALIASES = {
    "Выработка",
    "Выработка.Результирующий расчет",
    "Выработка. Результирующий расчет",
    "Выработка Результирующий расчет",
    "target",
}

REPAIR_ALIASES = {
    "Кол-во_ВЭУ_в_ремонте",
    "Количество_ВЭУ_в_ремонте",
    "РљРѕР»-РІРѕ_Р’Р­РЈ_РІ_СЂРµРјРѕРЅС‚Рµ",
    "repair_count",
}


def _compact_name(value: str) -> str:
    """
    Небольшая нормализация названий колонок.
    Не regex. Просто защита от пробелов, BOM и разных вариантов точки.
    """

    return (
        str(value)
        .replace("\ufeff", "")
        .replace("\u00a0", " ")
        .strip()
        .replace(" ", "")
        .lower()
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит критичные колонки к стабильным именам:
    - target -> Выработка
    - repair_count -> repair_count

    Важно:
    остальные погодные колонки не переименовываем, чтобы не плодить магию.
    models/base.py сам заберёт числовые признаки.
    """

    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]

    target_aliases_compact = {_compact_name(c) for c in TARGET_ALIASES}
    repair_aliases_compact = {_compact_name(c) for c in REPAIR_ALIASES}

    rename_map: dict[str, str] = {}

    for col in out.columns:
        compact = _compact_name(col)

        if compact in target_aliases_compact:
            rename_map[col] = TARGET_COL

        if compact in repair_aliases_compact:
            rename_map[col] = "repair_count"

    return out.rename(columns=rename_map)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def sort_by_datetime_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для обучения сортируем хронологически.
    Для predict НЕ используем, чтобы порядок строк submission совпал с valid_features.csv.
    """

    if DATETIME_COL not in df.columns:
        return df.reset_index(drop=True)

    out = df.copy()
    out["__dt_sort"] = pd.to_datetime(out[DATETIME_COL], errors="coerce")

    return (
        out
        .sort_values("__dt_sort", kind="mergesort")
        .drop(columns=["__dt_sort"])
        .reset_index(drop=True)
    )


def available_capacity_from_df(df: pd.DataFrame) -> pd.Series:
    """
    Физический потолок по строке:
    если часть турбин в ремонте, максимум выработки меньше 90.09 МВт.
    """

    normalized = normalize_columns(df)

    if "repair_count" in normalized.columns:
        repair = (
            pd.to_numeric(normalized["repair_count"], errors="coerce")
            .fillna(0.0)
            .clip(0, N_TURBINES)
        )
    else:
        repair = pd.Series(0.0, index=normalized.index)

    capacity = (N_TURBINES - repair) * TURBINE_CAPACITY_MW

    return capacity.clip(lower=0.0, upper=FARM_CAPACITY_MW)


def clip_predictions_to_available_capacity(
    predictions: np.ndarray,
    features_df: pd.DataFrame,
) -> np.ndarray:
    """
    Клип предсказаний:
    - не ниже 0;
    - не выше доступной мощности станции в конкретный час.
    """

    pred = np.asarray(predictions, dtype=float)
    row_capacity = available_capacity_from_df(features_df).to_numpy()

    return np.clip(pred, 0.0, row_capacity)


def competition_error_percent(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """
    Метрика хакатона:
    mean(abs(fact - forecast)) / Nуст * 100%.
    Nуст = 90.09 МВт.
    """

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    return float(np.mean(np.abs(y_true_arr - y_pred_arr)) / FARM_CAPACITY_MW * 100.0)