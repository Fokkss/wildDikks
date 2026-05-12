"""Feature engineering for hourly wind farm power forecasting.

The functions are intentionally schema-tolerant: they try to infer columns by
keywords and heights (10m, 80m, 120m, 180m), while preserving all numeric source
columns. If your competition data uses unusual names, add aliases in KIND_KEYWORDS.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

FARM_CAPACITY_MW = 90.09
N_TURBINES = 26
TURBINE_CAPACITY_MW = FARM_CAPACITY_MW / N_TURBINES
HUB_HEIGHT_M = 80
R_DRY_AIR = 287.05  # J/(kg*K)

# Extend these lists for your exact schema if needed.
KIND_KEYWORDS: dict[str, tuple[str, ...]] = {
    "wind_speed": ("wind_speed", "windspeed", "wind speed", "ws", "speed", "скорость", "ветер"),
    "wind_dir": ("wind_dir", "wind direction", "direction", "dir", "wd", "направ", "азимут", "румб"),
    "temp": ("temperature", "temp", "t2m", "температура", "темп"),
    "pressure": ("pressure", "press", "msl", "sp", "давление"),
    "precip": ("precip", "rain", "snow", "осад", "дожд", "снег"),
    "cloud": ("cloud", "облач"),
    "repair": ("repair", "ремонт", "веу", "turbines_off", "unavailable", "offline"),
}

DATE_KEYWORDS = ("datetime", "timestamp", "date", "time", "dt", "дата", "время")
TARGET_KEYWORDS = ("результирующий", "выработка", "generation", "power", "target", "fact", "actual")


def _norm(s: str) -> str:
    """Normalize a column name for fuzzy matching while keeping Cyrillic letters."""
    return re.sub(r"[^0-9a-zа-яё]+", "_", str(s).lower()).strip("_")


def find_datetime_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        n = _norm(col)
        if any(k in n for k in DATE_KEYWORDS):
            # Avoid treating already engineered hour/month columns as timestamps.
            if "hour" not in n and "month" not in n and "час" not in n and "месяц" not in n:
                return col
    return None


def find_target_col(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested and requested in df.columns:
        return requested
    if requested:
        req_norm = _norm(requested)
        for col in df.columns:
            if _norm(col) == req_norm:
                return col
    candidates: list[str] = []
    for col in df.columns:
        n = _norm(col)
        if any(k in n for k in TARGET_KEYWORDS):
            candidates.append(col)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer columns that look most like final energy/power calculation.
        for prefer in ("результирующий", "выработка", "generation", "power"):
            for col in candidates:
                if prefer in _norm(col):
                    return col
    raise ValueError(
        "Target column was not found. Pass it explicitly: "
        "python train.py --target 'Результирующий расчет'"
    )


def _has_height(col_norm: str, height_m: int) -> bool:
    # Works for names such as wind_speed_80m, speed_80_m, Скорость ветра на 80 м.
    return bool(re.search(rf"(^|_)({height_m})(m|м|meter|meters)?(_|$)", col_norm)) or str(height_m) in col_norm


def find_col(df: pd.DataFrame, kind: str, height_m: Optional[int] = None) -> Optional[str]:
    keywords = KIND_KEYWORDS[kind]
    scored: list[tuple[int, str]] = []
    for col in df.columns:
        n = _norm(col)
        if not any(k in n for k in keywords):
            continue
        if height_m is not None and not _has_height(n, height_m):
            continue
        score = 0
        # Prefer exact-ish kind matches and shorter names.
        if kind == "wind_speed" and ("speed" in n or "скорость" in n):
            score += 5
        if kind == "wind_dir" and ("dir" in n or "direction" in n or "направ" in n):
            score += 5
        if height_m is not None and _has_height(n, height_m):
            score += 10
        score -= len(n) // 20
        scored.append((score, col))
    if not scored:
        return None
    return sorted(scored, key=lambda x: (-x[0], _norm(x[1])))[0][1]


def find_cols(df: pd.DataFrame, kind: str) -> list[str]:
    keywords = KIND_KEYWORDS[kind]
    out = []
    for col in df.columns:
        n = _norm(col)
        if any(k in n for k in keywords):
            out.append(col)
    return out


def _safe_divide(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, default: float = np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.asarray(a, dtype=float) / np.asarray(b, dtype=float)
    x = np.where(np.isfinite(x), x, default)
    return x


def _to_numeric_series(df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    if col is None or col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _temperature_to_kelvin(temp: pd.Series) -> pd.Series:
    # Most weather datasets store temperature in Celsius. If median is already Kelvin, keep it.
    med = temp.dropna().median() if temp.notna().any() else np.nan
    if pd.notna(med) and med > 150:
        return temp
    return temp + 273.15


def _pressure_to_pa(pressure: pd.Series) -> pd.Series:
    # hPa/mbar are typically around 950-1050; Pa around 95000-105000.
    med = pressure.dropna().median() if pressure.notna().any() else np.nan
    if pd.notna(med) and med < 2000:
        return pressure * 100.0
    return pressure


def circular_diff_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    """Smallest signed difference a-b in degrees, in [-180, 180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt_col = find_datetime_col(out)
    dt = None
    if dt_col is not None:
        dt = pd.to_datetime(out[dt_col], errors="coerce")
        out["year"] = dt.dt.year
        out["month"] = dt.dt.month
        out["dayofyear"] = dt.dt.dayofyear
        out["dayofweek"] = dt.dt.dayofweek
        out["hour_of_day"] = dt.dt.hour
        out["is_weekend"] = (dt.dt.dayofweek >= 5).astype("float")

    # Use existing month/hour columns if no timestamp exists.
    month_col = next((c for c in out.columns if _norm(c) in {"month", "месяц"}), None)
    hour_col = next((c for c in out.columns if _norm(c) in {"hour", "hour_of_day", "час", "час_суток"}), None)

    month = pd.to_numeric(out[month_col], errors="coerce") if month_col else out.get("month")
    hour = pd.to_numeric(out[hour_col], errors="coerce") if hour_col else out.get("hour_of_day")
    dayofyear = out.get("dayofyear")

    if month is not None:
        out["month_sin"] = np.sin(2.0 * np.pi * pd.to_numeric(month, errors="coerce") / 12.0)
        out["month_cos"] = np.cos(2.0 * np.pi * pd.to_numeric(month, errors="coerce") / 12.0)
    if hour is not None:
        out["hour_sin"] = np.sin(2.0 * np.pi * pd.to_numeric(hour, errors="coerce") / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * pd.to_numeric(hour, errors="coerce") / 24.0)
    if dayofyear is not None:
        out["doy_sin"] = np.sin(2.0 * np.pi * pd.to_numeric(dayofyear, errors="coerce") / 365.25)
        out["doy_cos"] = np.cos(2.0 * np.pi * pd.to_numeric(dayofyear, errors="coerce") / 365.25)
    return out


def add_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    repair_col = find_col(out, "repair")
    if repair_col is not None:
        repair = pd.to_numeric(out[repair_col], errors="coerce").fillna(0.0).clip(0, N_TURBINES)
    else:
        repair = pd.Series(0.0, index=out.index)
    available_turbines = (N_TURBINES - repair).clip(0, N_TURBINES)
    out["repair_count_inferred"] = repair
    out["available_turbines"] = available_turbines
    out["availability_ratio"] = available_turbines / N_TURBINES
    out["available_capacity_mw"] = available_turbines * TURBINE_CAPACITY_MW
    return out


def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Speeds by height.
    speed: dict[int, Optional[pd.Series]] = {}
    for h in (10, 80, 120, 180):
        speed[h] = _to_numeric_series(out, find_col(out, "wind_speed", h))
        if speed[h] is not None:
            out[f"ws_{h}m"] = speed[h]
            out[f"ws_{h}m_sq"] = speed[h] ** 2
            out[f"ws_{h}m_cube"] = speed[h] ** 3

    # Hub-height speed. Prefer measured/forecast 80m; otherwise estimate from 10m and 120m by power law.
    if speed[80] is not None:
        v80 = speed[80]
        out["hub_ws_80m"] = v80
    elif speed[10] is not None and speed[120] is not None:
        alpha = pd.Series(
            _safe_divide(np.log((speed[120].clip(lower=0.05)) / (speed[10].clip(lower=0.05))), math.log(120.0 / 10.0)),
            index=out.index,
        ).clip(-0.5, 1.0)
        v80 = speed[10] * (80.0 / 10.0) ** alpha
        out["hub_ws_80m"] = v80
        out["hub_ws_80m_interpolated"] = 1.0
        out["shear_alpha_10_120"] = alpha
    else:
        v80 = out.get("hub_ws_80m")

    if v80 is not None:
        v80 = pd.to_numeric(v80, errors="coerce")
        out["hub_ws_80m_sq"] = v80 ** 2
        out["hub_ws_80m_cube"] = v80 ** 3
        out["hub_ws_cut_in_proxy"] = (v80 >= 3.0).astype("float")
        out["hub_ws_rated_proxy"] = (v80 >= 12.0).astype("float")
        out["hub_ws_cut_out_proxy"] = (v80 >= 25.0).astype("float")
        out["hub_ws_piece_3_12"] = (v80.clip(3.0, 12.0) - 3.0).clip(lower=0.0)
        out["hub_ws_piece_12_25"] = (v80.clip(12.0, 25.0) - 12.0).clip(lower=0.0)

    # Vertical wind shear / gradients.
    for h1, h2 in ((10, 80), (80, 120), (80, 180), (10, 120), (120, 180)):
        if speed[h1] is not None and speed[h2] is not None:
            s1 = speed[h1].clip(lower=0.05)
            s2 = speed[h2].clip(lower=0.05)
            out[f"ws_diff_{h2}_{h1}"] = speed[h2] - speed[h1]
            out[f"ws_ratio_{h2}_{h1}"] = pd.Series(_safe_divide(s2, s1), index=out.index).clip(0, 10)
            out[f"shear_alpha_{h1}_{h2}"] = pd.Series(
                _safe_divide(np.log(s2 / s1), math.log(h2 / h1)), index=out.index
            ).clip(-0.5, 1.0)

    # Direction as circular variables and directional interactions.
    direction: dict[int, Optional[pd.Series]] = {}
    for h in (10, 80, 120, 180):
        direction[h] = _to_numeric_series(out, find_col(out, "wind_dir", h))
        if direction[h] is not None:
            rad = np.deg2rad(direction[h] % 360.0)
            out[f"wd_{h}m_sin"] = np.sin(rad)
            out[f"wd_{h}m_cos"] = np.cos(rad)
            if h == 80 and v80 is not None:
                out["hub_ws_cube_x_wd_sin"] = (pd.to_numeric(v80, errors="coerce") ** 3) * out[f"wd_{h}m_sin"]
                out["hub_ws_cube_x_wd_cos"] = (pd.to_numeric(v80, errors="coerce") ** 3) * out[f"wd_{h}m_cos"]
    for h1, h2 in ((10, 80), (80, 120), (80, 180), (10, 120)):
        if direction[h1] is not None and direction[h2] is not None:
            out[f"wd_diff_{h2}_{h1}"] = circular_diff_deg(direction[h2], direction[h1])

    # Temporal dynamics from weather only. No target lags: valid_features does not contain target.
    if v80 is not None:
        v = pd.to_numeric(v80, errors="coerce")
        out["hub_ws_lag1"] = v.shift(1)
        out["hub_ws_diff1"] = v - v.shift(1)
        for w in (3, 6, 12):
            out[f"hub_ws_roll_mean_{w}"] = v.shift(1).rolling(w, min_periods=1).mean()
            out[f"hub_ws_roll_std_{w}"] = v.shift(1).rolling(w, min_periods=2).std()

    return out


def add_air_density_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    temp_col = find_col(out, "temp", 80) or find_col(out, "temp", 120) or find_col(out, "temp", 10) or find_col(out, "temp")
    pressure_col = find_col(out, "pressure", 80) or find_col(out, "pressure")
    temp = _to_numeric_series(out, temp_col)
    pressure = _to_numeric_series(out, pressure_col)

    if temp is not None:
        temp_k = _temperature_to_kelvin(temp)
        out["temp_k_inferred"] = temp_k
    else:
        temp_k = None
    if pressure is not None:
        pressure_pa = _pressure_to_pa(pressure)
        out["pressure_pa_inferred"] = pressure_pa
    else:
        pressure_pa = None

    if temp_k is not None and pressure_pa is not None:
        rho = pressure_pa / (R_DRY_AIR * temp_k)
        rho = rho.replace([np.inf, -np.inf], np.nan).clip(0.7, 1.6)
        out["air_density_kg_m3_proxy"] = rho
        if "hub_ws_80m_cube" in out.columns:
            out["wind_power_density_proxy"] = 0.5 * rho * out["hub_ws_80m_cube"]
            out["wind_power_density_x_availability"] = out["wind_power_density_proxy"] * out.get("availability_ratio", 1.0)
    elif temp_k is not None and pressure_pa is None:
        out["inverse_temp_k_proxy"] = 1.0 / temp_k
    return out


def add_weather_misc_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    precip_cols = find_cols(out, "precip")
    cloud_cols = find_cols(out, "cloud")
    if precip_cols:
        vals = out[precip_cols].apply(pd.to_numeric, errors="coerce")
        out["precip_sum_inferred"] = vals.sum(axis=1, min_count=1)
        out["has_precip_inferred"] = (out["precip_sum_inferred"].fillna(0.0) > 0).astype("float")
    if cloud_cols:
        vals = out[cloud_cols].apply(pd.to_numeric, errors="coerce")
        out["cloud_mean_inferred"] = vals.mean(axis=1)
        out["cloud_max_inferred"] = vals.max(axis=1)
    return out


def make_features(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    """Create model-ready features while keeping original numeric columns.

    Parameters
    ----------
    df:
        Raw train/test frame.
    target_col:
        Optional target column. If present, it is kept unchanged so train.py can
        remove it after feature engineering.
    """
    out = df.copy()
    out = add_time_features(out)
    out = add_availability_features(out)
    out = add_wind_features(out)
    out = add_air_density_features(out)
    out = add_weather_misc_features(out)

    # Convert object-like non-date columns to numeric category codes so that
    # rare textual flags do not break training. Date strings are dropped later.
    dt_col = find_datetime_col(out)
    for col in list(out.columns):
        if col == target_col or col == dt_col:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            parsed_num = pd.to_numeric(out[col], errors="coerce")
            if parsed_num.notna().mean() > 0.8:
                out[col] = parsed_num
            else:
                out[col] = out[col].astype("category").cat.codes.replace(-1, np.nan)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def available_capacity_from_raw(df: pd.DataFrame) -> pd.Series:
    """Return row-wise physical upper bound from repair count, in MW."""
    repair_col = find_col(df, "repair")
    if repair_col is None:
        return pd.Series(FARM_CAPACITY_MW, index=df.index)
    repair = pd.to_numeric(df[repair_col], errors="coerce").fillna(0.0).clip(0, N_TURBINES)
    cap = (N_TURBINES - repair) * TURBINE_CAPACITY_MW
    return cap.clip(0.0, FARM_CAPACITY_MW)


def sort_by_time_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    dt_col = find_datetime_col(df)
    if dt_col is None:
        return df.reset_index(drop=True)
    tmp = df.copy()
    tmp["__dt_sort"] = pd.to_datetime(tmp[dt_col], errors="coerce")
    tmp = tmp.sort_values("__dt_sort", kind="mergesort").drop(columns=["__dt_sort"])
    return tmp.reset_index(drop=True)
