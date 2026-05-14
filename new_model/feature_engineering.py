from __future__ import annotations

import re

import numpy as np
import pandas as pd

from new_model import config
from new_model.keywords import KIND_KEYWORDS, DATE_KEYWORDS, TARGET_KEYWORDS


# =====================================================================
# ---OSPREY SANDBOX---
# =====================================================================

def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    questa??

    df - это таблица, где колонки уже найдены и переименованы в удобный формат
    (например, 'ws_80m' - ветер на 80м, 'temp_k' - температура в Кельвинах).
    """
    out = df.copy()
    eps = 1e-6

    # availability features
    if "available_turbines" in out.columns:
        out["availability_ratio"] = out["available_turbines"] / config.N_TURBINES

    # cube dependence example
    if 'ws_80m' in out.columns:
        ws = out['ws_80m']

        # non-linear features
        out["ws80_sq"] = ws ** 2
        out["ws80_cube"] = ws ** 3
        out["wind_speed_cube"] = ws ** 3

        # Turbine operating zones
        out["is_below_cut_in"] = (ws < 3.0).astype(int)
        out["is_rated_zone"] = ((ws >= 12.0) & (ws <= 25.0)).astype(int)
        out["is_above_cut_out"] = (ws > 25.0).astype(int)

        # Рабочая зона турбины (отсекаем лишнее)
        out["hub_ws_piece_3_12"] = (ws.clip(3.0, 12.0) - 3.0).clip(lower=0.0)

        # approximate normalized power curve proxy
        out["power_curve_proxy"] = ((ws.clip(3.0, 12.0) - 3.0) / 9.0) ** 3
        out["power_curve_proxy"] = out["power_curve_proxy"].clip(0.0, 1.0)

        # lag and rolling weather features.
        # IMPORTANT: train.py should sort rows by datetime before make_features().
        out["hub_ws_lag1"] = ws.shift(1)
        out["hub_ws_lag2"] = ws.shift(2)
        out["hub_ws_lag3"] = ws.shift(3)

        out["hub_ws_diff1"] = ws - ws.shift(1)
        out["hub_ws_diff2"] = ws - ws.shift(2)

        out["hub_ws_roll_mean_3"] = ws.shift(1).rolling(3, min_periods=1).mean()
        out["hub_ws_roll_mean_6"] = ws.shift(1).rolling(6, min_periods=1).mean()
        out["hub_ws_roll_std_3"] = ws.shift(1).rolling(3, min_periods=2).std()
        out["hub_ws_roll_std_6"] = ws.shift(1).rolling(6, min_periods=2).std()

        # interactions with available capacity
        if "available_capacity_mw" in out.columns:
            out["expected_power_proxy"] = out["power_curve_proxy"] * out["available_capacity_mw"]
            out["ws80_x_available_capacity"] = ws * out["available_capacity_mw"]

        if "availability_ratio" in out.columns:
            out["ws80_cube_x_availability"] = out["ws80_cube"] * out["availability_ratio"]

    # wind shear features, (turbulence between 10m and 80m) example
    if "ws_10m" in out.columns and "ws_80m" in out.columns:
        out["wind_shear_80_10"] = out["ws_80m"] - out["ws_10m"]
        out["wind_shear_ratio_80_10"] = out["ws_80m"] / (out["ws_10m"] + eps)

        valid = (out["ws_10m"] > 0.5) & (out["ws_80m"] > 0.5)
        out["alpha_80_10"] = np.nan
        out.loc[valid, "alpha_80_10"] = (
                np.log(out.loc[valid, "ws_80m"] / out.loc[valid, "ws_10m"])
                / np.log(80 / 10)
        )

    if "ws_80m" in out.columns and "ws_120m" in out.columns:
        out["wind_shear_120_80"] = out["ws_120m"] - out["ws_80m"]
        out["wind_shear_ratio_120_80"] = out["ws_120m"] / (out["ws_80m"] + eps)

    if "ws_80m" in out.columns and "ws_180m" in out.columns:
        out["wind_shear_180_80"] = out["ws_180m"] - out["ws_80m"]
        out["wind_shear_ratio_180_80"] = out["ws_180m"] / (out["ws_80m"] + eps)

    # air density example
    if 'temp_k' in out.columns and 'pressure_pa' in out.columns:
        out['air_density'] = out['pressure_pa'] / (config.R_DRY_AIR * out['temp_k'])

        # Полная энергия ветрового потока: W = 0.5 * ρ * v^3
        if "ws80_cube" in out.columns:
            out["wind_power_density"] = 0.5 * out["air_density"] * out["ws80_cube"]

        if "expected_power_proxy" in out.columns:
            out["expected_power_density_adj"] = out["expected_power_proxy"] * out["air_density"]

    return out


# =====================================================================
# BASE
# =====================================================================
def make_features(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """
    cобирает сырой датасет прогоняет через блоки обработки
    и возвращает готовую матрицу для обучения XGBoost.
    """
    out = df.copy()

    # 1. Извлекаем базовые вещи (время, ремонты)
    out = _extract_time(out)
    out = _extract_availability(out)

    # 2. Находим и стандартизируем метеоданные
    out = _extract_weather(out)

    out = add_custom_features(out)

    # 4. Превращаем текстовые колонки в числа
    dt_col = find_datetime_col(out)

    for col in list(out.columns):
        if col == target_col or col == dt_col:
            continue

        if not pd.api.types.is_numeric_dtype(out[col]):
            # Если текст - кодируем его в числовые категории
            out[col] = out[col].astype("category").cat.codes.replace(-1, np.nan)

    return out.replace([np.inf, -np.inf], np.nan)


# =====================================================================
# --- DATA EXTRACTION LOGIC ---
# =====================================================================
def _extract_time(df: pd.DataFrame) -> pd.DataFrame:
    """Время в sin cos переводит для цикличности"""
    out = df.copy()
    dt_col = find_datetime_col(out)

    if dt_col:
        dt = pd.to_datetime(out[dt_col], errors="coerce")

        out["hour_of_day"] = dt.dt.hour
        out["month"] = dt.dt.month
        out["dayofyear"] = dt.dt.dayofyear
        out["dayofweek"] = dt.dt.dayofweek

        # Тригонометрия для замыкания суток (23:00 и 00:00 будут рядом)
        out["hour_sin"] = np.sin(2.0 * np.pi * out["hour_of_day"] / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * out["hour_of_day"] / 24.0)

        # trigonometry for months
        out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
        out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)

        # trigonometry for years
        out["dayofyear_sin"] = np.sin(2.0 * np.pi * out["dayofyear"] / 365.25)
        out["dayofyear_cos"] = np.cos(2.0 * np.pi * out["dayofyear"] / 365.25)

    return out


def _extract_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Считает, сколько турбин реально работает, вычитая те, что в ремонте."""
    out = df.copy()
    repair_col = find_col(out, "repair")

    if repair_col:
        repair_count = (
            pd.to_numeric(out[repair_col], errors="coerce")
            .fillna(0)
            .clip(0, config.N_TURBINES)
        )
        out["available_turbines"] = config.N_TURBINES - repair_count
        out["available_capacity_mw"] = out["available_turbines"] * config.TURBINE_CAPACITY_MW

    return out


def _extract_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ищет в сырых данных ветер/давление/температуру и создает для них
    стандартизированные колонки (ws_80m, temp_k)
    """
    out = df.copy()

    # Ищем ветер на разных высотах
    for h in (10, 80, 120, 180):
        col = find_col(out, "wind_speed", h)
        if col:
            out[f"ws_{h}m"] = pd.to_numeric(out[col], errors="coerce")

    # if hub-height wind is absent, approximate 80m wind from 10m and 120m
    if "ws_80m" not in out.columns:
        if "ws_10m" in out.columns and "ws_120m" in out.columns:
            # log-linear interpolation by height
            h1, h2, h = 10.0, 120.0, 80.0
            w1 = out["ws_10m"].clip(lower=0.1)
            w2 = out["ws_120m"].clip(lower=0.1)
            alpha = np.log(w2 / w1) / np.log(h2 / h1)
            out["ws_80m"] = w1 * (h / h1) ** alpha

    # Wind direction
    dir_col = find_col(out, "wind_dir", 80) or find_col(out, "wind_dir")
    if dir_col:
        wd = pd.to_numeric(out[dir_col], errors="coerce")
        wd_rad = np.deg2rad(wd)

        out["wind_dir_sin"] = np.sin(wd_rad)
        out["wind_dir_cos"] = np.cos(wd_rad)

        out["wind_sector_8"] = ((wd % 360) // 45).astype("float")
        out["wind_sector_16"] = ((wd % 360) // 22.5).astype("float")

    # Ищем температуру и переводим в Кельвины
    temp_col = find_col(out, "temp", 80) or find_col(out, "temp")
    if temp_col:
        temp = pd.to_numeric(out[temp_col], errors="coerce")
        med_temp = temp.median(skipna=True)

        if pd.notna(med_temp):
            out["temp_k"] = temp if med_temp > 150 else temp + 273.15

    # Ищем давление и переводим в Паскали
    press_col = find_col(out, "pressure", 80) or find_col(out, "pressure")
    if press_col:
        press = pd.to_numeric(out[press_col], errors="coerce")
        med_press = press.median(skipna=True)

        if pd.notna(med_press):
            out["pressure_pa"] = press * 100.0 if press.median() < 2000 else press

    # precipitation
    precip_col = find_col(out, "precip")
    if precip_col:
        out["precip"] = pd.to_numeric(out[precip_col], errors="coerce")

    # cloudiness
    cloud_col = find_col(out, "cloud")
    if cloud_col:
        out["cloud"] = pd.to_numeric(out[cloud_col], errors="coerce")

    return out


# =====================================================================
# COLUMN SEARCH UTILS
# =====================================================================
def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", "_", str(s).lower()).strip("_")


def _has_height(col_norm: str, height_m: int) -> bool:
    return (
            bool(re.search(rf"(^|_)({height_m})(m|м|meter|meters)?(_|$)", col_norm))
            or str(height_m) in col_norm
    )

def find_datetime_col(df: pd.DataFrame):
    for col in df.columns:
        n = _norm(col)
        if any(k in n for k in DATE_KEYWORDS) and "hour" not in n and "month" not in n:
            return col
    return None


def find_col(df: pd.DataFrame, kind: str, height_m: int | None = None):
    scored = []

    for col in df.columns:
        n = _norm(col)

        if not any(k in n for k in KIND_KEYWORDS[kind]):
            continue

        if height_m is not None and not _has_height(n, height_m):
            continue

        score = 10 if height_m is not None and _has_height(n, height_m) else 0
        scored.append((score, col))

    return sorted(scored, key=lambda x: -x[0])[0][1] if scored else None


def available_capacity_from_raw(df: pd.DataFrame) -> pd.Series:
    """Return row-wise physical upper bound from repair count, in MW."""
    repair_col = find_col(df, "repair")

    if repair_col:
        repair = (
            pd.to_numeric(df[repair_col], errors="coerce")
            .fillna(0.0)
            .clip(0, config.N_TURBINES)
        )
    else:
        repair = pd.Series(0.0, index=df.index)

    return (
            (config.N_TURBINES - repair) * config.TURBINE_CAPACITY_MW
    ).clip(0.0, config.FARM_CAPACITY_MW)

def find_target_col(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested and requested in df.columns:
        return requested

    for col in df.columns:
        if any(k in _norm(col) for k in TARGET_KEYWORDS):
            return col

    raise ValueError("Target column not found.")


# :NOTE: used in train.py (for what?)
def sort_by_time_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    dt_col = find_datetime_col(df)

    if dt_col is None:
        return df.reset_index(drop=True)

    tmp = df.copy()
    tmp["__dt_sort"] = pd.to_datetime(tmp[dt_col], errors="coerce")
    tmp = tmp.sort_values("__dt_sort", kind="mergesort").drop(columns=["__dt_sort"])

    return tmp.reset_index(drop=True)
