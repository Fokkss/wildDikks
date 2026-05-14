from __future__ import annotations

import re

import numpy as np
import pandas as pd

from new_model import config
from new_model.keywords import KIND_KEYWORDS, DATE_KEYWORDS, TARGET_KEYWORDS


# :TODO: check all constants for accuracy and move them to config.py
# :NB!: constants - almost every number in file which repeats or have some sacred logic
# =====================================================================
# ---OSPREY SANDBOX---
# =====================================================================
# :NOTE for OSPREY: all staff below (in add_custom_features) must be checked
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

    if 'ws_80m' in out.columns:
        ws = out['ws_80m']

        # non-linear features
        # cube dependence example
        out["ws80_sq"] = ws ** 2
        out["ws80_cube"] = ws ** 3

        # Turbine operating zones
        out["is_below_cut_in"] = (ws < 3.0).astype(int)
        out["is_rated_zone"] = ((ws >= 12.0) & (ws <= 25.0)).astype(int)
        out["is_above_cut_out"] = (ws > 25.0).astype(int)
        out["hub_ws_piece_3_12"] = (ws.clip(3.0, 12.0) - 3.0).clip(lower=0.0)

        # approximate normalized power curve proxy
        # :FIXED: maybe better proxy
        proxy = ((ws.clip(3.0, 12.0) - 3.0) / 9.0) ** 3
        out["power_curve_proxy"] = np.where(ws > 25.0, 0.0, proxy)

        # lag and rolling weather features.
        # :IMPORTANT: train.py should sort rows by datetime before make_features().
        # :NOTE: days lags added
        out["hub_ws_lag1"] = ws.shift(1)
        out["hub_ws_lag2"] = ws.shift(2)
        out["hub_ws_lag24"] = ws.shift(24)
        out["hub_ws_diff1"] = ws - ws.shift(1)
        out["hub_ws_roll_mean_3"] = ws.shift(1).rolling(3, min_periods=1).mean()

        # Взаимодействие с ремонтами
        if "available_capacity_mw" in out.columns:
            out["expected_power_proxy"] = out["power_curve_proxy"] * out["available_capacity_mw"]

        # :NOTE: ~doubtful. need be checked
    #     out["hub_ws_diff1"] = ws - ws.shift(1)
    #     out["hub_ws_diff2"] = ws - ws.shift(2)
    #
    #     out["hub_ws_roll_mean_3"] = ws.shift(1).rolling(3, min_periods=1).mean()
    #     out["hub_ws_roll_mean_6"] = ws.shift(1).rolling(6, min_periods=1).mean()
    #     out["hub_ws_roll_std_3"] = ws.shift(1).rolling(3, min_periods=2).std()
    #     out["hub_ws_roll_std_6"] = ws.shift(1).rolling(6, min_periods=2).std()
    #
    #     # interactions with available capacity
    #     if "available_capacity_mw" in out.columns:
    #         out["expected_power_proxy"] = out["power_curve_proxy"] * out["available_capacity_mw"]
    #         out["ws80_x_available_capacity"] = ws * out["available_capacity_mw"]
    #
    #     if "availability_ratio" in out.columns:
    #         out["ws80_cube_x_availability"] = out["ws80_cube"] * out["availability_ratio"]
    #
    # # wind shear features, (turbulence between 10m and 80m) example
    # if "ws_10m" in out.columns and "ws_80m" in out.columns:
    #     out["wind_shear_80_10"] = out["ws_80m"] - out["ws_10m"]
    #     out["wind_shear_ratio_80_10"] = out["ws_80m"] / (out["ws_10m"] + eps)
    #
    #     valid = (out["ws_10m"] > 0.5) & (out["ws_80m"] > 0.5)
    #     out["alpha_80_10"] = np.nan
    #     out.loc[valid, "alpha_80_10"] = (
    #             np.log(out.loc[valid, "ws_80m"] / out.loc[valid, "ws_10m"])
    #             / np.log(80 / 10)
    #     )
    #
    # if "ws_80m" in out.columns and "ws_120m" in out.columns:
    #     out["wind_shear_120_80"] = out["ws_120m"] - out["ws_80m"]
    #     out["wind_shear_ratio_120_80"] = out["ws_120m"] / (out["ws_80m"] + eps)
    #
    # if "ws_80m" in out.columns and "ws_180m" in out.columns:
    #     out["wind_shear_180_80"] = out["ws_180m"] - out["ws_80m"]
    #     out["wind_shear_ratio_180_80"] = out["ws_180m"] / (out["ws_80m"] + eps)

    # air density example
    # Полная энергия ветрового потока: W = 0.5 * ρ * v^3
    if 'temp_k' in out.columns and 'pressure_pa' in out.columns:
        out['air_density'] = out['pressure_pa'] / (config.R_DRY_AIR * out['temp_k'])
        if "ws80_cube" in out.columns:
            out["wind_power_density"] = 0.5 * out["air_density"] * out["ws80_cube"]

        # :NOTE: ~doubtful. need be checked
        # if "expected_power_proxy" in out.columns:
        #     out["expected_power_density_adj"] = out["expected_power_proxy"] * out["air_density"]

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

    inv_map = {v: k for k, v in config.COLUMN_MAP.items()}
    rename_dict = {k: v for k, v in config.COLUMN_MAP.items() if k in out.columns}
    out.rename(columns=rename_dict, inplace=True)

    # Извлекаем базовые вещи (время, ремонты)
    out = _extract_time(out)
    out = _extract_availability(out)
    out = _extract_weather(out)

    out = add_custom_features(out)

    for col in list(out.columns):
        if col == target_col or col == "datetime":
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("category").cat.codes.replace(-1, np.nan)

    return out.replace([np.inf, -np.inf], np.nan)


# =====================================================================
# --- DATA EXTRACTION LOGIC ---
# =====================================================================

def _extract_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        dt = pd.to_datetime(out["datetime"], errors="coerce")

        out["hour_of_day"] = dt.dt.hour
        out["month"] = dt.dt.month
        out["dayofyear"] = dt.dt.dayofyear
        out["dayofweek"] = dt.dt.dayofweek

        # trigonometry for hours
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
    """Считает доступные турбины и мощность, основываясь на колонке repair_count."""
    out = df.copy()
    if "repair_count" in out.columns:
        repair = (
            pd.to_numeric(out["repair_count"], errors="coerce")
            .fillna(0.0)
            .clip(0, config.N_TURBINES)
        )
        # Кол-во работающих
        out["available_turbines"] = config.N_TURBINES - repair
        # Доступная мощность в МВт (физический потолок для клипа таргета)
        out["available_capacity_mw"] = out["available_turbines"] * config.TURBINE_CAPACITY_MW
    else:
        # Дефолтные значения, если колонки нет
        out["available_turbines"] = config.N_TURBINES
        out["availability_ratio"] = 1.0
        out["available_capacity_mw"] = config.FARM_CAPACITY_MW
    return out


# Some features here
def _extract_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ищет в сырых данных ветер/давление/температуру и создает для них
    стандартизированные колонки (ws_80m, temp_k)
    """
    out = df.copy()

    # 1. Извлечение скоростей ветра на всех доступных высотах
    for h in (10, 80, 120, 180):
        # Пытаемся найти колонку либо через COLUMN_MAP, либо через find_col
        col = f"ws_{h}m"
        out[f"ws_{h}m"] = pd.to_numeric(out[col], errors="coerce")

    # if hub-height wind is absent, approximate 80m wind from 10m and 120m
    # logarithmic interpolation
    # :NOTE: what?
    if "ws_80m" not in out.columns or out["ws_80m"].isna().all():
        if "ws_10m" in out.columns and "ws_120m" in out.columns:
            # Power law profile: w2/w1 = (h2/h1)^alpha => alpha = log(w2/w1) / log(h2/h1)
            h1, h2, h = 10.0, 120.0, 80.0
            w1 = out["ws_10m"].clip(lower=0.1)
            w2 = out["ws_120m"].clip(lower=0.1)
            alpha = np.log(w2 / w1) / np.log(h2 / h1)
            out["ws_80m"] = w1 * (h / h1) ** alpha

    # 3. Обработка направления ветра (на 80м или ближайшее доступное)
    dir_col = "wd_80m"
    if dir_col:
        wd = pd.to_numeric(out[dir_col], errors="coerce")
        wd_rad = np.deg2rad(wd)

        out["wind_dir_sin"] = np.sin(wd_rad)
        out["wind_dir_cos"] = np.cos(wd_rad)

        # Сектора ветра для учета розы ветров
        out["wind_sector_8"] = ((wd % 360) // 45).astype("float")
        out["wind_sector_16"] = ((wd % 360) // 22.5).astype("float")

    # Pressure correllation feature (гПа -> Па)
    press_col = "pressure"
    if press_col:
        press = pd.to_numeric(out[press_col], errors="coerce")
        # Если медиана < 2000, значит это гПа/мбар, переводим в Паскали
        if press.median(skipna=True) < 2000:
            out["pressure_pa"] = press * 100.0
        else:
            out["pressure_pa"] = press

    # Finding temperature and converting Cels to Kelvin
    # :NOTE: Maybe Kelvin constant should be fixed
    # :NOTE: similary make for 120
    temp_col = "temp_k_80"

    temp = pd.to_numeric(out[temp_col], errors="coerce")
    med_temp = temp.median(skipna=True)
    # Если медиана < 150, значит это Цельсий, переводим в Кельвины
    if pd.notna(med_temp) and med_temp < 150:
        out["temp_k"] = temp + 273.15
    else:
        out["temp_k"] = temp

    # there must be somewhere
    # precipitation & cloudiness features

    return out


# =====================================================================
# COLUMN SEARCH UTILS
# =====================================================================

# :NOTE: used in train.py (for what?)
def sort_by_time_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    time_raw_name = next((k for k, v in config.COLUMN_MAP.items() if v == "datetime"), None)
    if time_raw_name and time_raw_name in df.columns:
        tmp = df.copy()
        tmp["__dt_sort"] = pd.to_datetime(tmp[time_raw_name], errors="coerce")
        return tmp.sort_values("__dt_sort", kind="mergesort").drop(columns=["__dt_sort"]).reset_index(drop=True)
    return df.reset_index(drop=True)


def available_capacity_from_raw(df: pd.DataFrame) -> pd.Series:
    """Для таргета: физический лимит с учетом жесткого маппинга"""
    repair_raw_name = next((k for k, v in config.COLUMN_MAP.items() if v == "repair_count"), None)

    if repair_raw_name and repair_raw_name in df.columns:
        repair = pd.to_numeric(df[repair_raw_name], errors="coerce").fillna(0.0).clip(0, config.N_TURBINES)
    else:
        repair = pd.Series(0.0, index=df.index)

    return ((config.N_TURBINES - repair) * config.TURBINE_CAPACITY_MW).clip(0.0, config.FARM_CAPACITY_MW)
