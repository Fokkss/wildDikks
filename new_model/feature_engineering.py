from __future__ import annotations

import re

import numpy as np
import pandas as pd
from keywords import KIND_KEYWORDS, DATE_KEYWORDS, TARGET_KEYWORDS
import config


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

    # cube dependence example
    if 'ws_80m' in out.columns:
        out['wind_speed_cube'] = out['ws_80m'] ** 3
        # Рабочая зона турбины (отсекаем лишнее)
        out["hub_ws_piece_3_12"] = (out['ws_80m'].clip(3.0, 12.0) - 3.0).clip(lower=0.0)

    # (turbulence between 10m and 80m) example
    if 'ws_10m' in out.columns and 'ws_80m' in out.columns:
        out['wind_shear'] = out['ws_80m'] - out['ws_10m']

    # air density example
    if 'temp_k' in out.columns and 'pressure_pa' in out.columns:
        out['air_density'] = out['pressure_pa'] / (config.R_DRY_AIR * out['temp_k'])

        # Полная энергия ветрового потока: W = 0.5 * ρ * v^3
        if 'wind_speed_cube' in out.columns:
            out["wind_power_density"] = 0.5 * out['air_density'] * out['wind_speed_cube']

    if 'ws_80m' in out.columns:
        # Сдвигаем на 1 строку вниз (значение прошлого часа)
        out["hub_ws_lag1"] = out['ws_80m'].shift(1)
        # Ускорение/замедление ветра
        out["hub_ws_diff1"] = out['ws_80m'] - out['ws_80m'].shift(1)
        # Скользящее среднее за 3 часа (сглаживает порывы)
        out["hub_ws_roll_mean_3"] = out['ws_80m'].shift(1).rolling(3, min_periods=1).mean()

    return out


# =====================================================================
# BASE
# =====================================================================
def make_features(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
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

        # Тригонометрия для замыкания суток (23:00 и 00:00 будут рядом)
        out["hour_sin"] = np.sin(2.0 * np.pi * out["hour_of_day"] / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * out["hour_of_day"] / 24.0)
    return out


def _extract_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Считает, сколько турбин реально работает, вычитая те, что в ремонте."""
    out = df.copy()
    repair_col = find_col(out, "repair")
    if repair_col:
        repair_count = pd.to_numeric(out[repair_col], errors="coerce").fillna(0).clip(0, config.N_TURBINES)
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
        if col: out[f"ws_{h}m"] = pd.to_numeric(out[col], errors="coerce")

    # Ищем температуру и переводим в Кельвины
    temp_col = find_col(out, "temp", 80) or find_col(out, "temp")
    if temp_col:
        temp = pd.to_numeric(out[temp_col], errors="coerce")
        out["temp_k"] = temp if temp.median() > 150 else temp + 273.15  # Защита от фаренгейтов/цельсиев

    # Ищем давление и переводим в Паскали
    press_col = find_col(out, "pressure", 80) or find_col(out, "pressure")
    if press_col:
        press = pd.to_numeric(out[press_col], errors="coerce")
        out["pressure_pa"] = press * 100.0 if press.median() < 2000 else press

    return out


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", "_", str(s).lower()).strip("_")


def _has_height(col_norm: str, height_m: int) -> bool:
    return bool(re.search(rf"(^|_)({height_m})(m|м|meter|meters)?(_|$)", col_norm)) or str(height_m) in col_norm


def find_datetime_col(df: pd.DataFrame):
    for col in df.columns:
        if any(k in _norm(col) for k in DATE_KEYWORDS) and "hour" not in _norm(col) and "month" not in _norm(col):
            return col
    return None


def find_col(df: pd.DataFrame, kind: str, height_m: int = None):
    scored = []
    for col in df.columns:
        n = _norm(col)
        if not any(k in n for k in KIND_KEYWORDS[kind]): continue
        if height_m is not None and not _has_height(n, height_m): continue
        score = 10 if height_m is not None and _has_height(n, height_m) else 0
        scored.append((score, col))
    return sorted(scored, key=lambda x: -x[0])[0][1] if scored else None


def available_capacity_from_raw(df: pd.DataFrame) -> pd.Series:
    """Return row-wise physical upper bound from repair count, in MW."""
    repair_col = find_col(df, "repair")
    repair = pd.to_numeric(df[repair_col], errors="coerce").fillna(0.0).clip(0, config.N_TURBINES) if repair_col else 0
    return ((config.N_TURBINES - repair) * config.TURBINE_CAPACITY_MW).clip(0.0, config.FARM_CAPACITY_MW)


def find_target_col(df: pd.DataFrame, requested: str = None) -> str:
    if requested and requested in df.columns: return requested
    for col in df.columns:
        if any(k in _norm(col) for k in TARGET_KEYWORDS): return col
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
