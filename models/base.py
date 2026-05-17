from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import re


TARGET_COL = "Выработка"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

INSTALLED_CAPACITY_MW = 90.09

N_TURBINES = 26
TURBINE_CAPACITY_MW = INSTALLED_CAPACITY_MW / N_TURBINES
R_DRY_AIR = 287.05
STANDARD_AIR_DENSITY = 1.225  # kg/m^3
BETZ_LIMIT_CP = 16.0 / 27.0   # ≈ 0.593, theoretical max efficiency
R_WATER_VAPOR = 461.5


ROTOR_DIAMETER_M = 132.0
ROTOR_SWEPT_AREA_M2 = np.pi * (ROTOR_DIAMETER_M / 2.0) ** 2

LAG_HOURS = (1, 2, 3, 6, 12, 24)
DIFF_HOURS = (1, 3, 24)
ROLLING_MEAN_HOURS = (3, 6, 12, 24)
ROLLING_STD_HOURS = (3, 6, 12)


# Названия, которые приводим к нормальному виду.
# Главная цель — не таскать по проекту битую кодировку.
COLUMN_RENAME_MAP = {
    "РљРѕР»-РІРѕ_Р’Р­РЈ_РІ_СЂРµРјРѕРЅС‚Рµ": "repair_count",
    "Кол-во_ВЭУ_в_ремонте": "repair_count",
    "Количество_ВЭУ_в_ремонте": "repair_count",
}


DEFAULT_FEATURE_FLAGS = {
    "time": True, #false?
    "availability": True, #fasle?
    "power_curve": True,
    "air_density": True,
    "wind_power_density": True,
    "density_adjusted_power": True,
    "wind_direction": True,
    "betz_limit": True,
    "lags": True,
    "multi_height_power": True,
    "gust_features": True,
    "wind_shear": True,
    "wind_regime": True,
    "icing": True,
    "ramp_zone_features": True,
    "precipitation_stress": True,
    "time_interactions": True,
    "rotor_equivalent_wind": True,
    "yaw_misalignment": True,
    "moist_air_density": True,
    "thermal_derating": True,
    "boundary_layer_stability": True,
    "momentum_flux": True,
}


def _norm_col(value: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", "_", str(value).lower()).strip("_")


def _find_col_by_tokens(
    df: pd.DataFrame,
    required_tokens: tuple[str, ...],
    optional_tokens: tuple[str, ...] = (),
    forbidden_tokens: tuple[str, ...] = (),
) -> str | None:
    """
    Fuzzy column finder.

    required_tokens: every group must be found.
    optional_tokens: increase priority if found.
    forbidden_tokens: reject columns if found.
    """
    candidates: list[tuple[int, str]] = []

    for col in df.columns:
        n = _norm_col(col)

        if any(tok in n for tok in forbidden_tokens):
            continue

        if not all(tok in n for tok in required_tokens):
            continue

        score = 0
        for tok in optional_tokens:
            if tok in n:
                score += 1

        candidates.append((score, col))

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: -x[0])[0][1]


def _find_wind_speed_col(df: pd.DataFrame, height_m: int) -> str | None:
    height = str(height_m)

    candidates: list[tuple[int, str]] = []

    for col in df.columns:
        n = _norm_col(col)

        has_speed = (
            "wind_speed" in n
            or "windspeed" in n
            or "speed" in n
            or "скорость" in n
        )
        has_height = height in n
        is_direction = (
            "direction" in n
            or "wind_dir" in n
            or "dir" in n
            or "направ" in n
        )

        if has_speed and has_height and not is_direction:
            score = 0
            if f"{height}m" in n or f"{height}_m" in n:
                score += 2
            candidates.append((score, col))

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: -x[0])[0][1]


def _find_wind_direction_col(df: pd.DataFrame, height_m: int) -> str | None:
    height = str(height_m)

    candidates: list[tuple[int, str]] = []

    for col in df.columns:
        n = _norm_col(col)

        has_direction = (
            "wind_direction" in n
            or "wind_dir" in n
            or "direction" in n
            or "dir" in n
            or "направ" in n
        )
        has_height = height in n
        is_speed = (
            "wind_speed" in n
            or "windspeed" in n
            or "speed" in n
            or "скорость" in n
        )

        if has_direction and has_height and not is_speed:
            score = 0
            if f"{height}m" in n or f"{height}_m" in n:
                score += 2
            candidates.append((score, col))

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: -x[0])[0][1]


def _find_wind_gust_col(df: pd.DataFrame, height_m: int = 10) -> str | None:
    height = str(height_m)

    candidates: list[tuple[int, str]] = []

    for col in df.columns:
        n = _norm_col(col)

        has_gust = (
            "gust" in n
            or "gusts" in n
            or "порыв" in n
        )
        has_height = height in n

        if has_gust and has_height:
            score = 0
            if f"{height}m" in n or f"{height}_m" in n:
                score += 2
            candidates.append((score, col))

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: -x[0])[0][1]


def _find_temperature_col(df: pd.DataFrame) -> str | None:
    return (
        _find_col_by_tokens(df, required_tokens=("temp",), optional_tokens=("80",))
        or _find_col_by_tokens(df, required_tokens=("temperature",), optional_tokens=("80",))
        or _find_col_by_tokens(df, required_tokens=("температура",), optional_tokens=("80",))
    )


def _find_pressure_col(df: pd.DataFrame) -> str | None:
    return (
        _find_col_by_tokens(df, required_tokens=("pressure",))
        or _find_col_by_tokens(df, required_tokens=("press",))
        or _find_col_by_tokens(df, required_tokens=("давление",))
        or _find_col_by_tokens(df, required_tokens=("msl",))
    )


def _numeric_series(
    df: pd.DataFrame,
    col: str | None,
    default: float = 0.0,
) -> pd.Series:
    """
    Safely returns numeric column as Series.
    If column is missing, returns constant Series with dataframe index.
    """

    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")

    return pd.to_numeric(df[col], errors="coerce")


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    eps: float = 1e-6,
) -> pd.Series:
    return numerator / (denominator + eps)


def _angle_diff_deg(
    a_deg: pd.Series,
    b_deg: pd.Series,
) -> pd.Series:
    """
    Smallest signed difference between two directions in degrees.
    Result is in [-180, 180].

    Example:
        a=350, b=10 -> -20
        a=10, b=350 -> 20
    """

    return ((a_deg - b_deg + 180.0) % 360.0) - 180.0


def _saturation_vapor_pressure_pa(temp_c: pd.Series) -> pd.Series:
    """
    Saturation vapor pressure proxy by Magnus/Tetens-style formula.

    Returns Pa.
    Uses water formula for T >= 0 C and ice-ish formula for T < 0 C.
    """

    temp_c = pd.to_numeric(temp_c, errors="coerce")

    # hPa over water
    es_water_hpa = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))

    # hPa over ice / cold conditions
    es_ice_hpa = 6.112 * np.exp((22.46 * temp_c) / (temp_c + 272.62))

    es_hpa = pd.Series(
        np.where(temp_c < 0.0, es_ice_hpa, es_water_hpa),
        index=temp_c.index,
    )

    return es_hpa * 100.0


def _normalize_cloud_fraction(cloud: pd.Series) -> pd.Series:
    """
    cloud_cover иногда бывает 0..1, иногда 0..100.
    Приводим к 0..1.
    """

    cloud = pd.to_numeric(cloud, errors="coerce").fillna(0.0)
    median = cloud.median(skipna=True)

    if pd.notna(median) and median > 1.5:
        cloud = cloud / 100.0

    return cloud.clip(0.0, 1.0)


def _add_moist_air_density_features(
    df: pd.DataFrame,
    temp_col: str | None,
    pressure_col: str | None,
) -> pd.DataFrame:
    """
    Proxy for moist-air density.

    Важно:
    - не заменяем df["air_density"], чтобы не сломать старый полезный сигнал;
    - добавляем отдельные moist_* признаки;
    - RH не измерена напрямую, поэтому оцениваем её через осадки/облачность.
    """

    if temp_col is None or pressure_col is None:
        return df

    if temp_col not in df.columns or pressure_col not in df.columns:
        return df

    out = df.copy()

    temp = pd.to_numeric(out[temp_col], errors="coerce")
    pressure = pd.to_numeric(out[pressure_col], errors="coerce")

    temp_median = temp.median(skipna=True)
    pressure_median = pressure.median(skipna=True)

    # C / K auto-detect
    if pd.notna(temp_median) and temp_median > 150:
        temp_k = temp
        temp_c = temp - 273.15
    else:
        temp_c = temp
        temp_k = temp + 273.15

    # hPa / Pa auto-detect
    if pd.notna(pressure_median) and pressure_median < 2000:
        pressure_pa = pressure * 100.0
    else:
        pressure_pa = pressure

    rain = _numeric_series(out, "rain", default=0.0).fillna(0.0).clip(lower=0.0)
    showers = _numeric_series(out, "showers", default=0.0).fillna(0.0).clip(lower=0.0)
    snowfall = _numeric_series(out, "snowfall", default=0.0).fillna(0.0).clip(lower=0.0)

    if "cloud_cover_low" in out.columns:
        cloud_low = _normalize_cloud_fraction(out["cloud_cover_low"])
    else:
        cloud_low = pd.Series(0.0, index=out.index)

    precip_total = rain + showers + snowfall

    # Осадки в "попугаях": не знаем единицы, поэтому log1p устойчивее.
    precip_intensity_proxy = np.log1p(precip_total).clip(0.0, 2.0) / 2.0
    showers_intensity_proxy = np.log1p(showers).clip(0.0, 2.0) / 2.0
    snow_intensity_proxy = np.log1p(snowfall).clip(0.0, 2.0) / 2.0

    has_rain = (rain > 0.0).astype(float)
    has_showers = (showers > 0.0).astype(float)
    has_snowfall = (snowfall > 0.0).astype(float)
    has_precip = (precip_total > 0.0).astype(float)

    # Базовая эвристика RH:
    # - без осадков не ставим 100%;
    # - при ливнях почти насыщение;
    # - при снеге высокая влажность, но эффект плотности меньше из-за холода.
    rh_proxy = (
        0.50
        + 0.25 * cloud_low
        + 0.25 * precip_intensity_proxy
        + 0.10 * showers_intensity_proxy
        + 0.05 * snow_intensity_proxy
    )

    rh_proxy = pd.Series(rh_proxy, index=out.index)

    rh_proxy = rh_proxy.where(
        has_precip <= 0.0,
        np.maximum(rh_proxy, 0.85),
    )

    rh_proxy = rh_proxy.where(
        has_rain <= 0.0,
        np.maximum(rh_proxy, 0.90),
    )

    rh_proxy = rh_proxy.where(
        has_showers <= 0.0,
        np.maximum(rh_proxy, 0.97),
    )

    rh_proxy = rh_proxy.where(
        has_snowfall <= 0.0,
        np.maximum(rh_proxy, 0.88),
    )

    rh_proxy = rh_proxy.clip(0.35, 1.0)

    saturation_vapor_pressure_pa = _saturation_vapor_pressure_pa(temp_c)

    vapor_pressure_pa = rh_proxy * saturation_vapor_pressure_pa

    # Не даём водяному пару стать физически невозможным.
    vapor_pressure_pa = vapor_pressure_pa.clip(
        lower=0.0,
        upper=pressure_pa * 0.99,
    )

    dry_air_density = pressure_pa / (R_DRY_AIR * temp_k)

    moist_air_density = (
        (pressure_pa - vapor_pressure_pa) / (R_DRY_AIR * temp_k)
        + vapor_pressure_pa / (R_WATER_VAPOR * temp_k)
    )

    out["relative_humidity_proxy"] = rh_proxy
    out["saturation_vapor_pressure_pa_proxy"] = saturation_vapor_pressure_pa
    out["vapor_pressure_pa_proxy"] = vapor_pressure_pa

    out["moist_air_density"] = moist_air_density
    out["air_density_dry_proxy_for_moist_calc"] = dry_air_density
    out["air_density_moist_delta"] = moist_air_density - dry_air_density
    out["air_density_moist_ratio"] = moist_air_density / (dry_air_density + 1e-6)

    # Specific humidity proxy. Полезно как отдельная метео-фича.
    out["specific_humidity_proxy"] = (
        0.622 * vapor_pressure_pa / (pressure_pa - 0.378 * vapor_pressure_pa + 1e-6)
    )

    # Vapor pressure deficit: чем меньше, тем ближе к насыщению.
    out["vapor_pressure_deficit_pa_proxy"] = (
        saturation_vapor_pressure_pa - vapor_pressure_pa
    ).clip(lower=0.0)

    # Механика: влажная погода как режим, не только физическая плотность.
    out["wet_air_proxy"] = has_precip
    out["rain_or_showers_proxy"] = ((rain + showers) > 0.0).astype(float)
    out["humid_low_cloud_proxy"] = rh_proxy * cloud_low
    out["humid_precip_proxy"] = rh_proxy * has_precip

    # Moist wind power density for existing wind-speed cubes.
    cube_to_output = {
        "ws80_cube": "wind_power_density_moist",
        "ws120_cube": "wind_power_density_120m_moist",
        "ws180_cube": "wind_power_density_180m_moist",
        "rotor_equiv_ws_cube": "wind_power_density_rews_moist",
        "gust10_cube": "gust10_power_density_moist",
    }

    for cube_col, out_col in cube_to_output.items():
        if cube_col in out.columns:
            out[out_col] = 0.5 * out["moist_air_density"] * out[cube_col]

    # Density-adjusted expected power variants.
    expected_power_cols = [
        "expected_power_proxy",
        "expected_power_proxy_120m",
        "expected_power_proxy_180m",
        "expected_power_proxy_rews",
    ]

    for col in expected_power_cols:
        if col in out.columns:
            out[f"{col}_moist_density_adj"] = (
                out[col] * out["moist_air_density"] / STANDARD_AIR_DENSITY
            )

            out[f"{col}_moist_minus_dry_density_adj"] = (
                out[f"{col}_moist_density_adj"]
                - out[col] * dry_air_density / STANDARD_AIR_DENSITY
            )

    return out

def _temp_to_celsius_and_kelvin(temp: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Auto-detect temperature units and return (temp_c, temp_k).
    """

    temp = pd.to_numeric(temp, errors="coerce")
    temp_median = temp.median(skipna=True)

    if pd.notna(temp_median) and temp_median > 150:
        temp_k = temp
        temp_c = temp - 273.15
    else:
        temp_c = temp
        temp_k = temp + 273.15

    return temp_c, temp_k


def _wind_components_from_speed_dir(
    speed: pd.Series,
    direction_raw: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Simple wind-vector components from speed and direction.

    Direction in your data is degrees / 1000, so 0.273 -> 273 degrees.
    For feature engineering, exact meteorological u/v sign convention is
    less important than stable vector differences between heights.
    """

    speed = pd.to_numeric(speed, errors="coerce")
    direction_deg = _to_wind_dir_deg(direction_raw)
    direction_rad = np.deg2rad(direction_deg)

    u = speed * np.sin(direction_rad)
    v = speed * np.cos(direction_rad)

    return u, v


def _add_boundary_layer_stability_features(
    df: pd.DataFrame,
    temp_col: str | None,
    ws80_col: str | None,
    ws120_col: str | None,
    ws180_col: str | None,
    datetime_col: str,
) -> pd.DataFrame:
    """
    Boundary-layer stability / heat-flux proxy features.

    Captures:
    - bulk Richardson proxy between 80m and 120m;
    - vertical wind-vector shear;
    - stable night regime;
    - convective daytime / heat-flux-like regime.

    This is not exact meteorological heat flux. It is a robust proxy from
    available columns: temperature, wind speeds, wind directions, cloud cover, time.
    """

    if temp_col is None or temp_col not in df.columns:
        return df

    if ws80_col is None or ws120_col is None:
        return df

    if ws80_col not in df.columns or ws120_col not in df.columns:
        return df

    out = df.copy()

    temp80_raw = pd.to_numeric(out[temp_col], errors="coerce")
    temp80_c, temp80_k = _temp_to_celsius_and_kelvin(temp80_raw)

    if "temperature_120m" in out.columns:
        temp120_raw = pd.to_numeric(out["temperature_120m"], errors="coerce")
        temp120_c, temp120_k = _temp_to_celsius_and_kelvin(temp120_raw)
    else:
        temp120_c = pd.Series(np.nan, index=out.index)
        temp120_k = pd.Series(np.nan, index=out.index)

    ws80 = pd.to_numeric(out[ws80_col], errors="coerce")
    ws120 = pd.to_numeric(out[ws120_col], errors="coerce")
    ws180 = _numeric_series(out, ws180_col, default=np.nan)

    dir80_col = _find_wind_direction_col(out, 80)
    dir120_col = _find_wind_direction_col(out, 120)
    dir180_col = _find_wind_direction_col(out, 180)

    if dir80_col is not None and dir120_col is not None:
        u80, v80 = _wind_components_from_speed_dir(ws80, out[dir80_col])
        u120, v120 = _wind_components_from_speed_dir(ws120, out[dir120_col])
        vector_shear_120_80 = np.sqrt((u120 - u80) ** 2 + (v120 - v80) ** 2)
    else:
        vector_shear_120_80 = (ws120 - ws80).abs()

    if dir80_col is not None and dir180_col is not None and ws180_col is not None:
        u80_for_180, v80_for_180 = _wind_components_from_speed_dir(ws80, out[dir80_col])
        u180, v180 = _wind_components_from_speed_dir(ws180, out[dir180_col])
        vector_shear_180_80 = np.sqrt((u180 - u80_for_180) ** 2 + (v180 - v80_for_180) ** 2)
    else:
        vector_shear_180_80 = (ws180 - ws80).abs()

    dz_120_80 = 40.0
    g = 9.80665
    eps = 1e-4

    delta_temp_120_80 = temp120_c - temp80_c

    # Bulk Richardson proxy:
    # positive -> more stable;
    # negative -> more convective / unstable;
    # near zero -> neutral-ish.
    ri_120_80 = (
        (g / (temp80_k + 1e-6))
        * delta_temp_120_80
        * dz_120_80
        / ((vector_shear_120_80 ** 2) + eps)
    )

    out["temp_gradient_120_80_c"] = delta_temp_120_80
    out["abs_temp_gradient_120_80_c"] = delta_temp_120_80.abs()

    out["wind_vector_shear_120_80"] = vector_shear_120_80
    out["wind_vector_shear_180_80"] = vector_shear_180_80
    out["wind_vector_shear_120_80_sq"] = vector_shear_120_80 ** 2

    # clip, чтобы одиночные почти нулевые shear не создавали безумные значения
    out["bulk_richardson_120_80_proxy"] = ri_120_80.clip(-10.0, 10.0)
    out["abs_bulk_richardson_120_80_proxy"] = out["bulk_richardson_120_80_proxy"].abs()

    out["is_unstable_layer_120_80"] = (ri_120_80 < -0.03).astype(float)
    out["is_neutral_layer_120_80"] = (ri_120_80.abs() <= 0.05).astype(float)
    out["is_stable_layer_120_80"] = (ri_120_80 > 0.05).astype(float)
    out["is_very_stable_layer_120_80"] = (ri_120_80 > 0.25).astype(float)

    # Time / cloud proxies for heat flux.
    if datetime_col in out.columns:
        dt = pd.to_datetime(out[datetime_col], errors="coerce")
        hour = dt.dt.hour
        month = dt.dt.month
    elif "hour_of_day" in out.columns:
        hour = pd.to_numeric(out["hour_of_day"], errors="coerce")
        month = pd.Series(np.nan, index=out.index)
    else:
        hour = pd.Series(np.nan, index=out.index)
        month = pd.Series(np.nan, index=out.index)

    if "cloud_cover_low" in out.columns:
        cloud_low = _normalize_cloud_fraction(out["cloud_cover_low"])
    else:
        cloud_low = pd.Series(0.0, index=out.index)

    daylight_proxy = ((hour >= 8) & (hour <= 18)).astype(float)
    night_proxy = ((hour <= 6) | (hour >= 20)).astype(float)
    clear_sky_proxy = (1.0 - cloud_low).clip(0.0, 1.0)
    warm_season_proxy = month.isin([4, 5, 6, 7, 8, 9]).astype(float)

    temp_above_10 = (temp80_c - 10.0).clip(lower=0.0)
    temp_above_20 = (temp80_c - 20.0).clip(lower=0.0)

    out["daylight_proxy"] = daylight_proxy
    out["night_proxy"] = night_proxy
    out["clear_sky_proxy"] = clear_sky_proxy
    out["warm_season_proxy"] = warm_season_proxy

    # Heat-flux-like proxy: warm + daytime + clear sky -> convection/mixing.
    out["convective_heat_flux_proxy"] = (
        daylight_proxy
        * clear_sky_proxy
        * (1.0 + 0.2 * warm_season_proxy)
        * np.log1p(temp_above_10)
    )

    out["strong_convective_heat_flux_proxy"] = (
        daylight_proxy
        * clear_sky_proxy
        * np.log1p(temp_above_20)
    )

    # Stable nocturnal boundary layer: clear night + weak wind + positive Ri.
    out["stable_night_proxy"] = (
        night_proxy
        * clear_sky_proxy
        * (ws80 < 6.0).astype(float)
        * (ri_120_80 > 0.05).astype(float)
    )

    out["very_stable_night_proxy"] = (
        night_proxy
        * clear_sky_proxy
        * (ws80 < 4.0).astype(float)
        * (ri_120_80 > 0.25).astype(float)
    )

    out["stable_night_x_shear"] = out["stable_night_proxy"] * vector_shear_120_80
    out["stable_night_x_ws120"] = out["stable_night_proxy"] * ws120

    out["convective_x_shear_120_80"] = (
        out["convective_heat_flux_proxy"] * vector_shear_120_80
    )
    out["convective_x_ws120"] = out["convective_heat_flux_proxy"] * ws120

    if "expected_power_proxy_120m" in out.columns:
        out["stable_night_x_expected_power_120m"] = (
            out["stable_night_proxy"] * out["expected_power_proxy_120m"]
        )
        out["convective_x_expected_power_120m"] = (
            out["convective_heat_flux_proxy"] * out["expected_power_proxy_120m"]
        )

    if "expected_power_proxy_rews" in out.columns:
        out["stable_night_x_expected_power_rews"] = (
            out["stable_night_proxy"] * out["expected_power_proxy_rews"]
        )
        out["convective_x_expected_power_rews"] = (
            out["convective_heat_flux_proxy"] * out["expected_power_proxy_rews"]
        )

    return out


def _add_momentum_flux_features(
    df: pd.DataFrame,
    ws10_col: str | None,
    ws80_col: str | None,
    ws120_col: str | None,
    ws180_col: str | None,
) -> pd.DataFrame:
    """
    Momentum / dynamic-pressure features.

    Physicist's idea: rho * v may help if v^3 helped.

    Adds:
    - rho * v: momentum per volume proxy;
    - rho * v^2: momentum flux / pressure-like proxy;
    - 0.5 * rho * v^2: dynamic pressure;
    - interactions with power curve / regimes.

    Does not replace wind_power_density = 0.5 * rho * v^3.
    """

    out = df.copy()

    # Prefer moist density if available, otherwise dry density.
    if "moist_air_density" in out.columns:
        rho = pd.to_numeric(out["moist_air_density"], errors="coerce")
        rho_name = "moist"
    elif "air_density" in out.columns:
        rho = pd.to_numeric(out["air_density"], errors="coerce")
        rho_name = "dry"
    else:
        return out

    speed_sources = [
        ("10m", ws10_col, None),
        ("80m", ws80_col, "power_curve_proxy"),
        ("120m", ws120_col, "power_curve_proxy_120m"),
        ("180m", ws180_col, "power_curve_proxy_180m"),
    ]

    for label, col, power_curve_col in speed_sources:
        if col is None or col not in out.columns:
            continue

        ws = pd.to_numeric(out[col], errors="coerce")
        prefix = f"momentum_{label}"

        out[f"{prefix}_rho_v_{rho_name}"] = rho * ws
        out[f"{prefix}_rho_v2_{rho_name}"] = rho * (ws ** 2)
        out[f"{prefix}_dynamic_pressure_{rho_name}"] = 0.5 * rho * (ws ** 2)

        out[f"{prefix}_rho_v2_ramp_{rho_name}"] = (
            out[f"{prefix}_rho_v2_{rho_name}"]
            * ((ws >= 3.0) & (ws < 12.0)).astype(float)
        )

        out[f"{prefix}_rho_v2_rated_{rho_name}"] = (
            out[f"{prefix}_rho_v2_{rho_name}"]
            * ((ws >= 12.0) & (ws < 25.0)).astype(float)
        )

        if power_curve_col is not None and power_curve_col in out.columns:
            out[f"{prefix}_dynamic_pressure_x_power_curve_{rho_name}"] = (
                out[f"{prefix}_dynamic_pressure_{rho_name}"]
                * out[power_curve_col]
            )

    if "rotor_equiv_ws" in out.columns:
        rews = pd.to_numeric(out["rotor_equiv_ws"], errors="coerce")

        out[f"momentum_rews_rho_v_{rho_name}"] = rho * rews
        out[f"momentum_rews_rho_v2_{rho_name}"] = rho * (rews ** 2)
        out[f"momentum_rews_dynamic_pressure_{rho_name}"] = 0.5 * rho * (rews ** 2)

        if "power_curve_proxy_rews" in out.columns:
            out[f"momentum_rews_dynamic_pressure_x_power_curve_{rho_name}"] = (
                out[f"momentum_rews_dynamic_pressure_{rho_name}"]
                * out["power_curve_proxy_rews"]
            )

    if "wind_vector_shear_120_80" in out.columns:
        shear = pd.to_numeric(out["wind_vector_shear_120_80"], errors="coerce")

        out[f"momentum_shear_120_80_rho_dv_{rho_name}"] = rho * shear
        out[f"momentum_shear_120_80_rho_dv2_{rho_name}"] = rho * (shear ** 2)

    return out

def _to_wind_dir_deg(raw: pd.Series) -> pd.Series:
    raw = pd.to_numeric(raw, errors="coerce")
    q99 = raw.dropna().quantile(0.99)

    # В ваших данных направление похоже на degrees / 1000:
    # 0.273 -> 273 degrees.
    return (raw * 1000.0) % 360.0


def _add_yaw_misalignment_features(
    df: pd.DataFrame,
    wind_dir_col: str | None,
    wind_speed_col: str | None,
    datetime_col: str,
    prefix: str,
    power_proxy_col: str | None = None,
    yaw_minutes: float = 10.0,
) -> pd.DataFrame:
    """
    Yaw / self-orientation loss features.

    Идея:
    - если направление ветра резко меняется, ВЭУ тратит время на поворот;
    - 10 минут поворота ~= 1/6 часа потенциальной просадки;
    - это не ручное вычитание из прогноза, а признаки для модели.
    """

    if (
        wind_dir_col is None
        or wind_speed_col is None
        or wind_dir_col not in df.columns
        or wind_speed_col not in df.columns
        or datetime_col not in df.columns
    ):
        return df

    out = df.copy()

    dt = pd.to_datetime(out[datetime_col], errors="coerce")

    # В ваших данных направление уже правильно трактуется как degrees / 1000.
    wind_dir_raw = pd.to_numeric(out[wind_dir_col], errors="coerce")
    wind_dir_deg = (wind_dir_raw * 1000.0) % 360.0

    ws = pd.to_numeric(out[wind_speed_col], errors="coerce")

    dir_by_datetime = (
        pd.DataFrame(
            {
                "__dt": dt,
                "__dir": wind_dir_deg,
            }
        )
        .dropna(subset=["__dt"])
        .drop_duplicates("__dt", keep="last")
        .set_index("__dt")["__dir"]
        .sort_index()
    )

    if dir_by_datetime.empty:
        return out

    lag1_dir = (dt - pd.Timedelta(hours=1)).map(dir_by_datetime)
    lag2_dir = (dt - pd.Timedelta(hours=2)).map(dir_by_datetime)
    lag3_dir = (dt - pd.Timedelta(hours=3)).map(dir_by_datetime)

    signed_change_1h = _angle_diff_deg(wind_dir_deg, lag1_dir)
    signed_change_2h = _angle_diff_deg(wind_dir_deg, lag2_dir)
    signed_change_3h = _angle_diff_deg(wind_dir_deg, lag3_dir)

    abs_change_1h = signed_change_1h.abs()
    abs_change_2h = signed_change_2h.abs()
    abs_change_3h = signed_change_3h.abs()

    out[f"{prefix}_dir_change_signed_1h"] = signed_change_1h
    out[f"{prefix}_dir_change_abs_1h"] = abs_change_1h
    out[f"{prefix}_dir_change_abs_2h"] = abs_change_2h
    out[f"{prefix}_dir_change_abs_3h"] = abs_change_3h

    for threshold in (15, 30, 45, 60, 90):
        out[f"{prefix}_turn_gt_{threshold}deg"] = (
            abs_change_1h >= threshold
        ).astype(float)

    # Чем выше ветер, тем потенциально больнее yaw-промах.
    out[f"{prefix}_change_abs_x_ws"] = abs_change_1h * ws
    out[f"{prefix}_change_abs_x_ws_sq"] = abs_change_1h * (ws ** 2)
    out[f"{prefix}_change_abs_x_ws_cube"] = abs_change_1h * (ws ** 3)

    # Рабочие зоны турбины.
    out[f"{prefix}_active_ws_zone"] = ((ws >= 3.0) & (ws <= 25.0)).astype(float)
    out[f"{prefix}_ramp_ws_zone"] = ((ws >= 3.0) & (ws < 12.0)).astype(float)
    out[f"{prefix}_rated_ws_zone"] = ((ws >= 12.0) & (ws <= 25.0)).astype(float)

    yaw_hour_fraction = yaw_minutes / 60.0

    # Насыщаем эффект: после 90 градусов считаем поворот уже большим.
    turn_intensity = (abs_change_1h / 90.0).clip(lower=0.0, upper=1.0)

    out[f"{prefix}_turn_intensity"] = turn_intensity
    out[f"{prefix}_yaw_loss_fraction_proxy"] = (
        yaw_hour_fraction
        * turn_intensity
        * out[f"{prefix}_active_ws_zone"]
    )

    out[f"{prefix}_yaw_loss_fraction_x_ramp"] = (
        out[f"{prefix}_yaw_loss_fraction_proxy"]
        * out[f"{prefix}_ramp_ws_zone"]
    )

    out[f"{prefix}_yaw_loss_fraction_x_rated"] = (
        out[f"{prefix}_yaw_loss_fraction_proxy"]
        * out[f"{prefix}_rated_ws_zone"]
    )

    if power_proxy_col is not None and power_proxy_col in out.columns:
        out[f"{prefix}_yaw_loss_mw_proxy"] = (
            out[power_proxy_col]
            * out[f"{prefix}_yaw_loss_fraction_proxy"]
        )

    # Частые смены направления за прошлые окна.
    change_by_datetime = (
        pd.DataFrame(
            {
                "__dt": dt,
                "__change": abs_change_1h,
            }
        )
        .dropna(subset=["__dt"])
        .drop_duplicates("__dt", keep="last")
        .set_index("__dt")["__change"]
        .sort_index()
    )

    turn30_by_datetime = (change_by_datetime >= 30.0).astype(float)
    turn60_by_datetime = (change_by_datetime >= 60.0).astype(float)

    for window_h in (3, 6, 12):
        out[f"{prefix}_change_sum_{window_h}h"] = dt.map(
            change_by_datetime.rolling(f"{window_h}h", min_periods=1).sum()
        )

        out[f"{prefix}_change_mean_{window_h}h"] = dt.map(
            change_by_datetime.rolling(f"{window_h}h", min_periods=1).mean()
        )

        out[f"{prefix}_change_max_{window_h}h"] = dt.map(
            change_by_datetime.rolling(f"{window_h}h", min_periods=1).max()
        )

        out[f"{prefix}_turn30_count_{window_h}h"] = dt.map(
            turn30_by_datetime.rolling(f"{window_h}h", min_periods=1).sum()
        )

        out[f"{prefix}_turn60_count_{window_h}h"] = dt.map(
            turn60_by_datetime.rolling(f"{window_h}h", min_periods=1).sum()
        )

    # Осцилляция: направление меняется туда-сюда, а не просто один раз повернулось.
    signed_by_datetime = (
        pd.DataFrame(
            {
                "__dt": dt,
                "__signed": signed_change_1h,
            }
        )
        .dropna(subset=["__dt"])
        .drop_duplicates("__dt", keep="last")
        .set_index("__dt")["__signed"]
        .sort_index()
    )

    sign_now = np.sign(signed_by_datetime)
    sign_prev = sign_now.shift(1)

    oscillation = (
        (sign_now != 0)
        & (sign_prev != 0)
        & (sign_now != sign_prev)
    ).astype(float)

    for window_h in (3, 6, 12):
        out[f"{prefix}_oscillation_count_{window_h}h"] = dt.map(
            oscillation.rolling(f"{window_h}h", min_periods=1).sum()
        )

    return out


def _add_time_lag_features(
    df: pd.DataFrame,
    value_col: str,
    prefix: str,
    datetime_col: str,
    lag_hours: tuple[int, ...] = LAG_HOURS,
    diff_hours: tuple[int, ...] = DIFF_HOURS,
    rolling_mean_hours: tuple[int, ...] = ROLLING_MEAN_HOURS,
    rolling_std_hours: tuple[int, ...] = ROLLING_STD_HOURS,
) -> pd.DataFrame:
    """
    Добавляет time-aware лаги, diff и rolling-фичи.

    Важно:
    - считаем по datetime, а не по порядку строк;
    - не используем target;
    - rolling считается только по прошлым значениям, без текущего часа.

    Примеры новых колонок:
        ws80_lag_1h
        ws80_diff_3h
        ws80_roll_mean_24h
        ws80_roll_std_72h
    """

    if datetime_col not in df.columns or value_col not in df.columns:
        return df

    out = df.copy()

    dt = pd.to_datetime(out[datetime_col], errors="coerce")
    value = pd.to_numeric(out[value_col], errors="coerce")

    value_by_datetime = (
        pd.DataFrame(
            {
                "__datetime": dt,
                "__value": value,
            }
        )
        .dropna(subset=["__datetime"])
        .drop_duplicates("__datetime", keep="last")
        .set_index("__datetime")["__value"]
        .sort_index()
    )

    if value_by_datetime.empty:
        return out

    # -------------------------
    # Lag features
    # -------------------------
    for lag_h in lag_hours:
        lagged_datetime = dt - pd.Timedelta(hours=lag_h)
        out[f"{prefix}_lag_{lag_h}h"] = lagged_datetime.map(value_by_datetime)

    # -------------------------
    # Diff features
    # current - lag
    # -------------------------
    for diff_h in diff_hours:
        lag_col = f"{prefix}_lag_{diff_h}h"

        if lag_col not in out.columns:
            lagged_datetime = dt - pd.Timedelta(hours=diff_h)
            out[lag_col] = lagged_datetime.map(value_by_datetime)

        out[f"{prefix}_diff_{diff_h}h"] = value - out[lag_col]

    # -------------------------
    # Rolling features
    # Только прошлые значения: shift(1)
    # -------------------------
    shifted = value_by_datetime.shift(1)

    for window_h in rolling_mean_hours:
        roll_mean = shifted.rolling(
            f"{window_h}h",
            min_periods=1,
        ).mean()

        out[f"{prefix}_roll_mean_{window_h}h"] = dt.map(roll_mean)

    for window_h in rolling_std_hours:
        roll_std = shifted.rolling(
            f"{window_h}h",
            min_periods=2,
        ).std()

        out[f"{prefix}_roll_std_{window_h}h"] = dt.map(roll_std)

    return out


def _add_past_window_sum_features(
    df: pd.DataFrame,
    value_col: str,
    prefix: str,
    datetime_col: str,
    windows_hours: tuple[int, ...] = (3, 6, 12, 24),
) -> pd.DataFrame:
    """
    Rolling sum over previous hours only.

    Examples:
        rain_sum_prev_6h
        showers_sum_prev_12h
        snowfall_sum_prev_24h

    Важно:
    shift(1) означает, что текущий час не включаем,
    смотрим только на прошлые часы.
    """

    if datetime_col not in df.columns or value_col not in df.columns:
        return df

    out = df.copy()

    dt = pd.to_datetime(out[datetime_col], errors="coerce")
    value = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)

    value_by_datetime = (
        pd.DataFrame(
            {
                "__datetime": dt,
                "__value": value,
            }
        )
        .dropna(subset=["__datetime"])
        .drop_duplicates("__datetime", keep="last")
        .set_index("__datetime")["__value"]
        .sort_index()
    )

    if value_by_datetime.empty:
        return out

    shifted = value_by_datetime.shift(1)

    for window_h in windows_hours:
        rolling_sum = shifted.rolling(
            f"{window_h}h",
            min_periods=1,
        ).sum()

        out[f"{prefix}_sum_prev_{window_h}h"] = dt.map(rolling_sum)

    return out


def _add_thermal_derating_features(
    df: pd.DataFrame,
    temp_col: str | None,
    ws80_col: str | None,
    ws120_col: str | None,
    ws180_col: str | None,
) -> pd.DataFrame:
    """
    Transformer / electrical equipment high-temperature derating proxy.

    Идея:
    - высокая температура сама по себе не обязательно проблема;
    - проблема = высокая температура + высокая ожидаемая мощность / нагрузка;
    - добавляем proxy-фичи, но не вычитаем руками из прогноза.
    """

    if temp_col is None or temp_col not in df.columns:
        return df

    out = df.copy()

    temp = pd.to_numeric(out[temp_col], errors="coerce")
    temp_median = temp.median(skipna=True)

    # C / K auto-detect
    if pd.notna(temp_median) and temp_median > 150:
        temp_c = temp - 273.15
    else:
        temp_c = temp

    ws80 = _numeric_series(out, ws80_col, default=np.nan)
    ws120 = _numeric_series(out, ws120_col, default=np.nan)
    ws180 = _numeric_series(out, ws180_col, default=np.nan)

    # Температурные пороги: лучше дать модели несколько ступеней.
    out["temp_c_inferred"] = temp_c
    out["temp_above_20c"] = (temp_c - 20.0).clip(lower=0.0)
    out["temp_above_25c"] = (temp_c - 25.0).clip(lower=0.0)
    out["temp_above_30c"] = (temp_c - 30.0).clip(lower=0.0)
    out["temp_above_35c"] = (temp_c - 35.0).clip(lower=0.0)

    out["is_hot_25c"] = (temp_c >= 25.0).astype(float)
    out["is_hot_30c"] = (temp_c >= 30.0).astype(float)
    out["is_hot_35c"] = (temp_c >= 35.0).astype(float)

    # Нагрузка: лучше брать физический proxy мощности, а не просто wind speed.
    if "available_capacity_mw" in out.columns:
        capacity = out["available_capacity_mw"]
    else:
        capacity = pd.Series(INSTALLED_CAPACITY_MW, index=out.index)

    candidate_power_cols = [
        "expected_power_proxy_rews",
        "expected_power_proxy_120m",
        "expected_power_proxy",
        "expected_power_density_adj_rews",
        "expected_power_density_adj",
    ]

    load_proxy = pd.Series(np.nan, index=out.index)

    for col in candidate_power_cols:
        if col in out.columns:
            current = pd.to_numeric(out[col], errors="coerce")
            load_proxy = load_proxy.fillna(current)

    # fallback, если power proxies почему-то ещё не созданы
    if load_proxy.isna().all():
        load_proxy = ((ws120.clip(3.0, 12.0) - 3.0) / 9.0) ** 3
        load_proxy = load_proxy.clip(0.0, 1.0) * capacity

    load_ratio = (load_proxy / (capacity + 1e-6)).clip(0.0, 1.5)

    out["thermal_load_proxy_mw"] = load_proxy
    out["thermal_load_ratio_proxy"] = load_ratio

    # Электрические потери ближе к load^2, поэтому даём и квадрат.
    out["thermal_load_ratio_sq"] = load_ratio ** 2

    # Главные interaction-фичи.
    out["hot25_x_load_ratio"] = out["temp_above_25c"] * load_ratio
    out["hot30_x_load_ratio"] = out["temp_above_30c"] * load_ratio
    out["hot35_x_load_ratio"] = out["temp_above_35c"] * load_ratio

    out["hot25_x_load_ratio_sq"] = out["temp_above_25c"] * (load_ratio ** 2)
    out["hot30_x_load_ratio_sq"] = out["temp_above_30c"] * (load_ratio ** 2)
    out["hot35_x_load_ratio_sq"] = out["temp_above_35c"] * (load_ratio ** 2)

    # Режим: жарко и модель ожидает высокую генерацию.
    out["is_hot_and_high_load"] = (
        (temp_c >= 30.0) & (load_ratio >= 0.70)
    ).astype(float)

    out["is_very_hot_and_high_load"] = (
        (temp_c >= 35.0) & (load_ratio >= 0.70)
    ).astype(float)

    # Через ветер: при слабом ветре жара не так важна, при rated-zone важна.
    out["hot30_x_ws80_cube"] = out["temp_above_30c"] * (ws80 ** 3)
    out["hot30_x_ws120_cube"] = out["temp_above_30c"] * (ws120 ** 3)
    out["hot30_x_ws180_cube"] = out["temp_above_30c"] * (ws180 ** 3)

    out["hot30_x_rated_80m"] = (
        out["temp_above_30c"]
        * ((ws80 >= 12.0) & (ws80 < 25.0)).astype(float)
    )

    out["hot30_x_rated_120m"] = (
        out["temp_above_30c"]
        * ((ws120 >= 12.0) & (ws120 < 25.0)).astype(float)
    )

    # Возможное охлаждение ветром у наружного оборудования:
    # даём модели понять, что жарко + штиль отличается от жарко + ветер.
    ws10_col = _find_wind_speed_col(out, 10)
    ws10 = _numeric_series(out, ws10_col, default=np.nan)

    out["hot30_x_low_cooling_wind10"] = (
        out["temp_above_30c"] * (ws10 < 3.0).astype(float)
    )

    out["hot30_x_cooling_wind10"] = out["temp_above_30c"] * ws10

    return out


@dataclass
class ModelPreprocessor:
    """
    Минимальный препроцессор для табличных моделей.

    Что делает:
    1. Переименовывает битые/неудобные колонки.
    2. Парсит datetime.
    3. Добавляет 2 демонстрационные фичи: hour_sin и hour_cos.
    4. Убирает target и сырой datetime из признаков.
    5. Запоминает список признаков на train и так же применяет его на valid/test.

    Важно:
    Фичи добавляются ТОЛЬКО в методе add_features().
    Физику команды удобно расширять именно его.
    """

    target_col: str = TARGET_COL
    datetime_col: str = DATETIME_COL
    add_demo_features: bool = True
    feature_flags: dict[str, bool] = field(
        default_factory=lambda: DEFAULT_FEATURE_FLAGS.copy()
    )
    feature_columns_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "ModelPreprocessor":
        prepared = self._prepare_dataframe(df)

        ignored = {self.target_col, self.datetime_col}
        feature_columns = [
            col for col in prepared.columns
            if col not in ignored
        ]

        # Оставляем только числовые признаки.
        # Так мы не ловим внезапные ошибки XGBoost из-за object/string колонок.
        numeric_feature_columns = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(prepared[col]):
                numeric_feature_columns.append(col)

        self.feature_columns_ = numeric_feature_columns
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns_:
            raise RuntimeError(
                "Preprocessor is not fitted. Call preprocessor.fit(df) first."
            )

        prepared = self._prepare_dataframe(df)

        # Если в valid/test нет какой-то train-колонки — создаём её как NaN.
        # Это лучше, чем падать в последний момент.
        for col in self.feature_columns_:
            if col not in prepared.columns:
                prepared[col] = np.nan

        X = prepared[self.feature_columns_].copy()

        # На всякий случай приводим всё к числам.
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        return X.replace([np.inf, -np.inf], np.nan)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in dataframe."
            )

        return pd.to_numeric(df[self.target_col], errors="coerce")

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()

        prepared = prepared.rename(columns=COLUMN_RENAME_MAP)

        if self.datetime_col in prepared.columns:
            prepared[self.datetime_col] = pd.to_datetime(
                prepared[self.datetime_col],
                errors="coerce",
            )

        prepared = self.add_features(prepared)

        return prepared

    def _flag(self, name: str) -> bool:
        if self.feature_flags is None:
            return False
        return bool(self.feature_flags.get(name, False))

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ЕДИНСТВЕННОЕ МЕСТО для добавления новых признаков.

        Сейчас добавлены только 2 примитивные демонстрационные фичи:
        - hour_sin
        - hour_cos

        Они кодируют час суток как цикл:
        23:00 и 00:00 становятся близкими, а не максимально далёкими.

        Физик команды может добавлять новые признаки ниже по такому же принципу:
            df["new_feature"] = ...

        Главное правило:
        не использовать target 'Выработка' при создании признаков для valid/test.
        """

        if not self.add_demo_features:
            return df

        df = df.copy()

        # OSPREY checking needed
        # :NOTE: should we consider 80m --> 84m ?
        # :NOTE: !wake effect - really needed
        # ------------------------------------------------------------
        # 1. Time features
        # ------------------------------------------------------------
        if self._flag("time"):
            if "hour_of_day" in df.columns:
                hour = pd.to_numeric(df["hour_of_day"], errors="coerce")
            elif self.datetime_col in df.columns:
                hour = df[self.datetime_col].dt.hour
            else:
                hour = None

            if hour is not None:
                df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

                if self._flag("time_interactions"):
                    df["is_evening_20_23"] = (
                        (hour >= 20) & (hour <= 23)
                    ).astype(float)

                    df["is_late_evening_21_23"] = (
                        (hour >= 21) & (hour <= 23)
                    ).astype(float)

                    df["is_night_0_6"] = (
                        (hour >= 0) & (hour <= 6)
                    ).astype(float)

                    df["is_good_midday_10_12"] = (
                        (hour >= 10) & (hour <= 12)
                    ).astype(float)

            if self.datetime_col in df.columns:
                month = df[self.datetime_col].dt.month
                dayofyear = df[self.datetime_col].dt.dayofyear

                df["month_sin"] = np.sin(2 * np.pi * month / 12)
                df["month_cos"] = np.cos(2 * np.pi * month / 12)

                df["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
                df["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)

        # ------------------------------------------------------------
        # 2. Availability features
        # ------------------------------------------------------------
        if self._flag("availability"):
            if "repair_count" in df.columns:
                repair_count = (
                    pd.to_numeric(df["repair_count"], errors="coerce")
                    .fillna(0.0)
                    .clip(0, N_TURBINES)
                )
            else:
                repair_count = pd.Series(0.0, index=df.index)

            df["available_turbines"] = N_TURBINES - repair_count
            df["availability_ratio"] = df["available_turbines"] / N_TURBINES
            df["available_capacity_mw"] = (
                    df["available_turbines"] * TURBINE_CAPACITY_MW
            )

        # ------------------------------------------------------------
        # 3. Hub-height wind speed
        # ------------------------------------------------------------
        ws10_col = _find_wind_speed_col(df, 10)
        ws80_col = _find_wind_speed_col(df, 80)
        ws120_col = _find_wind_speed_col(df, 120)
        ws180_col = _find_wind_speed_col(df, 180)
        gust10_col = _find_wind_gust_col(df, 10)

        if ws80_col is not None:
            ws80 = pd.to_numeric(df[ws80_col], errors="coerce")

            if self._flag("power_curve"):
                df["ws80_sq"] = ws80 ** 2
                df["ws80_cube"] = ws80 ** 3

                # Approximate Siemens Gamesa style power-curve proxy:
                # 0 below 3 m/s, cubic growth until 12 m/s, flat after.
                power_curve_proxy = ((ws80.clip(3.0, 12.0) - 3.0) / 9.0) ** 3
                df["power_curve_proxy"] = power_curve_proxy.clip(0.0, 1.0)

                if "available_capacity_mw" in df.columns:
                    df["expected_power_proxy"] = (
                            df["power_curve_proxy"] * df["available_capacity_mw"]
                    )
                else:
                    df["expected_power_proxy"] = (
                            df["power_curve_proxy"] * INSTALLED_CAPACITY_MW
                    )

        # ------------------------------------------------------------
        # 3.1 Multi-height wind physics: 120m / 180m
        # ------------------------------------------------------------

        if self._flag("multi_height_power"):
            for height, col in ((120, ws120_col), (180, ws180_col)):
                if col is None:
                    continue

                ws_h = pd.to_numeric(df[col], errors="coerce")
                prefix = f"ws{height}"

                df[f"{prefix}_sq"] = ws_h ** 2
                df[f"{prefix}_cube"] = ws_h ** 3

                power_curve_h = ((ws_h.clip(3.0, 12.0) - 3.0) / 9.0) ** 3
                df[f"power_curve_proxy_{height}m"] = power_curve_h.clip(0.0, 1.0)

                if "available_capacity_mw" in df.columns:
                    df[f"expected_power_proxy_{height}m"] = (
                            df[f"power_curve_proxy_{height}m"]
                            * df["available_capacity_mw"]
                    )
                else:
                    df[f"expected_power_proxy_{height}m"] = (
                            df[f"power_curve_proxy_{height}m"]
                            * INSTALLED_CAPACITY_MW
                    )

        # ------------------------------------------------------------
        # 3.2 Wind shear features
        # ------------------------------------------------------------
        if self._flag("wind_shear"):
            ws10 = _numeric_series(df, ws10_col, default=np.nan)
            ws80 = _numeric_series(df, ws80_col, default=np.nan)
            ws120 = _numeric_series(df, ws120_col, default=np.nan)
            ws180 = _numeric_series(df, ws180_col, default=np.nan)

            df["wind_shear_80_10"] = ws80 - ws10
            df["wind_shear_120_80"] = ws120 - ws80
            df["wind_shear_180_80"] = ws180 - ws80
            df["wind_shear_180_120"] = ws180 - ws120

            df["wind_shear_ratio_80_10"] = _safe_ratio(ws80, ws10)
            df["wind_shear_ratio_120_80"] = _safe_ratio(ws120, ws80)
            df["wind_shear_ratio_180_80"] = _safe_ratio(ws180, ws80)
            df["wind_shear_ratio_180_120"] = _safe_ratio(ws180, ws120)

            valid_80_120 = (ws80 > 0.5) & (ws120 > 0.5)
            alpha_120_80 = pd.Series(np.nan, index=df.index)
            alpha_120_80.loc[valid_80_120] = (
                np.log(ws120.loc[valid_80_120] / ws80.loc[valid_80_120])
                / np.log(120.0 / 80.0)
            )

            valid_80_180 = (ws80 > 0.5) & (ws180 > 0.5)
            alpha_180_80 = pd.Series(np.nan, index=df.index)
            alpha_180_80.loc[valid_80_180] = (
                np.log(ws180.loc[valid_80_180] / ws80.loc[valid_80_180])
                / np.log(180.0 / 80.0)
            )

            df["wind_shear_alpha_120_80"] = alpha_120_80
            df["wind_shear_alpha_180_80"] = alpha_180_80

        # ------------------------------------------------------------
        # 3.3 Wind regime features: cut-in / ramp / rated / storm
        # ------------------------------------------------------------
        if self._flag("wind_regime"):
            ws80 = _numeric_series(df, ws80_col, default=np.nan)
            ws120 = _numeric_series(df, ws120_col, default=np.nan)
            ws180 = _numeric_series(df, ws180_col, default=np.nan)
            gust10 = _numeric_series(df, gust10_col, default=np.nan)

            # Approximate turbine zones.
            # Cut-in около 3 м/с, rated около 12 м/с, cut-out/storm около 25 м/с.
            df["is_below_cut_in_80m"] = (ws80 < 3.0).astype(float)
            df["is_ramp_zone_80m"] = ((ws80 >= 3.0) & (ws80 < 12.0)).astype(float)
            df["is_rated_zone_80m"] = ((ws80 >= 12.0) & (ws80 < 25.0)).astype(float)
            df["is_storm_zone_80m"] = (ws80 >= 25.0).astype(float)

            df["is_below_cut_in_120m"] = (ws120 < 3.0).astype(float)
            df["is_ramp_zone_120m"] = ((ws120 >= 3.0) & (ws120 < 12.0)).astype(float)
            df["is_rated_zone_120m"] = ((ws120 >= 12.0) & (ws120 < 25.0)).astype(float)
            df["is_storm_zone_120m"] = (ws120 >= 25.0).astype(float)

            df["is_high_wind_180m"] = (ws180 >= 15.0).astype(float)
            df["is_extreme_wind_180m"] = (ws180 >= 22.0).astype(float)

            # Gust-driven stress.
            df["is_extreme_gust10"] = (gust10 >= 18.0).astype(float)
            df["gust10_to_ws80_ratio_regime"] = _safe_ratio(gust10, ws80)
            df["gust10_excess_over_ws80_regime"] = (gust10 - ws80).clip(lower=0.0)

            # Interaction: high gusts during already strong upper wind.
            df["storm_risk_proxy"] = (
                df["is_extreme_gust10"]
                * ((ws120 >= 12.0) | (ws180 >= 15.0)).astype(float)
            )

        # ------------------------------------------------------------
        # 3.4 Ramp-zone features: painful 6-12 m/s range
        # ------------------------------------------------------------
        if self._flag("ramp_zone_features"):
            ws80 = _numeric_series(df, ws80_col, default=np.nan)
            ws120 = _numeric_series(df, ws120_col, default=np.nan)
            ws180 = _numeric_series(df, ws180_col, default=np.nan)

            # Основные проблемные зоны по debug:
            # wind_speed_120m 6-9 и 9-12 дают максимальную ошибку.
            df["is_ws120_6_9"] = ((ws120 >= 6.0) & (ws120 < 9.0)).astype(float)
            df["is_ws120_9_12"] = ((ws120 >= 9.0) & (ws120 < 12.0)).astype(float)
            df["is_ws120_6_12"] = ((ws120 >= 6.0) & (ws120 < 12.0)).astype(float)

            # Piecewise линейные куски, чтобы модель лучше ловила форму power curve.
            df["ws120_piece_3_6"] = (ws120.clip(3.0, 6.0) - 3.0).clip(lower=0.0)
            df["ws120_piece_6_9"] = (ws120.clip(6.0, 9.0) - 6.0).clip(lower=0.0)
            df["ws120_piece_9_12"] = (ws120.clip(9.0, 12.0) - 9.0).clip(lower=0.0)

            df["ws80_piece_3_6"] = (ws80.clip(3.0, 6.0) - 3.0).clip(lower=0.0)
            df["ws80_piece_6_9"] = (ws80.clip(6.0, 9.0) - 6.0).clip(lower=0.0)
            df["ws80_piece_9_12"] = (ws80.clip(9.0, 12.0) - 9.0).clip(lower=0.0)

            # Насколько 120м уходит от 80м именно в ramp-zone.
            df["ws120_minus_ws80_in_ramp"] = (
                    (ws120 - ws80) * df["is_ws120_6_12"]
            )

            df["ws180_minus_ws120_in_ramp"] = (
                    (ws180 - ws120) * df["is_ws120_6_12"]
            )

            # Interaction с физическими proxy.
            if "expected_power_proxy_120m" in df.columns:
                df["expected_power_120m_x_ws120_6_12"] = (
                        df["expected_power_proxy_120m"] * df["is_ws120_6_12"]
                )

            if "wind_power_density_120m" in df.columns:
                df["wind_power_density_120m_x_ramp"] = (
                        df["wind_power_density_120m"] * df["is_ws120_6_12"]
                )
        # ------------------------------------------------------------
        # 4. Air density
        # ------------------------------------------------------------
        temp_col = _find_temperature_col(df)
        pressure_col = _find_pressure_col(df)

        if self._flag("air_density") and temp_col is not None and pressure_col is not None:
            temp = pd.to_numeric(df[temp_col], errors="coerce")
            pressure = pd.to_numeric(df[pressure_col], errors="coerce")

            temp_median = temp.median(skipna=True)
            pressure_median = pressure.median(skipna=True)

            if pd.notna(temp_median):
                temp_k = temp if temp_median > 150 else temp + 273.15
            else:
                temp_k = temp + 273.15

            if pd.notna(pressure_median):
                pressure_pa = pressure * 100.0 if pressure_median < 2000 else pressure
            else:
                pressure_pa = pressure

            df["air_density"] = pressure_pa / (R_DRY_AIR * temp_k)

        # ------------------------------------------------------------
        # 4.05 Rotor-equivalent wind speed proxy
        # ------------------------------------------------------------
        if self._flag("rotor_equivalent_wind"):
            ws10 = _numeric_series(df, ws10_col, default=np.nan)
            ws80 = _numeric_series(df, ws80_col, default=np.nan)
            ws120 = _numeric_series(df, ws120_col, default=np.nan)
            ws180 = _numeric_series(df, ws180_col, default=np.nan)

            # Rough rotor-disk proxy.
            # Hub is close to 84m, rotor spans roughly 18-150m.
            # 80m and 120m get most weight; 10m and 180m are weak boundary signals.
            rews_cube = (
                    0.10 * (ws10 ** 3)
                    + 0.45 * (ws80 ** 3)
                    + 0.40 * (ws120 ** 3)
                    + 0.05 * (ws180 ** 3)
            )

            df["rotor_equiv_ws_cube"] = rews_cube
            df["rotor_equiv_ws"] = np.cbrt(rews_cube.clip(lower=0.0))

            rews = df["rotor_equiv_ws"]

            rews_power_curve = ((rews.clip(3.0, 12.0) - 3.0) / 9.0) ** 3
            df["power_curve_proxy_rews"] = rews_power_curve.clip(0.0, 1.0)

            if "available_capacity_mw" in df.columns:
                df["expected_power_proxy_rews"] = (
                        df["power_curve_proxy_rews"] * df["available_capacity_mw"]
                )
            else:
                df["expected_power_proxy_rews"] = (
                        df["power_curve_proxy_rews"] * INSTALLED_CAPACITY_MW
                )

            if "air_density" in df.columns:
                df["wind_power_density_rews"] = (
                        0.5 * df["air_density"] * df["rotor_equiv_ws_cube"]
                )
            if "expected_power_proxy_rews" in df.columns and "air_density" in df.columns:
                df["expected_power_density_adj_rews"] = (
                        df["expected_power_proxy_rews"]
                        * df["air_density"]
                        / STANDARD_AIR_DENSITY
                )

        # ------------------------------------------------------------
        # 4.075 Moist air density proxy
        # ------------------------------------------------------------
        if self._flag("moist_air_density"):
            df = _add_moist_air_density_features(
                df=df,
                temp_col=temp_col,
                pressure_col=pressure_col,
            )
        # ------------------------------------------------------------
        # 4.08 Boundary-layer stability / heat-flux proxy
        # ------------------------------------------------------------
        if self._flag("boundary_layer_stability"):
            df = _add_boundary_layer_stability_features(
                df=df,
                temp_col=temp_col,
                ws80_col=ws80_col,
                ws120_col=ws120_col,
                ws180_col=ws180_col,
                datetime_col=self.datetime_col,
            )

        # ------------------------------------------------------------
        # 4.09 Momentum / dynamic-pressure proxy
        # ------------------------------------------------------------
        if self._flag("momentum_flux"):
            df = _add_momentum_flux_features(
                df=df,
                ws10_col=ws10_col,
                ws80_col=ws80_col,
                ws120_col=ws120_col,
                ws180_col=ws180_col,
            )

        # ------------------------------------------------------------
        # 4.1 Icing / cold wet weather features
        # ------------------------------------------------------------
        if self._flag("icing"):
            temp80 = _numeric_series(df, temp_col, default=np.nan)

            if "temperature_120m" in df.columns:
                temp120 = pd.to_numeric(df["temperature_120m"], errors="coerce")
            else:
                temp120 = pd.Series(np.nan, index=df.index)

            rain = _numeric_series(df, "rain", default=0.0).fillna(0.0)
            showers = _numeric_series(df, "showers", default=0.0).fillna(0.0)
            snowfall = _numeric_series(df, "snowfall", default=0.0).fillna(0.0)
            cloud_low = _numeric_series(df, "cloud_cover_low", default=0.0).fillna(0.0)

            precip_total = rain + showers + snowfall

            df["precip_total"] = precip_total
            df["has_precip"] = (precip_total > 0.0).astype(float)
            df["has_snowfall"] = (snowfall > 0.0).astype(float)

            # Температурные зоны риска.
            df["is_freezing_temp_80m"] = (
                (temp80 >= -6.0) & (temp80 <= 2.0)
            ).astype(float)

            df["is_near_zero_temp_80m"] = (
                (temp80 >= -2.0) & (temp80 <= 2.0)
            ).astype(float)

            df["is_deep_cold_80m"] = (temp80 < -10.0).astype(float)

            # Мокрый холод — один из самых подозрительных режимов.
            df["is_wet_freezing_80m"] = (
                (temp80 >= -4.0)
                & (temp80 <= 2.0)
                & (precip_total > 0.0)
            ).astype(float)

            # Снег при отрицательных температурах.
            df["icing_risk_snow"] = (
                (temp80 >= -8.0)
                & (temp80 <= 1.0)
                & (snowfall > 0.0)
            ).astype(float)

            # Низкая облачность + влажная/снежная погода + около нуля.
            df["icing_risk_cloud_precip"] = (
                df["is_freezing_temp_80m"]
                * (precip_total > 0.0).astype(float)
                * (cloud_low > 0.05).astype(float)
            )

            # Непрерывный индекс риска, не только 0/1.
            df["icing_risk_score"] = (
                df["is_freezing_temp_80m"]
                * (1.0 + precip_total)
                * (1.0 + cloud_low)
            )

            # Вертикальный температурный градиент.
            df["temp_gradient_120_80"] = temp120 - temp80

        # ------------------------------------------------------------
        # 4.2 Precipitation stress features
        # ------------------------------------------------------------
        if self._flag("precipitation_stress"):
            rain = _numeric_series(df, "rain", default=0.0).fillna(0.0)
            showers = _numeric_series(df, "showers", default=0.0).fillna(0.0)
            snowfall = _numeric_series(df, "snowfall", default=0.0).fillna(0.0)

            ws80 = _numeric_series(df, ws80_col, default=np.nan)
            ws120 = _numeric_series(df, ws120_col, default=np.nan)

            df["rain_plus_showers"] = rain + showers
            df["precip_stress_score"] = rain + showers + 0.5 * snowfall

            df["is_any_rain"] = (rain > 0.0).astype(float)
            df["is_medium_rain"] = ((rain > 0.1) & (rain <= 1.0)).astype(float)
            df["is_heavy_rain"] = (rain > 1.0).astype(float)

            df["is_any_showers"] = (showers > 0.0).astype(float)
            df["is_medium_showers"] = ((showers > 0.1) & (showers <= 1.0)).astype(float)
            df["is_heavy_showers"] = (showers > 1.0).astype(float)

            # Влажная погода × скорость ветра.
            df["precip_stress_x_ws80"] = df["precip_stress_score"] * ws80
            df["precip_stress_x_ws120"] = df["precip_stress_score"] * ws120

            # Самая подозрительная зона: дождь/ливни + ramp-zone 6-12 м/с.
            df["wet_ramp_120m"] = (
                df["precip_stress_score"]
                * ((ws120 >= 6.0) & (ws120 < 12.0)).astype(float)
            )

            if "expected_power_proxy_120m" in df.columns:
                df["expected_power_120m_x_precip"] = (
                    df["expected_power_proxy_120m"] * df["precip_stress_score"]
                )

            if "is_evening_20_23" in df.columns:
                df["evening_x_precip_stress"] = (
                    df["is_evening_20_23"] * df["precip_stress_score"]
                )

                df["evening_x_showers"] = (
                    df["is_evening_20_23"] * df["is_any_showers"]
                )
        # ------------------------------------------------------------
        # 5. Wind power density
        # ------------------------------------------------------------
        if (
                self._flag("wind_power_density")
                and "air_density" in df.columns
                and "ws80_cube" in df.columns
        ):
            df["wind_power_density"] = 0.5 * df["air_density"] * df["ws80_cube"]

        # ------------------------------------------------------------
        # 5.1 Multi-height wind power density
        # ------------------------------------------------------------
        if self._flag("multi_height_power") and "air_density" in df.columns:
            for height in (120, 180):
                cube_col = f"ws{height}_cube"

                if cube_col in df.columns:
                    df[f"wind_power_density_{height}m"] = (
                        0.5 * df["air_density"] * df[cube_col]
                    )

        # ------------------------------------------------------------
        # 5.2 Gust features
        # ------------------------------------------------------------
        gust10_col = _find_wind_gust_col(df, 10)

        if self._flag("gust_features") and gust10_col is not None:
            gust10 = pd.to_numeric(df[gust10_col], errors="coerce")

            df["gust10_sq"] = gust10 ** 2
            df["gust10_cube"] = gust10 ** 3

            if ws80_col is not None:
                ws80_for_gust = pd.to_numeric(df[ws80_col], errors="coerce")

                df["gust10_to_ws80_ratio"] = gust10 / (ws80_for_gust + 1e-6)
                df["gust10_minus_ws80"] = gust10 - ws80_for_gust
                df["gust10_excess_over_ws80"] = (
                    gust10 - ws80_for_gust
                ).clip(lower=0.0)

                df["gust10_x_ws80"] = gust10 * ws80_for_gust

            if "air_density" in df.columns:
                df["gust10_power_density"] = (
                    0.5 * df["air_density"] * df["gust10_cube"]
                )

        # ------------------------------------------------------------
        # 6. Density-adjusted expected power
        # ------------------------------------------------------------
        if (
                self._flag("density_adjusted_power")
                and "expected_power_proxy" in df.columns
                and "air_density" in df.columns
        ):
            df["expected_power_density_adj"] = (
                    df["expected_power_proxy"]
                    * df["air_density"]
                    / STANDARD_AIR_DENSITY
            )

        # ------------------------------------------------------------
        # 6.1 Transformer / electrical equipment thermal derating
        # ------------------------------------------------------------
        if self._flag("thermal_derating"):
            df = _add_thermal_derating_features(
                df=df,
                temp_col=temp_col,
                ws80_col=ws80_col,
                ws120_col=ws120_col,
                ws180_col=ws180_col,
            )

        # ------------------------------------------------------------
        # 7. Wind direction features: 80m / 120m / 180m + veer
        # ------------------------------------------------------------
        if self._flag("wind_direction"):
            wind_dir_deg_by_height: dict[int, pd.Series] = {}

            for height in (80, 120, 180):
                wind_dir_col = _find_wind_direction_col(df, height)
                wind_speed_col = _find_wind_speed_col(df, height)

                if wind_dir_col is None:
                    continue

                wind_dir_raw = pd.to_numeric(df[wind_dir_col], errors="coerce")

                # В данных направление ветра хранится как degrees / 1000.
                # Например:
                #   0.273 -> 273 degrees
                wind_dir_deg = (wind_dir_raw * 1000.0) % 360.0
                wind_dir_rad = np.deg2rad(wind_dir_deg)

                wind_dir_deg_by_height[height] = wind_dir_deg

                df[f"wind_dir_{height}m_sin"] = np.sin(wind_dir_rad)
                df[f"wind_dir_{height}m_cos"] = np.cos(wind_dir_rad)

                df[f"wind_sector_{height}m_8"] = (
                    wind_dir_deg // 45.0
                ).astype(float)

                df[f"wind_sector_{height}m_16"] = (
                    wind_dir_deg // 22.5
                ).astype(float)

                # Speed × direction components.
                # Это не строго физическая u/v-конвенция, но для ML даёт направленный ветер.
                if wind_speed_col is not None:
                    ws_h = pd.to_numeric(df[wind_speed_col], errors="coerce")

                    df[f"wind_u_{height}m"] = ws_h * df[f"wind_dir_{height}m_sin"]
                    df[f"wind_v_{height}m"] = ws_h * df[f"wind_dir_{height}m_cos"]

                    cube_col = f"ws{height}_cube"
                    if height == 80:
                        cube_col = "ws80_cube"

                    if cube_col in df.columns:
                        df[f"{cube_col}_x_dir_sin"] = (
                            df[cube_col] * df[f"wind_dir_{height}m_sin"]
                        )
                        df[f"{cube_col}_x_dir_cos"] = (
                            df[cube_col] * df[f"wind_dir_{height}m_cos"]
                        )

                # Power curve × sector.
                if height == 80:
                    power_curve_col = "power_curve_proxy"
                else:
                    power_curve_col = f"power_curve_proxy_{height}m"

                if power_curve_col in df.columns:
                    df[f"{power_curve_col}_x_sector_8"] = (
                        df[power_curve_col] * df[f"wind_sector_{height}m_8"]
                    )
                    df[f"{power_curve_col}_x_sector_16"] = (
                        df[power_curve_col] * df[f"wind_sector_{height}m_16"]
                    )

            # --------------------------------------------------------
            # Wind veer: direction change with height
            # --------------------------------------------------------
            if 80 in wind_dir_deg_by_height and 120 in wind_dir_deg_by_height:
                df["wind_veer_120_80"] = _angle_diff_deg(
                    wind_dir_deg_by_height[120],
                    wind_dir_deg_by_height[80],
                )

            if 80 in wind_dir_deg_by_height and 180 in wind_dir_deg_by_height:
                df["wind_veer_180_80"] = _angle_diff_deg(
                    wind_dir_deg_by_height[180],
                    wind_dir_deg_by_height[80],
                )

            if 120 in wind_dir_deg_by_height and 180 in wind_dir_deg_by_height:
                df["wind_veer_180_120"] = _angle_diff_deg(
                    wind_dir_deg_by_height[180],
                    wind_dir_deg_by_height[120],
                )

            if "wind_veer_120_80" in df.columns:
                df["abs_wind_veer_120_80"] = df["wind_veer_120_80"].abs()

            if "wind_veer_180_80" in df.columns:
                df["abs_wind_veer_180_80"] = df["wind_veer_180_80"].abs()

        df = df.copy()

        # ------------------------------------------------------------
        # 7.1 Yaw / frequent wind direction changes
        # ------------------------------------------------------------
        if self._flag("yaw_misalignment"):
            for height in (80, 120, 180):
                wind_dir_col = _find_wind_direction_col(df, height)
                wind_speed_col = _find_wind_speed_col(df, height)

                if height == 80:
                    power_proxy_col = "expected_power_proxy"
                else:
                    power_proxy_col = f"expected_power_proxy_{height}m"

                df = _add_yaw_misalignment_features(
                    df=df,
                    wind_dir_col=wind_dir_col,
                    wind_speed_col=wind_speed_col,
                    datetime_col=self.datetime_col,
                    prefix=f"yaw{height}",
                    power_proxy_col=power_proxy_col,
                    yaw_minutes=10.0,
                )

        # ------------------------------------------------------------
        # 8. Betz limit features
        # ------------------------------------------------------------
        if (
                self._flag("betz_limit")
                and "air_density" in df.columns
                and "ws80_cube" in df.columns
        ):
            # Теоретический максимум мощности с одной турбины по лимиту Беца:
            # P = 0.5 * rho * A * v^3 * Cp
            # где Cp <= 16/27.
            betz_power_one_turbine_mw = (
                    0.5
                    * df["air_density"]
                    * ROTOR_SWEPT_AREA_M2
                    * df["ws80_cube"]
                    * BETZ_LIMIT_CP
                    / 1_000_000.0
            )

            df["betz_power_one_turbine_mw"] = betz_power_one_turbine_mw

            if "available_turbines" in df.columns:
                df["betz_power_farm_mw"] = (
                        df["betz_power_one_turbine_mw"] * df["available_turbines"]
                )
            else:
                df["betz_power_farm_mw"] = (
                        df["betz_power_one_turbine_mw"] * N_TURBINES
                )

            # Физически станция всё равно не может выдать выше установленной мощности.
            if "available_capacity_mw" in df.columns:
                df["betz_power_farm_clipped_mw"] = np.minimum(
                    df["betz_power_farm_mw"],
                    df["available_capacity_mw"],
                )
            else:
                df["betz_power_farm_clipped_mw"] = np.minimum(
                    df["betz_power_farm_mw"],
                    INSTALLED_CAPACITY_MW,
                )

            # Насколько наш простой proxy близок к физическому максимуму.
            if "expected_power_proxy" in df.columns:
                df["expected_to_betz_ratio"] = (
                        df["expected_power_proxy"]
                        / (df["betz_power_farm_clipped_mw"] + 1e-6)
                )

        df = df.copy()
        # ------------------------------------------------------------
        # 9. Lag / diff / rolling features (actually can't be used for 18.05)
        # ------------------------------------------------------------
        if self._flag("lags"):
            """
            lags for theese features added:
            wind_speed_80m
            wind_speed_120m
            wind_gusts_10m
            pressure_msl
            temperature_80m
            expected_power_proxy
            expected_power_density_adj
            wind_power_density
            wind_dir_80m_sin
            wind_dir_80m_cos
            """
            lag_sources: list[tuple[str, str]] = []

            # Raw weather features
            ws80_col = _find_wind_speed_col(df, 80)
            ws120_col = _find_wind_speed_col(df, 120)
            ws180_col = _find_wind_speed_col(df, 180)
            gust10_col = _find_wind_gust_col(df, 10)

            if ws80_col is not None:
                lag_sources.append(("ws80", ws80_col))

            if ws120_col is not None:
                lag_sources.append(("ws120", ws120_col))

            if ws180_col is not None:
                lag_sources.append(("ws180", ws180_col))

            if gust10_col is not None:
                lag_sources.append(("gust10", gust10_col))

            if pressure_col is not None:
                lag_sources.append(("pressure", pressure_col))

            if temp_col is not None:
                lag_sources.append(("temp80", temp_col))

            # Engineered physical features
            engineered_lag_columns = [
                "expected_power_proxy",
                "expected_power_density_adj",
                "wind_power_density",

                # 120m / 180m physics
                "expected_power_proxy_120m",
                "expected_power_proxy_180m",
                "wind_power_density_120m",
                "wind_power_density_180m",

                # rotor-equivalent physics
                "expected_power_proxy_rews",
                "expected_power_density_adj_rews",
                "wind_power_density_rews",
                "rotor_equiv_ws",

                # direction
                "wind_dir_80m_sin",
                "wind_dir_80m_cos",
                "wind_dir_120m_sin",
                "wind_dir_120m_cos",
            ]

            for col in engineered_lag_columns:
                if col in df.columns:
                    lag_sources.append((col, col))

            # Deduplicate in case fuzzy search found same column twice
            seen_prefixes: set[str] = set()
            unique_lag_sources: list[tuple[str, str]] = []

            for prefix, col in lag_sources:
                if prefix not in seen_prefixes:
                    unique_lag_sources.append((prefix, col))
                    seen_prefixes.add(prefix)

            for prefix, col in unique_lag_sources:
                df = _add_time_lag_features(
                    df=df,
                    value_col=col,
                    prefix=prefix,
                    datetime_col=self.datetime_col,
                )

            # Precipitation memory:
            # не просто "дождь сейчас", а "дождь/ливни/снег шли последние часы".
            for prefix, col in [
                ("rain", "rain"),
                ("showers", "showers"),
                ("snowfall", "snowfall"),
                ("precip_stress", "precip_stress_score"),
            ]:
                if col in df.columns:
                    df = _add_past_window_sum_features(
                        df=df,
                        value_col=col,
                        prefix=prefix,
                        datetime_col=self.datetime_col,
                    )

        return df.replace([np.inf, -np.inf], np.nan)


class BaseWindModel:
    """
    Базовый класс-обёртка.

    Наследники должны реализовать:
    - _build_model()
    - fit()
    - predict()

    Здесь лежат только общие штуки:
    - preprocessor
    - save/load
    - clip предсказаний в физически допустимый диапазон
    """

    def __init__(
        self,
        preprocessor: ModelPreprocessor | None = None,
        random_seed: int = 42,
        clip_predictions: bool = True,
    ) -> None:
        self.random_seed = random_seed
        self.clip_predictions = clip_predictions
        self.preprocessor = preprocessor or ModelPreprocessor()
        self.model = None

    def _clip(self, predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=float)

        if self.clip_predictions:
            predictions = np.clip(predictions, 0.0, INSTALLED_CAPACITY_MW)

        return predictions

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseWindModel":
        return joblib.load(path)

    @staticmethod
    def _validate_input_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame as input.")