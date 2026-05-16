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

# Из README / условия: 26 турбин * 3.465 МВт = 90.09 МВт.
INSTALLED_CAPACITY_MW = 90.09

# smth
N_TURBINES = 26
TURBINE_CAPACITY_MW = INSTALLED_CAPACITY_MW / N_TURBINES
R_DRY_AIR = 287.05
STANDARD_AIR_DENSITY = 1.225  # kg/m^3
BETZ_LIMIT_CP = 16.0 / 27.0   # ≈ 0.593, theoretical max efficiency


ROTOR_DIAMETER_M = 132.0
ROTOR_SWEPT_AREA_M2 = np.pi * (ROTOR_DIAMETER_M / 2.0) ** 2

LAG_HOURS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
DIFF_HOURS = (1, 3, 24)
ROLLING_MEAN_HOURS = (3, 6, 12, 24, 72)
ROLLING_STD_HOURS = (6, 24, 72)


# Названия, которые приводим к нормальному виду.
# Главная цель — не таскать по проекту битую кодировку.
COLUMN_RENAME_MAP = {
    "РљРѕР»-РІРѕ_Р’Р­РЈ_РІ_СЂРµРјРѕРЅС‚Рµ": "repair_count",
    "Кол-во_ВЭУ_в_ремонте": "repair_count",
    "Количество_ВЭУ_в_ремонте": "repair_count",
}


DEFAULT_FEATURE_FLAGS = {
    "time": True,
    "availability": True,
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
        # :NOTE: lugs
        # :NOTE: Bets limit
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
            gust10_col = _find_wind_gust_col(df, 10)

            if ws80_col is not None:
                lag_sources.append(("ws80", ws80_col))

            if ws120_col is not None:
                lag_sources.append(("ws120", ws120_col))

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
                "wind_dir_80m_sin",
                "wind_dir_80m_cos",
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