from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd


TARGET_COL = "Выработка"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

# 26 turbines * 3.465 MW = 90.09 MW
INSTALLED_CAPACITY_MW = 90.09
N_TURBINES = 26
TURBINE_CAPACITY_MW = INSTALLED_CAPACITY_MW / N_TURBINES
R_DRY_AIR = 287.05


COLUMN_RENAME_MAP = {
    "РљРѕР»-РІРѕ_Р’Р­РЈ_РІ_СЂРµРјРѕРЅС‚Рµ": "repair_count",
    "Кол-во_ВЭУ_в_ремонте": "repair_count",
    "Количество_ВЭУ_в_ремонте": "repair_count",
}


DEFAULT_FEATURE_FLAGS: dict[str, bool] = {
    "time": True,
    "availability": True,
    "power_curve": True,
    "air_density": True,
    "wind_power_density": True,
    "wind_direction": True,
    "wind_shear": True,
}


def _norm_col(value: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", "_", str(value).lower()).strip("_")


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
            or "азимут" in n
        )

        if has_speed and has_height and not is_direction:
            score = 0
            if f"{height}m" in n or f"{height}_m" in n:
                score += 2
            candidates.append((score, col))

    if not candidates:
        return None
    return sorted(candidates, key=lambda x: -x[0])[0][1]


def _find_wind_direction_col(df: pd.DataFrame, height_m: int | None = 80) -> str | None:
    candidates: list[tuple[int, str]] = []

    for col in df.columns:
        n = _norm_col(col)
        has_direction = (
            "wind_dir" in n
            or "direction" in n
            or "dir" in n
            or "направ" in n
            or "азимут" in n
            or "румб" in n
        )
        has_wind = "wind" in n or "ветер" in n or "ветр" in n

        if not has_direction:
            continue

        score = 0
        if has_wind:
            score += 1
        if height_m is not None and str(height_m) in n:
            score += 2
        candidates.append((score, col))

    if not candidates:
        return None
    return sorted(candidates, key=lambda x: -x[0])[0][1]


def _find_temperature_col(df: pd.DataFrame) -> str | None:
    candidates: list[tuple[int, str]] = []
    for col in df.columns:
        n = _norm_col(col)
        if any(tok in n for tok in ("temperature", "temp", "t2m", "температура", "темп")):
            score = 2 if "80" in n else 0
            candidates.append((score, col))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: -x[0])[0][1]


def _find_pressure_col(df: pd.DataFrame) -> str | None:
    candidates: list[tuple[int, str]] = []
    for col in df.columns:
        n = _norm_col(col)
        if any(tok in n for tok in ("pressure", "press", "давление", "msl", "sp")):
            candidates.append((0, col))
    if not candidates:
        return None
    return candidates[0][1]


@dataclass
class ModelPreprocessor:
    """
    Shared preprocessor for tabular wind-power models.

    Feature engineering is intentionally centralized here because all model wrappers
    import this class. The code does not use target leakage and works for both train
    and valid/test data.
    """

    target_col: str = TARGET_COL
    datetime_col: str = DATETIME_COL
    add_demo_features: bool = True
    feature_flags: dict[str, bool] | None = None
    feature_columns_: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.feature_flags is None:
            self.feature_flags = DEFAULT_FEATURE_FLAGS.copy()
        else:
            merged = DEFAULT_FEATURE_FLAGS.copy()
            merged.update(self.feature_flags)
            self.feature_flags = merged

    def fit(self, df: pd.DataFrame) -> "ModelPreprocessor":
        prepared = self._prepare_dataframe(df)

        ignored = {self.target_col, self.datetime_col}
        feature_columns = [col for col in prepared.columns if col not in ignored]

        numeric_feature_columns = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(prepared[col]):
                numeric_feature_columns.append(col)

        self.feature_columns_ = numeric_feature_columns
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns_:
            raise RuntimeError("Preprocessor is not fitted. Call preprocessor.fit(df) first.")

        prepared = self._prepare_dataframe(df)

        for col in self.feature_columns_:
            if col not in prepared.columns:
                prepared[col] = np.nan

        X = prepared[self.feature_columns_].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        return X.replace([np.inf, -np.inf], np.nan)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        if self.target_col not in df.columns:
            raise ValueError(f"Target column {self.target_col!r} not found in dataframe.")
        return pd.to_numeric(df[self.target_col], errors="coerce")

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        prepared = prepared.rename(columns=COLUMN_RENAME_MAP)

        if self.datetime_col in prepared.columns:
            prepared[self.datetime_col] = pd.to_datetime(prepared[self.datetime_col], errors="coerce")

        prepared = self.add_features(prepared)
        return prepared

    def _flag(self, name: str) -> bool:
        return bool((self.feature_flags or {}).get(name, False))

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering block.

        Do not use target here. This function must work for both train and valid/test.
        """
        if not self.add_demo_features:
            return df

        df = df.copy()

        # 1. Time features
        if self._flag("time"):
            hour = None
            month = None
            dayofyear = None

            if "hour_of_day" in df.columns:
                hour = pd.to_numeric(df["hour_of_day"], errors="coerce")
            elif self.datetime_col in df.columns:
                hour = df[self.datetime_col].dt.hour
                month = df[self.datetime_col].dt.month
                dayofyear = df[self.datetime_col].dt.dayofyear
            if "month" in df.columns and month is None:
                month = pd.to_numeric(df["month"], errors="coerce")

            if hour is not None:
                df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            if month is not None:
                df["month_sin"] = np.sin(2 * np.pi * month / 12)
                df["month_cos"] = np.cos(2 * np.pi * month / 12)
            if dayofyear is not None:
                df["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
                df["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)

        # 2. Availability features
        if self._flag("availability"):
            if "repair_count" in df.columns:
                repair_count = pd.to_numeric(df["repair_count"], errors="coerce").fillna(0.0).clip(0, N_TURBINES)
            else:
                repair_count = pd.Series(0.0, index=df.index)

            df["available_turbines"] = N_TURBINES - repair_count
            df["availability_ratio"] = df["available_turbines"] / N_TURBINES
            df["available_capacity_mw"] = df["available_turbines"] * TURBINE_CAPACITY_MW

        # 3. Wind speed features at hub height
        ws80_col = _find_wind_speed_col(df, 80)
        ws80 = None
        if ws80_col is not None:
            ws80 = pd.to_numeric(df[ws80_col], errors="coerce")

        # If no exact 80m wind speed, try log interpolation from 10m and 120m.
        if ws80 is None:
            ws10_col = _find_wind_speed_col(df, 10)
            ws120_col = _find_wind_speed_col(df, 120)
            if ws10_col is not None and ws120_col is not None:
                w1 = pd.to_numeric(df[ws10_col], errors="coerce").clip(lower=0.1)
                w2 = pd.to_numeric(df[ws120_col], errors="coerce").clip(lower=0.1)
                alpha = np.log(w2 / w1) / np.log(120.0 / 10.0)
                ws80 = w1 * (80.0 / 10.0) ** alpha
                df["ws80_interpolated"] = ws80

        if ws80 is not None and self._flag("power_curve"):
            df["ws80_sq"] = ws80 ** 2
            df["ws80_cube"] = ws80 ** 3

            power_curve_proxy = ((ws80.clip(3.0, 12.0) - 3.0) / 9.0) ** 3
            df["power_curve_proxy"] = power_curve_proxy.clip(0.0, 1.0)

            if "available_capacity_mw" in df.columns:
                df["expected_power_proxy"] = df["power_curve_proxy"] * df["available_capacity_mw"]
                df["ws80_x_available_capacity"] = ws80 * df["available_capacity_mw"]
            else:
                df["expected_power_proxy"] = df["power_curve_proxy"] * INSTALLED_CAPACITY_MW

        # 4. Wind direction features
        if self._flag("wind_direction"):
            direction_col = _find_wind_direction_col(df, 80) or _find_wind_direction_col(df, None)
            if direction_col is not None:
                wd = pd.to_numeric(df[direction_col], errors="coerce")
                # If values look like 0..1, treat them as normalized direction.
                wd_median = wd.median(skipna=True)
                wd_max = wd.max(skipna=True)
                if pd.notna(wd_max) and wd_max <= 1.5:
                    wd_deg = wd * 360.0
                else:
                    wd_deg = wd
                wd_rad = np.deg2rad(wd_deg)
                df["wind_dir_sin"] = np.sin(wd_rad)
                df["wind_dir_cos"] = np.cos(wd_rad)
                df["wind_sector_8"] = ((wd_deg % 360) // 45).astype("float")

        # 5. Wind shear features
        if self._flag("wind_shear"):
            ws10_col = _find_wind_speed_col(df, 10)
            ws120_col = _find_wind_speed_col(df, 120)
            if ws80 is not None and ws10_col is not None:
                ws10 = pd.to_numeric(df[ws10_col], errors="coerce")
                eps = 1e-6
                df["wind_shear_80_10"] = ws80 - ws10
                df["wind_shear_ratio_80_10"] = ws80 / (ws10 + eps)
                valid = (ws10 > 0.5) & (ws80 > 0.5)
                df["alpha_80_10"] = np.nan
                df.loc[valid, "alpha_80_10"] = np.log(ws80.loc[valid] / ws10.loc[valid]) / np.log(80 / 10)
            if ws80 is not None and ws120_col is not None:
                ws120 = pd.to_numeric(df[ws120_col], errors="coerce")
                df["wind_shear_120_80"] = ws120 - ws80
                df["wind_shear_ratio_120_80"] = ws120 / (ws80 + 1e-6)

        # 6. Air density
        temp_col = _find_temperature_col(df)
        pressure_col = _find_pressure_col(df)

        if self._flag("air_density") and temp_col is not None and pressure_col is not None:
            temp = pd.to_numeric(df[temp_col], errors="coerce")
            pressure = pd.to_numeric(df[pressure_col], errors="coerce")

            temp_median = temp.median(skipna=True)
            pressure_median = pressure.median(skipna=True)

            temp_k = temp if pd.notna(temp_median) and temp_median > 150 else temp + 273.15
            pressure_pa = pressure * 100.0 if pd.notna(pressure_median) and pressure_median < 2000 else pressure

            df["air_density"] = pressure_pa / (R_DRY_AIR * temp_k)

        # 7. Wind power density
        if self._flag("wind_power_density") and "air_density" in df.columns and "ws80_cube" in df.columns:
            df["wind_power_density"] = 0.5 * df["air_density"] * df["ws80_cube"]

        return df.replace([np.inf, -np.inf], np.nan)


class BaseWindModel:
    """
    Base wrapper class used by existing CatBoost/XGBoost wrappers.
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
