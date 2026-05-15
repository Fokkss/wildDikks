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
        ws80_col = _find_wind_speed_col(df, 80)

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
        # 5. Wind power density
        # ------------------------------------------------------------
        if (
                self._flag("wind_power_density")
                and "air_density" in df.columns
                and "ws80_cube" in df.columns
        ):
            df["wind_power_density"] = 0.5 * df["air_density"] * df["ws80_cube"]

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