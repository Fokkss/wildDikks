from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd


TARGET_COL = "Выработка"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

# Из README / условия: 26 турбин * 3.465 МВт = 90.09 МВт.
INSTALLED_CAPACITY_MW = 90.09


# Названия, которые приводим к нормальному виду.
# Главная цель — не таскать по проекту битую кодировку.
COLUMN_RENAME_MAP = {
    "РљРѕР»-РІРѕ_Р’Р­РЈ_РІ_СЂРµРјРѕРЅС‚Рµ": "repair_count",
    "Кол-во_ВЭУ_в_ремонте": "repair_count",
    "Количество_ВЭУ_в_ремонте": "repair_count",
}


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

        return X

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

        if "hour_of_day" in df.columns:
            hour = pd.to_numeric(df["hour_of_day"], errors="coerce")
        elif self.datetime_col in df.columns:
            hour = df[self.datetime_col].dt.hour
        else:
            return df

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        return df


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