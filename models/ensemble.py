from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .base import INSTALLED_CAPACITY_MW
from .catboost_model import CatBoostWindModel
from .xgboost_model import XGBoostWindModel


class WeightedEnsembleModel:
    """
    Ансамбль CatBoost + XGBoost.

    Ансамбль здесь — это не новая сложная нейросеть.
    Это просто объединение предсказаний нескольких моделей.

    Пример:
        итог = 0.5 * catboost_prediction + 0.5 * xgboost_prediction

    Если одна модель на лидерборде/валидации лучше, можно поменять веса:
        catboost_weight=0.7
        xgboost_weight=0.3
    """

    def __init__(
        self,
        catboost_model: CatBoostWindModel | None = None,
        xgboost_model: XGBoostWindModel | None = None,
        catboost_weight: float = 0.5,
        xgboost_weight: float = 0.5,
        clip_predictions: bool = True,
    ) -> None:
        self.catboost_model = catboost_model or CatBoostWindModel()
        self.xgboost_model = xgboost_model or XGBoostWindModel()

        self.catboost_weight = catboost_weight
        self.xgboost_weight = xgboost_weight
        self.clip_predictions = clip_predictions

        self._validate_weights()

    def _validate_weights(self) -> None:
        total_weight = self.catboost_weight + self.xgboost_weight

        if total_weight <= 0:
            raise ValueError("Sum of ensemble weights must be positive.")

        # Нормализуем веса, чтобы сумма была 1.
        self.catboost_weight = self.catboost_weight / total_weight
        self.xgboost_weight = self.xgboost_weight / total_weight

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
    ) -> "WeightedEnsembleModel":
        """
        Обучает обе модели на одних и тех же данных.

        Важно:
        У CatBoost и XGBoost свои препроцессоры.
        Это сделано специально, чтобы модели были независимыми.
        """

        self.catboost_model.fit(train_df=train_df, valid_df=valid_df)
        self.xgboost_model.fit(train_df=train_df, valid_df=valid_df)

        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        cat_pred = self.catboost_model.predict(features_df)
        xgb_pred = self.xgboost_model.predict(features_df)

        predictions = (
            self.catboost_weight * cat_pred
            + self.xgboost_weight * xgb_pred
        )

        if self.clip_predictions:
            predictions = np.clip(predictions, 0.0, INSTALLED_CAPACITY_MW)

        return predictions

    def predict_to_csv(
        self,
        features_df: pd.DataFrame,
        output_path: str | Path,
        header: bool = False,
        index: bool = False,
    ) -> None:
        """
        Сохраняет CSV в формате платформы:
        - одна колонка;
        - без индекса;
        - по умолчанию без заголовка.

        Количество строк будет равно количеству строк в features_df.
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        predictions = self.predict(features_df)

        pd.DataFrame(predictions).to_csv(
            output_path,
            index=index,
            header=header,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "WeightedEnsembleModel":
        return joblib.load(path)

    def get_feature_importance(self) -> dict[str, pd.DataFrame]:
        """
        Возвращает важности признаков отдельно для CatBoost и XGBoost.
        """

        return {
            "catboost": self.catboost_model.get_feature_importance(),
            "xgboost": self.xgboost_model.get_feature_importance(),
        }