from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .base import BaseWindModel, ModelPreprocessor


class XGBoostWindModel(BaseWindModel):
    """
    Минимальная обёртка над XGBRegressor.

    Почему XGBoost:
    - сильный классический бустинг;
    - хорошо работает с числовыми табличными признаками;
    - даёт модель, отличающуюся от CatBoost, поэтому полезен для ансамбля.
    """

    def __init__(
        self,
        preprocessor: ModelPreprocessor | None = None,
        random_seed: int = 42,
        clip_predictions: bool = True,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            preprocessor=preprocessor,
            random_seed=random_seed,
            clip_predictions=clip_predictions,
        )

        self.params = params or self.default_params()
        self.model = XGBRegressor(**self.params)

    def default_params(self) -> dict[str, Any]:
        return {
            "objective": "reg:absoluteerror",
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
            "random_state": self.random_seed,
            "tree_method": "hist",
            "n_jobs": -1,
        }

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
    ) -> "XGBoostWindModel":
        self._validate_input_dataframe(train_df)

        X_train = self.preprocessor.fit_transform(train_df)
        y_train = self.preprocessor.get_target(train_df)

        if valid_df is not None:
            self._validate_input_dataframe(valid_df)
            X_valid = self.preprocessor.transform(valid_df)
            y_valid = self.preprocessor.get_target(valid_df)

            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=100,
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        self._validate_input_dataframe(features_df)

        X = self.preprocessor.transform(features_df)
        predictions = self.model.predict(X)

        return self._clip(predictions)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Удобно для анализа после обучения.
        Не влияет на обучение и предсказание.
        """

        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        feature_names = self.preprocessor.feature_columns_
        importances = self.model.feature_importances_

        return (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )