from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .base import BaseWindModel, ModelPreprocessor


class CatBoostWindModel(BaseWindModel):
    """
    Минимальная обёртка над CatBoostRegressor.

    Почему CatBoost:
    - устойчивый бустинг;
    - нормально работает с пропусками;
    - часто хорошо заходит на табличных задачах без дикого тюнинга.
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
        self.model = CatBoostRegressor(**self.params)

    def default_params(self) -> dict[str, Any]:
        return {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "iterations": 1200,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "random_seed": self.random_seed,
            "verbose": 100,
            "allow_writing_files": False,
        }

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
    ) -> "CatBoostWindModel":
        self._validate_input_dataframe(train_df)

        X_train = self.preprocessor.fit_transform(train_df)
        y_train = self.preprocessor.get_target(train_df)

        eval_set = None
        if valid_df is not None:
            self._validate_input_dataframe(valid_df)
            X_valid = self.preprocessor.transform(valid_df)
            y_valid = self.preprocessor.get_target(valid_df)
            eval_set = (X_valid, y_valid)

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            use_best_model=valid_df is not None,
        )

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
        importances = self.model.get_feature_importance()

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