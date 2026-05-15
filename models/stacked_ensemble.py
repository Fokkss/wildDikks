from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from new_model.config import available_capacity_from_df
from .base import INSTALLED_CAPACITY_MW, ModelPreprocessor


@dataclass
class FittedModelPack:
    name: str
    fold_models: list[tuple[ModelPreprocessor, Any]] = field(default_factory=list)
    full_model: tuple[ModelPreprocessor, Any] | None = None


class TimeSeriesStackingEnsemble:
    """
    N-model time-series stacking ensemble.

    What it does:
    1. For every base model f_i, it trains TimeSeriesSplit fold models.
    2. Fold validation predictions become out-of-fold meta-features.
    3. A simple Ridge meta-model learns how to combine base predictions.
    4. For test/valid inference, every f_i predicts with its fold models and their
       mean is passed to the meta-model.

    This is safer than pseudo-labeling future weeks with model predictions because
    the meta-model is trained only on out-of-fold predictions with known targets.
    """

    def __init__(
        self,
        model_specs: list[dict[str, Any]],
        n_splits: int = 5,
        meta_alpha: float = 1.0,
        random_seed: int = 42,
        clip_predictions: bool = True,
        prediction_mode: str = "fold_mean",
    ) -> None:
        if not model_specs:
            raise ValueError("model_specs must contain at least one base model.")
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        if prediction_mode not in {"fold_mean", "full"}:
            raise ValueError("prediction_mode must be 'fold_mean' or 'full'.")

        self.model_specs = model_specs
        self.n_splits = n_splits
        self.meta_alpha = meta_alpha
        self.random_seed = random_seed
        self.clip_predictions = clip_predictions
        self.prediction_mode = prediction_mode

        self.fitted_packs_: list[FittedModelPack] = []
        self.meta_model_: Ridge | None = None
        self.meta_feature_names_: list[str] = []
        self.oof_score_: float | None = None

    @staticmethod
    def default_xgb_params(seed: int = 42) -> dict[str, Any]:
        return {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": 2000,
            "learning_rate": 0.02,
            "max_depth": 4,
            "min_child_weight": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.1,
            "reg_lambda": 6.0,
            "tree_method": "hist",
            "max_bin": 256,
            "random_state": seed,
            "n_jobs": -1,
            "early_stopping_rounds": 150,
        }

    @classmethod
    def default_specs(
        cls,
        n_models: int = 5,
        seed: int = 42,
        feature_flags: dict[str, bool] | None = None,
    ) -> list[dict[str, Any]]:
        specs = []
        for i in range(n_models):
            params = cls.default_xgb_params(seed + i)
            params.update(
                {
                    # Small diversity between base models.
                    "max_depth": [3, 4, 5, 4, 3, 5][i % 6],
                    "learning_rate": [0.015, 0.02, 0.025, 0.018, 0.03, 0.012][i % 6],
                    "subsample": [0.75, 0.85, 0.9, 0.8, 0.7, 0.88][i % 6],
                    "colsample_bytree": [0.65, 0.75, 0.85, 0.7, 0.9, 0.8][i % 6],
                    "random_state": seed + i,
                }
            )
            specs.append(
                {
                    "name": f"xgb_{i}",
                    "model_type": "xgb",
                    "seed_offset": i,
                    "params": params,
                    "feature_flags": feature_flags,
                }
            )
        return specs

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TimeSeriesStackingEnsemble":
        return cls(
            model_specs=config["model_specs"],
            n_splits=int(config.get("n_splits", 5)),
            meta_alpha=float(config.get("meta_alpha", 1.0)),
            random_seed=int(config.get("random_seed", 42)),
            clip_predictions=bool(config.get("clip_predictions", True)),
            prediction_mode=str(config.get("prediction_mode", "fold_mean")),
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "model_specs": self.model_specs,
            "n_splits": self.n_splits,
            "meta_alpha": self.meta_alpha,
            "random_seed": self.random_seed,
            "clip_predictions": self.clip_predictions,
            "prediction_mode": self.prediction_mode,
        }

    def _make_model(self, spec: dict[str, Any], fold_seed_add: int = 0, for_full_fit: bool = False) -> Any:
        model_type = spec.get("model_type", "xgb")
        params = dict(spec.get("params", {}))

        seed = self.random_seed + int(spec.get("seed_offset", 0)) + fold_seed_add
        params["random_state"] = seed

        if for_full_fit:
            # XGBoost cannot use early stopping without eval_set.
            params.pop("early_stopping_rounds", None)

        if model_type != "xgb":
            raise ValueError(f"Unsupported model_type: {model_type!r}. Current stacked ensemble uses XGBoost base models.")

        return XGBRegressor(**params)

    def _clip_to_row_capacity(self, pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        pred = np.asarray(pred, dtype=float)
        if not self.clip_predictions:
            return pred

        row_cap = available_capacity_from_df(df).to_numpy()
        row_cap = np.minimum(row_cap, INSTALLED_CAPACITY_MW)
        return np.clip(pred, 0.0, row_cap)

    def _fit_one_spec(self, df: pd.DataFrame, y: pd.Series, spec: dict[str, Any]) -> tuple[np.ndarray, FittedModelPack]:
        name = str(spec.get("name", f"model_{len(self.fitted_packs_)}"))
        pack = FittedModelPack(name=name)
        oof = np.full(shape=len(df), fill_value=np.nan, dtype=float)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for fold_id, (train_idx, valid_idx) in enumerate(tscv.split(df), start=1):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            valid_df = df.iloc[valid_idx].reset_index(drop=True)

            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_valid = y.iloc[valid_idx].reset_index(drop=True)

            train_mask = y_train.notna()
            train_df = train_df.loc[train_mask].reset_index(drop=True)
            y_train = y_train.loc[train_mask].reset_index(drop=True)

            preprocessor = ModelPreprocessor(feature_flags=spec.get("feature_flags"))
            X_train = preprocessor.fit_transform(train_df)
            X_valid = preprocessor.transform(valid_df)

            model = self._make_model(spec, fold_seed_add=fold_id, for_full_fit=False)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )

            pred = model.predict(X_valid)
            pred = self._clip_to_row_capacity(pred, valid_df)
            oof[valid_idx] = pred

            pack.fold_models.append((preprocessor, model))

        # Full model is optional for inference mode='full' and useful for inspection.
        full_preprocessor = ModelPreprocessor(feature_flags=spec.get("feature_flags"))
        X_full = full_preprocessor.fit_transform(df)
        full_model = self._make_model(spec, fold_seed_add=9999, for_full_fit=True)
        full_model.fit(X_full, y, verbose=False)
        pack.full_model = (full_preprocessor, full_model)

        return oof, pack

    def fit(self, train_df: pd.DataFrame) -> "TimeSeriesStackingEnsemble":
        y = pd.to_numeric(train_df["Выработка"], errors="coerce")
        mask = y.notna()
        df = train_df.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

        oof_columns: list[np.ndarray] = []
        packs: list[FittedModelPack] = []
        names: list[str] = []

        for spec in self.model_specs:
            oof, pack = self._fit_one_spec(df=df, y=y, spec=spec)
            oof_columns.append(oof)
            packs.append(pack)
            names.append(pack.name)

        meta_X = pd.DataFrame({name: col for name, col in zip(names, oof_columns)})
        meta_mask = meta_X.notna().all(axis=1)

        if int(meta_mask.sum()) < max(100, len(self.model_specs) * 10):
            raise RuntimeError("Not enough out-of-fold rows to train meta-model. Reduce n_splits or check data size.")

        self.meta_model_ = Ridge(alpha=self.meta_alpha)
        self.meta_model_.fit(meta_X.loc[meta_mask], y.loc[meta_mask])

        meta_pred = self.meta_model_.predict(meta_X.loc[meta_mask])
        meta_pred = self._clip_to_row_capacity(meta_pred, df.loc[meta_mask].reset_index(drop=True))
        self.oof_score_ = float(np.mean(np.abs(y.loc[meta_mask].to_numpy() - meta_pred)))

        self.fitted_packs_ = packs
        self.meta_feature_names_ = names
        return self

    def _predict_one_pack(self, pack: FittedModelPack, features_df: pd.DataFrame) -> np.ndarray:
        if self.prediction_mode == "full":
            if pack.full_model is None:
                raise RuntimeError(f"Pack {pack.name} has no full_model.")
            preprocessor, model = pack.full_model
            X = preprocessor.transform(features_df)
            pred = model.predict(X)
            return self._clip_to_row_capacity(pred, features_df)

        preds = []
        for preprocessor, model in pack.fold_models:
            X = preprocessor.transform(features_df)
            pred = model.predict(X)
            pred = self._clip_to_row_capacity(pred, features_df)
            preds.append(pred)

        return np.mean(np.vstack(preds), axis=0)

    def predict_base_matrix(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_packs_:
            raise RuntimeError("Ensemble is not fitted yet.")

        data = {}
        for pack in self.fitted_packs_:
            data[pack.name] = self._predict_one_pack(pack, features_df)

        return pd.DataFrame(data)

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if self.meta_model_ is None:
            raise RuntimeError("Meta-model is not fitted yet.")

        meta_X = self.predict_base_matrix(features_df)
        pred = self.meta_model_.predict(meta_X[self.meta_feature_names_])
        return self._clip_to_row_capacity(pred, features_df)

    def meta_weights(self) -> pd.DataFrame:
        if self.meta_model_ is None:
            raise RuntimeError("Meta-model is not fitted yet.")

        return pd.DataFrame(
            {
                "model": self.meta_feature_names_,
                "weight": self.meta_model_.coef_,
            }
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TimeSeriesStackingEnsemble":
        return joblib.load(path)
