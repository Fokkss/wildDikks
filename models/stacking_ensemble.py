from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency during import checks
    CatBoostRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None  # type: ignore

from new_model.config import TARGET_COL, available_capacity_from_df
from .base import INSTALLED_CAPACITY_MW, ModelPreprocessor


MetaModelType = Literal["ridge", "linear", "xgb", "lightgbm"]
PredictionMode = Literal["full", "fold_mean", "blend_full_fold"]


@dataclass
class FittedBasePack:
    name: str
    model_type: str
    fold_models: list[tuple[ModelPreprocessor, Any]] = field(default_factory=list)
    full_model: tuple[ModelPreprocessor, Any] | None = None


class TimeSeriesStackingEnsemble:
    """
    Proper time-series stacking ensemble.

    Level 0:
        Several diverse base models: XGBoost and optional CatBoost.

    Level 1:
        A meta-model trained on out-of-fold base predictions.
        Supported meta-models:
            - Ridge
            - LinearRegression
            - small XGBoost
            - LightGBM, if installed

    Why this is safe:
        The meta-model does not see train predictions made by models that were
        trained on the same rows. It sees only out-of-fold predictions.
    """

    def __init__(
        self,
        base_model_specs: list[dict[str, Any]],
        meta_model_type: MetaModelType = "ridge",
        meta_model_params: dict[str, Any] | None = None,
        n_splits: int = 5,
        random_seed: int = 42,
        clip_predictions: bool = True,
        prediction_mode: PredictionMode = "full",
        append_meta_stats: bool = True,
    ) -> None:
        if not base_model_specs:
            raise ValueError("base_model_specs must contain at least one model spec.")
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        if prediction_mode not in {"full", "fold_mean", "blend_full_fold"}:
            raise ValueError("prediction_mode must be: full, fold_mean, blend_full_fold.")

        self.base_model_specs = base_model_specs
        self.meta_model_type = meta_model_type
        self.meta_model_params = meta_model_params or {}
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.clip_predictions = clip_predictions
        self.prediction_mode = prediction_mode
        self.append_meta_stats = append_meta_stats

        self.fitted_packs_: list[FittedBasePack] = []
        self.meta_model_: Any | None = None
        self.meta_feature_names_: list[str] = []
        self.oof_mae_mw_: float | None = None
        self.oof_competition_error_percent_: float | None = None

    @staticmethod
    def default_xgb_params(seed: int = 42) -> dict[str, Any]:
        return {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": 2500,
            "learning_rate": 0.02,
            "max_depth": 4,
            "min_child_weight": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.10,
            "reg_lambda": 6.0,
            "tree_method": "hist",
            "max_bin": 256,
            "random_state": seed,
            "n_jobs": -1,
            "early_stopping_rounds": 150,
        }

    @staticmethod
    def default_catboost_params(seed: int = 42) -> dict[str, Any]:
        return {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "iterations": 1800,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "od_type": "Iter",
            "od_wait": 150,
        }

    @classmethod
    def default_specs(
        cls,
        n_xgb: int = 4,
        n_cat: int = 1,
        feature_flags: dict[str, bool] | None = None,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """
        Diverse default Level-0 models.
        Diversity is more important than just cloning the same XGB many times.
        """
        specs: list[dict[str, Any]] = []

        xgb_variants = [
            {"max_depth": 3, "min_child_weight": 12, "subsample": 0.90, "colsample_bytree": 0.85, "reg_lambda": 8.0},
            {"max_depth": 4, "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 6.0},
            {"max_depth": 5, "min_child_weight": 8, "subsample": 0.80, "colsample_bytree": 0.70, "reg_lambda": 5.0},
            {"max_depth": 3, "min_child_weight": 18, "subsample": 0.75, "colsample_bytree": 0.95, "reg_lambda": 12.0},
            {"max_depth": 4, "min_child_weight": 6, "subsample": 0.95, "colsample_bytree": 0.65, "reg_lambda": 4.0},
        ]

        for i in range(n_xgb):
            params = cls.default_xgb_params(seed + i)
            params.update(xgb_variants[i % len(xgb_variants)])
            specs.append(
                {
                    "name": f"xgb_{i}",
                    "model_type": "xgb",
                    "seed_offset": i,
                    "params": params,
                    "feature_flags": feature_flags,
                }
            )

        for i in range(n_cat):
            params = cls.default_catboost_params(seed + 100 + i)
            params.update({"depth": 5 + (i % 3), "l2_leaf_reg": 5.0 + 2.0 * i})
            specs.append(
                {
                    "name": f"cat_{i}",
                    "model_type": "catboost",
                    "seed_offset": 100 + i,
                    "params": params,
                    "feature_flags": feature_flags,
                }
            )

        return specs

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TimeSeriesStackingEnsemble":
        return cls(
            base_model_specs=config["base_model_specs"],
            meta_model_type=config.get("meta_model_type", "ridge"),
            meta_model_params=config.get("meta_model_params", {}),
            n_splits=int(config.get("n_splits", 5)),
            random_seed=int(config.get("random_seed", 42)),
            clip_predictions=bool(config.get("clip_predictions", True)),
            prediction_mode=config.get("prediction_mode", "full"),
            append_meta_stats=bool(config.get("append_meta_stats", True)),
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "base_model_specs": self.base_model_specs,
            "meta_model_type": self.meta_model_type,
            "meta_model_params": self.meta_model_params,
            "n_splits": self.n_splits,
            "random_seed": self.random_seed,
            "clip_predictions": self.clip_predictions,
            "prediction_mode": self.prediction_mode,
            "append_meta_stats": self.append_meta_stats,
        }

    def _make_model(self, spec: dict[str, Any], seed_add: int = 0):
        model_type = spec["model_type"]
        params = dict(spec.get("params", {}))
        seed = self.random_seed + int(spec.get("seed_offset", 0)) + seed_add

        if model_type == "xgb":
            params.setdefault("random_state", seed)
            return XGBRegressor(**params)

        if model_type == "catboost":
            if CatBoostRegressor is None:
                raise ImportError("CatBoost is not installed. Install catboost or remove catboost specs.")
            params.setdefault("random_seed", seed)
            return CatBoostRegressor(**params)

        raise ValueError(f"Unknown model_type: {model_type!r}")

    def _make_preprocessor(self, spec: dict[str, Any]) -> ModelPreprocessor:
        return ModelPreprocessor(feature_flags=spec.get("feature_flags"))

    def _fit_model(
        self,
        model: Any,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        if model_type == "xgb":
            if X_valid is not None and y_valid is not None:
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                # XGBoost raises if early_stopping_rounds is set without eval_set.
                try:
                    model.set_params(early_stopping_rounds=None)
                except Exception:
                    pass
                model.fit(X_train, y_train, verbose=False)
            return

        if model_type == "catboost":
            if X_valid is not None and y_valid is not None:
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
            else:
                model.fit(X_train, y_train)
            return

        raise ValueError(f"Unknown model_type: {model_type!r}")

    def _make_meta_model(self):
        params = dict(self.meta_model_params)

        if self.meta_model_type == "ridge":
            params.setdefault("alpha", 1.0)
            return Ridge(**params)

        if self.meta_model_type == "linear":
            return LinearRegression(**params)

        if self.meta_model_type == "xgb":
            defaults = {
                "objective": "reg:absoluteerror",
                "eval_metric": "mae",
                "n_estimators": 400,
                "learning_rate": 0.03,
                "max_depth": 2,
                "min_child_weight": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 5.0,
                "tree_method": "hist",
                "random_state": self.random_seed + 777,
                "n_jobs": -1,
            }
            defaults.update(params)
            return XGBRegressor(**defaults)

        if self.meta_model_type == "lightgbm":
            if LGBMRegressor is None:
                raise ImportError("LightGBM is not installed. Use meta_model_type='ridge'/'linear'/'xgb' or install lightgbm.")
            defaults = {
                "objective": "regression_l1",
                "n_estimators": 300,
                "learning_rate": 0.03,
                "num_leaves": 8,
                "min_child_samples": 40,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 5.0,
                "random_state": self.random_seed + 888,
                "n_jobs": -1,
                "verbose": -1,
            }
            defaults.update(params)
            return LGBMRegressor(**defaults)

        raise ValueError(f"Unknown meta_model_type: {self.meta_model_type!r}")

    def _augment_meta_features(self, base_pred_matrix: np.ndarray) -> np.ndarray:
        if not self.append_meta_stats:
            return base_pred_matrix

        mean_pred = np.nanmean(base_pred_matrix, axis=1, keepdims=True)
        std_pred = np.nanstd(base_pred_matrix, axis=1, keepdims=True)
        min_pred = np.nanmin(base_pred_matrix, axis=1, keepdims=True)
        max_pred = np.nanmax(base_pred_matrix, axis=1, keepdims=True)

        return np.hstack([base_pred_matrix, mean_pred, std_pred, min_pred, max_pred])

    def fit(self, train_df: pd.DataFrame) -> "TimeSeriesStackingEnsemble":
        if TARGET_COL not in train_df.columns:
            raise ValueError(f"Target column {TARGET_COL!r} not found.")

        train_df = train_df.reset_index(drop=True)
        y_all = pd.to_numeric(train_df[TARGET_COL], errors="coerce")
        valid_target_mask = y_all.notna()
        if not valid_target_mask.all():
            train_df = train_df.loc[valid_target_mask].reset_index(drop=True)
            y_all = y_all.loc[valid_target_mask].reset_index(drop=True)

        n_rows = len(train_df)
        n_models = len(self.base_model_specs)
        oof_base_preds = np.full((n_rows, n_models), np.nan, dtype=float)
        self.fitted_packs_ = []

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for model_idx, spec in enumerate(self.base_model_specs):
            name = spec.get("name", f"model_{model_idx}")
            model_type = spec["model_type"]
            pack = FittedBasePack(name=name, model_type=model_type)

            for fold, (train_idx, valid_idx) in enumerate(tscv.split(train_df), start=1):
                fold_train = train_df.iloc[train_idx].reset_index(drop=True)
                fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)

                preprocessor = self._make_preprocessor(spec)
                X_train = preprocessor.fit_transform(fold_train)
                y_train = pd.to_numeric(fold_train[TARGET_COL], errors="coerce")
                X_valid = preprocessor.transform(fold_valid)
                y_valid = pd.to_numeric(fold_valid[TARGET_COL], errors="coerce")

                model = self._make_model(spec, seed_add=fold * 1000)
                self._fit_model(model, model_type, X_train, y_train, X_valid, y_valid)

                fold_pred = np.asarray(model.predict(X_valid), dtype=float)
                fold_pred = np.clip(fold_pred, 0.0, available_capacity_from_df(fold_valid).to_numpy())
                oof_base_preds[valid_idx, model_idx] = fold_pred
                pack.fold_models.append((preprocessor, model))

            # Full model for inference.
            full_preprocessor = self._make_preprocessor(spec)
            X_full = full_preprocessor.fit_transform(train_df)
            full_model = self._make_model(spec, seed_add=99999)
            self._fit_model(full_model, model_type, X_full, y_all)
            pack.full_model = (full_preprocessor, full_model)
            self.fitted_packs_.append(pack)

        complete_oof_mask = ~np.isnan(oof_base_preds).any(axis=1)
        if complete_oof_mask.sum() == 0:
            raise RuntimeError("No complete OOF rows were produced. Reduce n_splits or check dataset size.")

        X_meta = self._augment_meta_features(oof_base_preds[complete_oof_mask])
        y_meta = y_all.loc[complete_oof_mask].to_numpy(dtype=float)

        self.meta_feature_names_ = [spec.get("name", f"model_{i}") for i, spec in enumerate(self.base_model_specs)]
        if self.append_meta_stats:
            self.meta_feature_names_ += ["base_mean", "base_std", "base_min", "base_max"]

        self.meta_model_ = self._make_meta_model()
        self.meta_model_.fit(X_meta, y_meta)

        oof_pred = np.asarray(self.meta_model_.predict(X_meta), dtype=float)
        oof_capacity = available_capacity_from_df(train_df.loc[complete_oof_mask]).to_numpy()
        oof_pred = np.clip(oof_pred, 0.0, oof_capacity)

        self.oof_mae_mw_ = float(mean_absolute_error(y_meta, oof_pred))
        self.oof_competition_error_percent_ = self.oof_mae_mw_ / INSTALLED_CAPACITY_MW * 100.0

        return self

    def _predict_base_matrix(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.fitted_packs_:
            raise RuntimeError("Stacking ensemble is not fitted.")

        features_df = features_df.reset_index(drop=True)
        base_preds: list[np.ndarray] = []

        for pack in self.fitted_packs_:
            preds_for_pack: list[np.ndarray] = []

            if self.prediction_mode in {"fold_mean", "blend_full_fold"}:
                for preprocessor, model in pack.fold_models:
                    X = preprocessor.transform(features_df)
                    preds_for_pack.append(np.asarray(model.predict(X), dtype=float))

            if self.prediction_mode in {"full", "blend_full_fold"}:
                if pack.full_model is None:
                    raise RuntimeError(f"Full model for pack {pack.name!r} is missing.")
                preprocessor, model = pack.full_model
                X = preprocessor.transform(features_df)
                preds_for_pack.append(np.asarray(model.predict(X), dtype=float))

            pack_pred = np.mean(np.vstack(preds_for_pack), axis=0)
            base_preds.append(pack_pred)

        base_matrix = np.vstack(base_preds).T
        row_capacity = available_capacity_from_df(features_df).to_numpy()
        base_matrix = np.clip(base_matrix, 0.0, row_capacity.reshape(-1, 1))
        return base_matrix

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if self.meta_model_ is None:
            raise RuntimeError("Stacking ensemble is not fitted.")

        base_matrix = self._predict_base_matrix(features_df)
        X_meta = self._augment_meta_features(base_matrix)
        pred = np.asarray(self.meta_model_.predict(X_meta), dtype=float)

        if self.clip_predictions:
            row_capacity = available_capacity_from_df(features_df).to_numpy()
            pred = np.clip(pred, 0.0, row_capacity)

        return pred

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TimeSeriesStackingEnsemble":
        return joblib.load(path)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "n_base_models": len(self.base_model_specs),
            "meta_model_type": self.meta_model_type,
            "n_splits": self.n_splits,
            "prediction_mode": self.prediction_mode,
            "append_meta_stats": self.append_meta_stats,
            "oof_mae_mw": self.oof_mae_mw_,
            "oof_competition_error_percent": self.oof_competition_error_percent_,
            "meta_feature_names": self.meta_feature_names_,
            "base_model_names": [spec.get("name", f"model_{i}") for i, spec in enumerate(self.base_model_specs)],
        }
