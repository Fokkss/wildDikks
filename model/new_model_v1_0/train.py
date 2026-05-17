from __future__ import annotations

# этот кусок загружает аргументы, json и воспроизводимые random seed
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# эта зона для XGBoost anchor и общей feature engineering логики
from .feature_engineering import (
    available_capacity_from_raw,
    find_datetime_col,
    find_target_col,
    make_features,
    sort_by_time_if_possible,
)
from .predict import predict_with_v1_pipeline, write_submission
from .train_predict import (
    clip_pred,
    fit_model,
    make_sample_weights,
    read_csv,
    select_features,
    set_seed,
    summary,
)


def _fit_catboost(
    raw: pd.DataFrame,
    y: pd.Series,
    *,
    target_col: str,
    seed: int,
) -> dict[str, Any]:
    """Train CatBoost companion model on the same physics-aware features."""

    # этот кусок импортирует CatBoost только здесь, чтобы ошибка зависимости была понятной
    try:
        from catboost import CatBoostRegressor
    except Exception as exc:
        raise SystemExit(
            "CatBoost is required for V1.0 production model. Install dependencies from requirements.txt. "
            f"Original error: {exc!r}"
        )

    # эта зона строит признаки ровно так же, как XGBoost anchor
    fe = make_features(raw, target_col=target_col, use_lags=True, use_time=True, use_availability=True)
    cols = select_features(fe, target_col)
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(fe[cols]), columns=cols)

    # этот кусок обучает CatBoost MAE-модель как компаньон, а не как замену XGBoost
    model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=7.0,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)

    return {
        "model": model,
        "imputer": imputer,
        "feature_cols": cols,
        "target_col": target_col,
        "use_lags": True,
        "use_time": True,
        "use_availability": True,
        "name": "catboost_companion",
        "iterations": 2500,
        "depth": 6,
        "learning_rate": 0.03,
        "n_features": len(cols),
    }


def _save_config(artifact_dir: Path, config_path: Path | None) -> dict[str, Any]:
    """Copy final blend config into artifacts for reproducible inference."""

    # этот кусок фиксирует финальные веса лучшего production-профиля
    if config_path is None:
        config_path = Path("configs_v1_0/final_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model_v1_0_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config


def _load_train_frame(train_path: str | Path, target_arg: str) -> tuple[pd.DataFrame, pd.Series, str]:
    """Read train CSV, find target, sort by time and apply physical target clipping."""

    # эта зона читает train и приводит битую колонку ремонта к единому имени
    raw = read_csv(train_path).rename(columns={"Кол-во_ВЭУ_в_ремонте": "repair_count"})
    target_col = find_target_col(raw, target_arg)

    # этот кусок сортирует train по времени только внутри обучения; порядок valid/test не трогаем
    raw = sort_by_time_if_possible(raw)

    # эта зона клиппит target в физически допустимый диапазон станции
    y_raw = pd.to_numeric(raw[target_col], errors="coerce")
    cap = available_capacity_from_raw(raw)
    y = y_raw.clip(lower=0.0, upper=cap)
    ok = y.notna()

    return raw.loc[ok].reset_index(drop=True), y.loc[ok].reset_index(drop=True), target_col


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final V1.0 wind power production model")
    parser.add_argument("--train_path", required=True, help="Path to train_dataset.csv")
    parser.add_argument("--valid_path", default=None, help="Optional valid/test features CSV for immediate prediction")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--artifact_dir", default="artifacts_v1_0")
    parser.add_argument("--output_path", default="submissions_v1_0/submission_v1_0.csv")
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--header", dest="header", action="store_true", default=True, help="Write CSV header 'prediction'. Default: enabled.")
    parser.add_argument("--no_header", dest="header", action="store_false", help="Write CSV without header.")
    args = parser.parse_args()

    # этот кусок фиксирует random seed для воспроизводимости
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # эта зона готовит train target и финальный config
    raw, y, target_col = _load_train_frame(args.train_path, args.target)
    config = _save_config(artifact_dir, Path(args.config_path) if args.config_path else None)

    # этот кусок обучает XGBoost legacy1700: сильный неглубокий регрессор
    placeholder_valid = raw.head(1).copy()
    legacy_pred_dummy, legacy_meta = fit_model(
        raw,
        y,
        placeholder_valid,
        target_col=target_col,
        name="legacy1700",
        seed=args.seed,
        n_estimators=1700,
        max_depth=3,
        use_lags=True,
        use_time=True,
        use_availability=True,
        sample_weight=make_sample_weights(raw, "none"),
        objective="reg:absoluteerror",
    )
    joblib.dump(legacy_meta, artifact_dir / "legacy1700.joblib")

    # эта зона обучает depth4-компаньон для смешивания с legacy1700
    depth4_pred_dummy, depth4_meta = fit_model(
        raw,
        y,
        placeholder_valid,
        target_col=target_col,
        name="depth4_1892",
        seed=args.seed + 1,
        n_estimators=1892,
        max_depth=4,
        use_lags=True,
        use_time=True,
        use_availability=True,
        sample_weight=make_sample_weights(raw, "none"),
        objective="reg:absoluteerror",
    )
    joblib.dump(depth4_meta, artifact_dir / "depth4_1892.joblib")

    # этот кусок обучает CatBoost companion, который улучшил ансамбль на valid leaderboard
    cat_meta = _fit_catboost(raw, y, target_col=target_col, seed=args.seed + 43)
    joblib.dump(cat_meta, artifact_dir / "catboost_companion.joblib")

    # эта зона сохраняет компактное описание весов и гиперпараметров
    model_card = {
        "version": "V1.0",
        "target_col": target_col,
        "seed": args.seed,
        "config": config,
        "artifacts": ["legacy1700.joblib", "depth4_1892.joblib", "catboost_companion.joblib"],
        "xgb_models": {
            "legacy1700": {k: v for k, v in legacy_meta.items() if k not in ("model", "imputer", "feature_cols")},
            "depth4_1892": {k: v for k, v in depth4_meta.items() if k not in ("model", "imputer", "feature_cols")},
        },
        "catboost_model": {k: v for k, v in cat_meta.items() if k not in ("model", "imputer", "feature_cols")},
    }
    (artifact_dir / "model_v1_0_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # этот кусок, если передан valid/test, сразу формирует один итоговый submission
    if args.valid_path:
        valid_raw = read_csv(args.valid_path)
        pred = predict_with_v1_pipeline(valid_raw, artifact_dir=artifact_dir, config=config)
        output_path = Path(args.output_path)
        write_submission(output_path, pred, header=args.header)
        summary(output_path.with_suffix(".summary.json"), pred, {
            "version": "V1.0",
            "kind": "train_then_predict_once",
            "config": config,
            "header": bool(args.header),
        })
        print(f"Saved prediction: {output_path}")

    print(f"Saved trained artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
