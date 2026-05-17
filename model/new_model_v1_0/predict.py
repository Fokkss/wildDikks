from __future__ import annotations

# этот кусок загружает аргументы командной строки и базовые библиотеки
import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# эта зона для общей feature engineering логики и физического клиппинга
from .feature_engineering import make_features
from .train_predict import (
    clip_pred,
    profile_delta,
    read_csv,
    rule_delta,
    summary,
)


def write_submission(path: Path, pred: np.ndarray, *, header: bool) -> None:
    """Save one-column CSV in the competition format."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # этот кусок делает официальный формат: один столбец, без индекса
    if header:
        pd.DataFrame({"prediction": pred}).to_csv(path, index=False)
    else:
        pd.DataFrame(pred).to_csv(path, index=False, header=False)


def _prepare_matrix(raw: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Build feature matrix exactly as during training."""

    # эта зона для повторения тех же флагов feature engineering, что были при train
    fe = make_features(
        raw,
        target_col=None,
        use_lags=artifact.get("use_lags", True),
        use_time=artifact.get("use_time", True),
        use_availability=artifact.get("use_availability", True),
    )

    # этот кусок выравнивает valid/test признаки под train-колонки
    cols = artifact["feature_cols"]
    for col in cols:
        if col not in fe.columns:
            fe[col] = np.nan

    return pd.DataFrame(artifact["imputer"].transform(fe[cols]), columns=cols)


def _predict_model(artifact_path: Path, raw: pd.DataFrame) -> np.ndarray:
    """Load one trained model artifact and predict clipped MW values."""

    # этот кусок загружает веса модели и медианный imputer
    artifact = joblib.load(artifact_path)
    X = _prepare_matrix(raw, artifact)

    # эта зона для физического ограничения прогноза по мощности станции
    return clip_pred(raw, artifact["model"].predict(X))


def load_config(config_path: Path | None, artifact_dir: Path) -> dict[str, Any]:
    """Load final V1.0 blend/rule configuration."""

    # этот кусок сначала ищет конфиг в artifacts, чтобы predict был самодостаточным
    if config_path is None:
        config_path = artifact_dir / "model_v1_0_config.json"
    if not config_path.exists():
        config_path = Path("configs_v1_0/final_config.json")
    return json.loads(config_path.read_text(encoding="utf-8"))


def predict_with_v1_pipeline(
    raw: pd.DataFrame,
    *,
    artifact_dir: Path,
    config: dict[str, Any],
) -> np.ndarray:
    """Final production predictor: XGB anchor + CatBoost companion + rule layer."""

    # этот кусок считает два XGBoost компонента production anchor
    legacy1700 = _predict_model(artifact_dir / "legacy1700.joblib", raw)
    depth4 = _predict_model(artifact_dir / "depth4_1892.joblib", raw)

    xgb_cfg = config["xgb_anchor"]
    anchor = (
        xgb_cfg["legacy1700_weight"] * legacy1700
        + xgb_cfg["depth4_1892_weight"] * depth4
    )

    # эта зона добавляет CatBoost companion, который дал комплементарный сигнал
    catboost_pred = _predict_model(artifact_dir / "catboost_companion.joblib", raw)
    blend_cfg = config["catboost_blend"]
    base_pred = (
        blend_cfg["anchor_weight"] * anchor
        + blend_cfg["catboost_weight"] * catboost_pred
    )

    # этот кусок применяет устойчивую физическую calibration-map без teacher CSV
    rule_cfg = config["rule_layer"]
    pred = (
        base_pred
        + rule_delta(raw, base_pred, rule_cfg["rules_strength"])
        + profile_delta(raw, base_pred, rule_cfg["profile"])
        + rule_cfg["bias_mw"]
    )

    return clip_pred(raw, pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="V1.0 inference for wind farm generation forecast")
    parser.add_argument("--features_path", required=True, help="Path to valid/test features CSV")
    parser.add_argument("--artifact_dir", default="artifacts_v1_0", help="Directory with trained weights")
    parser.add_argument("--output_path", default="submissions_v1_0/submission_v1_0.csv")
    parser.add_argument("--config_path", default=None, help="Optional config JSON path")
    parser.add_argument("--header", dest="header", action="store_true", default=True, help="Write CSV header 'prediction'. Default: enabled.")
    parser.add_argument("--no_header", dest="header", action="store_false", help="Write CSV without header.")
    args = parser.parse_args()

    # эта зона для загрузки данных и финального конфига
    raw = read_csv(args.features_path)
    artifact_dir = Path(args.artifact_dir)
    config = load_config(Path(args.config_path) if args.config_path else None, artifact_dir)

    # этот кусок формирует один итоговый прогноз
    pred = predict_with_v1_pipeline(raw, artifact_dir=artifact_dir, config=config)

    # эта зона сохраняет submission и summary для контроля
    output_path = Path(args.output_path)
    write_submission(output_path, pred, header=args.header)
    summary(output_path.with_suffix(".summary.json"), pred, {
        "version": "V1.0",
        "kind": "xgb_anchor_catboost_companion",
        "config": config,
        "header": bool(args.header),
    })
    print(f"Saved prediction: {output_path}")


if __name__ == "__main__":
    main()
