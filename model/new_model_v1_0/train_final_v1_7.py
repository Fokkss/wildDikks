from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .predict_final_v1_7 import predict_to_file
from .train import _fit_catboost, _load_train_frame
from .train_predict import fit_model, make_sample_weights, set_seed


DEFAULT_WEIGHTS_PATH = Path("configs_v1_7/final_v1_7_weights.json")


def _load_weights(path: Path | None) -> dict[str, Any]:
    # веса лежат в json, чтобы итоговую смесь было легко проверить
    if path is None:
        path = DEFAULT_WEIGHTS_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def _save_weights(artifact_dir: Path, weights: dict[str, Any]) -> None:
    # копируем веса рядом с обученными моделями
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / "final_v1_7_weights.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # этот файл нужен для совместимости со старым production-кодом
    (artifact_dir / "model_v1_0_config.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _copy_weights_to_artifacts(weights_path: Path | None, artifact_dir: Path) -> dict[str, Any]:
    # если передан свой json, используем его вместо дефолтного
    weights = _load_weights(weights_path)
    _save_weights(artifact_dir, weights)
    return weights


def _safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    # тяжелые объекты модели не пишем в карточку
    return {
        k: v
        for k, v in meta.items()
        if k not in ("model", "imputer", "feature_cols")
    }


def train_artifacts(
    *,
    train_path: str | Path,
    target: str,
    artifact_dir: Path,
    weights: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    # фиксируем случайность перед обучением
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    raw, y, target_col = _load_train_frame(train_path, target)

    # valid нужен только для построения матрицы, модель обучается на всем train
    placeholder_valid = raw.head(1).copy()

    # основной устойчивый xgboost
    _, legacy_meta = fit_model(
        raw,
        y,
        placeholder_valid,
        target_col=target_col,
        name="legacy1700",
        seed=seed,
        n_estimators=1700,
        max_depth=3,
        use_lags=True,
        use_time=True,
        use_availability=True,
        sample_weight=make_sample_weights(raw, "none"),
        objective="reg:absoluteerror",
    )
    joblib.dump(legacy_meta, artifact_dir / "legacy1700.joblib")

    # более глубокий xgboost-компаньон
    _, depth4_meta = fit_model(
        raw,
        y,
        placeholder_valid,
        target_col=target_col,
        name="depth4_1892",
        seed=seed + 1,
        n_estimators=1892,
        max_depth=4,
        use_lags=True,
        use_time=True,
        use_availability=True,
        sample_weight=make_sample_weights(raw, "none"),
        objective="reg:absoluteerror",
    )
    joblib.dump(depth4_meta, artifact_dir / "depth4_1892.joblib")

    # catboost добавляет другой табличный сигнал
    cat_meta = _fit_catboost(raw, y, target_col=target_col, seed=seed + 43)
    joblib.dump(cat_meta, artifact_dir / "catboost_companion.joblib")

    model_card = {
        "version": weights.get("version", "V1.7-final"),
        "target_col": target_col,
        "seed": seed,
        "weights_path": str(artifact_dir / "final_v1_7_weights.json"),
        "weights": weights,
        "artifacts": [
            "legacy1700.joblib",
            "depth4_1892.joblib",
            "catboost_companion.joblib",
        ],
        "xgb_models": {
            "legacy1700": _safe_meta(legacy_meta),
            "depth4_1892": _safe_meta(depth4_meta),
        },
        "catboost_model": _safe_meta(cat_meta),
    }

    (artifact_dir / "model_v1_7_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return model_card


def main() -> None:
    parser = argparse.ArgumentParser(description="train final v1.7 model")
    parser.add_argument("--train_path", required=True, help="path to train csv")
    parser.add_argument("--features_path", default=None, help="optional csv for immediate prediction")
    parser.add_argument("--target", default="Выработка. Результирующий расчет", help="target column")
    parser.add_argument("--artifact_dir", default="artifacts_v1_7", help="where model weights are saved")
    parser.add_argument("--weights_path", default=None, help="optional final weights json")
    parser.add_argument("--output_path", default="submissions_v1_7/submission.csv")
    parser.add_argument(
        "--prediction_mode",
        choices=["normal", "generic", "day18"],
        default="normal",
        help="prediction mode used if --features_path is passed",
    )
    parser.add_argument("--output_col", default="prediction")
    parser.add_argument("--filled_table_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--header", dest="header", action="store_true", default=True)
    parser.add_argument("--no_header", dest="header", action="store_false")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    weights = _copy_weights_to_artifacts(
        Path(args.weights_path) if args.weights_path else None,
        artifact_dir,
    )

    train_artifacts(
        train_path=args.train_path,
        target=args.target,
        artifact_dir=artifact_dir,
        weights=weights,
        seed=args.seed,
    )

    # сразу делаем прогноз, если csv передан в команду обучения
    if args.features_path:
        predict_to_file(
            features_path=Path(args.features_path),
            artifact_dir=artifact_dir,
            output_path=Path(args.output_path),
            weights_path=artifact_dir / "final_v1_7_weights.json",
            mode=args.prediction_mode,
            header=bool(args.header),
            output_col=args.output_col,
            target_col=args.target,
            filled_table_path=Path(args.filled_table_path) if args.filled_table_path else None,
            strict_24=True,
        )
        print(f"saved prediction: {args.output_path}")

    print(f"saved artifacts: {artifact_dir}")
    print(f"saved weights: {artifact_dir / 'final_v1_7_weights.json'}")
    print(f"saved model card: {artifact_dir / 'model_v1_7_card.json'}")


if __name__ == "__main__":
    main()
