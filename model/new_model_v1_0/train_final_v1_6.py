from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .predict_final_v1_6 import (
    _write_submission,
    predict_day_target_column,
    predict_final_array,
)
from .train import _fit_catboost, _load_train_frame
from .train_predict import fit_model, make_sample_weights, read_csv, set_seed, summary


DEFAULT_WEIGHTS_PATH = Path("configs_v1_6/final_v1_6_weights.json")


def _load_weights(path: Path | None) -> dict[str, Any]:
    # веса лежат в json, чтобы финальный postprocess был явно зафиксирован
    if path is None:
        path = DEFAULT_WEIGHTS_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def _save_weights(artifact_dir: Path, weights: dict[str, Any]) -> None:
    # сохраняем копию весов рядом с обученными моделями
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / "final_v1_6_weights.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # этот файл оставлен для совместимости со старым кодом v1
    (artifact_dir / "model_v1_0_config.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _copy_weights_to_artifacts(weights_path: Path | None, artifact_dir: Path) -> dict[str, Any]:
    # если передан свой json, он становится источником истины для predict
    weights = _load_weights(weights_path)
    _save_weights(artifact_dir, weights)
    return weights


def _safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    # большие python-объекты не кладем в человекочитаемую карточку
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
    # фиксируем случайность перед обучением всех моделей
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    raw, y, target_col = _load_train_frame(train_path, target)

    # валидационный placeholder нужен старому fit_model, но модель обучается на всех строках
    placeholder_valid = raw.head(1).copy()

    # основной неглубокий xgboost дал самый стабильный вклад на leaderboard
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

    # второй xgboost чуть глубже и добавляет разнообразие без смены подхода
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

    # catboost оставлен как независимый табличный компаньон
    cat_meta = _fit_catboost(raw, y, target_col=target_col, seed=seed + 43)
    joblib.dump(cat_meta, artifact_dir / "catboost_companion.joblib")

    model_card = {
        "version": weights.get("version", "V1.6-final"),
        "target_col": target_col,
        "seed": seed,
        "weights_path": str(artifact_dir / "final_v1_6_weights.json"),
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

    (artifact_dir / "model_v1_6_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return model_card


def main() -> None:
    parser = argparse.ArgumentParser(description="train final v1.6 model")
    parser.add_argument("--train_path", required=True, help="path to train csv")
    parser.add_argument("--features_path", default=None, help="optional features csv for immediate prediction")
    parser.add_argument("--target", default="Выработка. Результирующий расчет", help="target column")
    parser.add_argument("--artifact_dir", default="artifacts_v1_6", help="where model weights are saved")
    parser.add_argument("--weights_path", default=None, help="optional final weights json")
    parser.add_argument("--output_path", default="submissions_v1_6/prediction_18_05.csv")
    parser.add_argument("--prediction_mode", choices=["day_target", "all"], default="day_target")
    parser.add_argument("--date", default=None, help="optional date filter, format yyyy-mm-dd")
    parser.add_argument("--expected_rows", type=int, default=24, help="expected prediction rows; use 0 to disable")
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

    # сразу создаем прогноз, если дан файл с 17 и 18 числом
    if args.features_path:
        raw = read_csv(args.features_path)

        if args.prediction_mode == "all":
            pred, report = predict_final_array(raw, artifact_dir=artifact_dir, weights=weights)
            column = weights.get("output", {}).get("prediction_column", "prediction")
        else:
            expected_rows = None if args.expected_rows == 0 else args.expected_rows
            pred, report = predict_day_target_column(
                raw,
                artifact_dir=artifact_dir,
                weights=weights,
                target_col=args.target,
                date=args.date,
                expected_rows=expected_rows,
            )
            column = args.target

        output_path = Path(args.output_path)
        _write_submission(output_path, pred, header=args.header, column=column)

        summary(
            output_path.with_suffix(".summary.json"),
            pred,
            {
                "version": weights.get("version", "V1.6-final"),
                "kind": f"train_then_predict_v1_6_{args.prediction_mode}",
                "weights": weights,
                "postprocess_report": report,
                "header": bool(args.header),
                "column": column,
            },
        )

        print(f"saved prediction: {output_path}")

    print(f"saved artifacts: {artifact_dir}")
    print(f"saved weights: {artifact_dir / 'final_v1_6_weights.json'}")
    print(f"saved model card: {artifact_dir / 'model_v1_6_card.json'}")


if __name__ == "__main__":
    main()
