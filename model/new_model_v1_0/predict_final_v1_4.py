from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import FARM_CAPACITY_MW, make_features
from .predict import load_config
from .train_predict import clip_pred, profile_delta, read_csv, rule_delta, summary


DEFAULT_WEIGHTS_PATH = Path("configs_v1_4/final_v1_4_weights.json")


def _read_weights(path: Path | None, artifact_dir: Path) -> dict[str, Any]:
    # сначала берем веса из артефактов, чтобы predict был воспроизводимым после train
    if path is None:
        artifact_path = artifact_dir / "final_v1_4_weights.json"
        if artifact_path.exists():
            path = artifact_path
        else:
            path = DEFAULT_WEIGHTS_PATH

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_submission(path: Path, pred: np.ndarray, *, header: bool) -> None:
    # сохраняем ровно один столбец, индекс не пишем
    path.parent.mkdir(parents=True, exist_ok=True)

    if header:
        pd.DataFrame({"prediction": pred}).to_csv(path, index=False)
    else:
        pd.DataFrame(pred).to_csv(path, index=False, header=False)


def _prepare_matrix(raw: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    # повторяем feature engineering с теми же флагами, что были при обучении модели
    fe = make_features(
        raw,
        target_col=None,
        use_lags=artifact.get("use_lags", True),
        use_time=artifact.get("use_time", True),
        use_availability=artifact.get("use_availability", True),
    )

    cols = artifact["feature_cols"]

    # если на новом датасете нет train-колонки, создаем ее как пропуск
    for col in cols:
        if col not in fe.columns:
            fe[col] = np.nan

    X = pd.DataFrame(artifact["imputer"].transform(fe[cols]), columns=cols)
    return X


def _predict_raw_model(artifact_path: Path, raw: pd.DataFrame) -> np.ndarray:
    # raw-прогноз нужен только для мягкого снятия ремонтного потолка на верхнем хвосте
    artifact = joblib.load(artifact_path)
    X = _prepare_matrix(raw, artifact)
    return np.asarray(artifact["model"].predict(X), dtype=float)


def _predict_component_pair(artifact_path: Path, raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # возвращаем и обычный прогноз с текущим капом, и raw-прогноз без ремонтного капа
    raw_pred = _predict_raw_model(artifact_path, raw)
    clipped_pred = clip_pred(raw, raw_pred)
    return clipped_pred, raw_pred


def _blend_components(
    raw: pd.DataFrame,
    artifact_dir: Path,
    weights: dict[str, Any],
    *,
    use_raw_components: bool,
) -> np.ndarray:
    # считаем тот же production-ансамбль, но можем выбрать clipped или raw компоненты
    legacy_clipped, legacy_raw = _predict_component_pair(artifact_dir / "legacy1700.joblib", raw)
    depth4_clipped, depth4_raw = _predict_component_pair(artifact_dir / "depth4_1892.joblib", raw)
    cat_clipped, cat_raw = _predict_component_pair(artifact_dir / "catboost_companion.joblib", raw)

    if use_raw_components:
        legacy = legacy_raw
        depth4 = depth4_raw
        catboost_pred = cat_raw
    else:
        legacy = legacy_clipped
        depth4 = depth4_clipped
        catboost_pred = cat_clipped

    xgb_cfg = weights["xgb_anchor"]
    anchor = (
        xgb_cfg["legacy1700_weight"] * legacy
        + xgb_cfg["depth4_1892_weight"] * depth4
    )

    blend_cfg = weights["catboost_blend"]
    base_pred = (
        blend_cfg["anchor_weight"] * anchor
        + blend_cfg["catboost_weight"] * catboost_pred
    )

    rule_cfg = weights["rule_layer"]
    pred = (
        base_pred
        + rule_delta(raw, base_pred, rule_cfg["rules_strength"])
        + profile_delta(raw, base_pred, rule_cfg["profile"])
        + rule_cfg["bias_mw"]
    )

    return np.asarray(pred, dtype=float)


def predict_final_array(
    raw: pd.DataFrame,
    *,
    artifact_dir: Path,
    weights: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    # base полностью повторяет v1_1/v1_2 production-предсказание
    base_raw = _blend_components(
        raw,
        artifact_dir,
        weights,
        use_raw_components=False,
    )
    base_pred = clip_pred(raw, base_raw)

    cap_cfg = weights.get("cap_relax", {})

    if not cap_cfg.get("enabled", True):
        final_pred = np.clip(base_pred, 0.0, FARM_CAPACITY_MW)
        report = {
            "postprocess": "base_only",
            "n": int(len(final_pred)),
        }
        return final_pred, report

    # raw_global повторяет тот же ensemble, но без ремонта как жесткого потолка
    raw_global = _blend_components(
        raw,
        artifact_dir,
        weights,
        use_raw_components=True,
    )
    raw_global = np.clip(
        raw_global,
        float(cap_cfg.get("global_min_mw", 0.0)),
        float(cap_cfg.get("global_max_mw", FARM_CAPACITY_MW)),
    )

    # caprelax10 = base + 0.10 * (raw_global - base)
    uncap_fraction = float(cap_cfg.get("caprelax_uncap_fraction", 0.10))
    relaxed_pred = base_pred + uncap_fraction * (raw_global - base_pred)

    # финальный вариант повторяет лучший leaderboard-бленд: 0.97 * base + 0.03 * caprelax10
    base_weight = float(cap_cfg.get("base_weight", 0.97))
    relaxed_weight = float(cap_cfg.get("relaxed_weight", 0.03))
    total = base_weight + relaxed_weight
    if total <= 0:
        raise ValueError("cap relax weights must have positive sum")

    final_pred = (base_weight * base_pred + relaxed_weight * relaxed_pred) / total
    final_pred = np.clip(
        final_pred,
        float(cap_cfg.get("global_min_mw", 0.0)),
        float(cap_cfg.get("global_max_mw", FARM_CAPACITY_MW)),
    )

    delta = final_pred - base_pred
    report = {
        "postprocess": "base_plus_internal_caprelax_blend",
        "base_weight": base_weight,
        "relaxed_weight": relaxed_weight,
        "caprelax_uncap_fraction": uncap_fraction,
        "equivalent_unclipped_global_weight": relaxed_weight * uncap_fraction / total,
        "n": int(len(final_pred)),
        "delta_mean": float(np.mean(delta)),
        "delta_abs_mean": float(np.mean(np.abs(delta))),
        "delta_abs_max": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "changed_gt_0p001": int((np.abs(delta) > 0.001).sum()),
        "base_max": float(np.max(base_pred)) if len(base_pred) else 0.0,
        "raw_global_max": float(np.max(raw_global)) if len(raw_global) else 0.0,
        "final_max": float(np.max(final_pred)) if len(final_pred) else 0.0,
    }

    return final_pred, report


def main() -> None:
    parser = argparse.ArgumentParser(description="final v1.4 inference")
    parser.add_argument("--features_path", required=True, help="path to features csv")
    parser.add_argument("--artifact_dir", default="artifacts_v1_4", help="directory with trained artifacts")
    parser.add_argument("--output_path", default="submissions_v1_4/submission_final_v1_4.csv")
    parser.add_argument("--weights_path", default=None, help="optional final weights json")
    parser.add_argument("--header", dest="header", action="store_true", default=True)
    parser.add_argument("--no_header", dest="header", action="store_false")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    raw = read_csv(args.features_path)
    weights = _read_weights(Path(args.weights_path) if args.weights_path else None, artifact_dir)

    pred, report = predict_final_array(
        raw,
        artifact_dir=artifact_dir,
        weights=weights,
    )

    output_path = Path(args.output_path)
    _write_submission(output_path, pred, header=args.header)

    summary(
        output_path.with_suffix(".summary.json"),
        pred,
        {
            "version": weights.get("version", "V1.4-final"),
            "kind": "final_reproducible_inference",
            "weights": weights,
            "postprocess_report": report,
            "header": bool(args.header),
        },
    )

    print(f"saved prediction: {output_path}")
    print(f"saved summary: {output_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
