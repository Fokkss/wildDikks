from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import FARM_CAPACITY_MW, make_features
from .train_predict import clip_pred, profile_delta, read_csv, rule_delta, summary


DEFAULT_WEIGHTS_PATH = Path("configs_v1_7/final_v1_7_weights.json")
DEFAULT_TARGET_COL = "Выработка. Результирующий расчет"


def _read_weights(path: Path | None, artifact_dir: Path) -> dict[str, Any]:
    # сначала ищем веса рядом с артефактами, чтобы инференс был воспроизводимым
    if path is None:
        artifact_path = artifact_dir / "final_v1_7_weights.json"
        if artifact_path.exists():
            path = artifact_path
        else:
            path = DEFAULT_WEIGHTS_PATH

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_one_column(
    path: Path,
    values: np.ndarray,
    *,
    header: bool,
    column_name: str,
) -> None:
    # пишем ровно один столбец и не пишем индекс
    path.parent.mkdir(parents=True, exist_ok=True)

    if header:
        pd.DataFrame({column_name: values}).to_csv(path, index=False)
    else:
        pd.DataFrame(values).to_csv(path, index=False, header=False)


def _is_missing_target(series: pd.Series) -> pd.Series:
    # пустые строки и nan считаем строками для прогноза
    as_text = series.astype("string").str.strip()
    return series.isna() | as_text.isna() | as_text.eq("")


def _prepare_matrix(raw: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    # повторяем те же фичи, с которыми обучался конкретный артефакт
    fe = make_features(
        raw,
        target_col=None,
        use_lags=artifact.get("use_lags", True),
        use_time=artifact.get("use_time", True),
        use_availability=artifact.get("use_availability", True),
    )

    cols = artifact["feature_cols"]

    # на новом csv может не быть части train-признаков
    for col in cols:
        if col not in fe.columns:
            fe[col] = np.nan

    X = pd.DataFrame(artifact["imputer"].transform(fe[cols]), columns=cols)
    return X


def _predict_raw_model(artifact_path: Path, raw: pd.DataFrame) -> np.ndarray:
    # raw-прогноз нужен для мягкого снятия ремонтного потолка
    artifact = joblib.load(artifact_path)
    X = _prepare_matrix(raw, artifact)
    return np.asarray(artifact["model"].predict(X), dtype=float)


def _predict_component_pair(artifact_path: Path, raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # возвращаем прогноз с текущим капом и прогноз без ремонтного капа
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
    # считаем тот же production-ансамбль, но выбираем clipped или raw компоненты
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
    # base повторяет лучший production-прогноз до маленького cap-relax слоя
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

    # raw_global повторяет тот же ансамбль, но без ремонта как жесткого потолка
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

    # финал повторяет лучший результат без чтения старого csv
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


def _select_day18_rows(
    raw: pd.DataFrame,
    *,
    target_col: str,
    strict_24: bool,
) -> pd.Series:
    # для 18.05 берем строки с пустой колонкой выработки
    if target_col not in raw.columns:
        raise ValueError(
            f"target column {target_col!r} not found; day18 mode needs this column"
        )

    mask = _is_missing_target(raw[target_col])

    if strict_24 and int(mask.sum()) != 24:
        raise ValueError(
            f"day18 mode expected exactly 24 empty target rows, got {int(mask.sum())}"
        )

    return mask


def predict_to_file(
    *,
    features_path: Path,
    artifact_dir: Path,
    output_path: Path,
    weights_path: Path | None,
    mode: str,
    header: bool,
    output_col: str,
    target_col: str,
    filled_table_path: Path | None,
    strict_24: bool,
) -> dict[str, Any]:
    # одна функция обслуживает обычный сабмит, любой csv и режим 18.05
    raw = read_csv(features_path)
    weights = _read_weights(weights_path, artifact_dir)

    pred_all, report = predict_final_array(
        raw,
        artifact_dir=artifact_dir,
        weights=weights,
    )

    if mode == "day18":
        mask = _select_day18_rows(raw, target_col=target_col, strict_24=strict_24)
        pred_out = pred_all[mask.to_numpy()]
        column_name = target_col

        if filled_table_path is not None:
            # дополнительно можно сохранить исходную таблицу с заполненным 18.05
            filled = raw.copy()
            filled.loc[mask, target_col] = pred_out
            filled_table_path.parent.mkdir(parents=True, exist_ok=True)
            filled.to_csv(filled_table_path, index=False)
    elif mode in {"normal", "generic"}:
        pred_out = pred_all
        column_name = output_col
    else:
        raise ValueError("mode must be one of: normal, generic, day18")

    _write_one_column(
        output_path,
        pred_out,
        header=header,
        column_name=column_name,
    )

    out_report = {
        "version": weights.get("version", "V1.7-final"),
        "mode": mode,
        "input_path": str(features_path),
        "output_path": str(output_path),
        "n_input_rows": int(len(raw)),
        "n_output_rows": int(len(pred_out)),
        "output_column": column_name,
        "header": bool(header),
        "weights": weights,
        "postprocess_report": report,
    }

    summary(
        output_path.with_suffix(".summary.json"),
        pred_out,
        out_report,
    )

    return out_report


def main() -> None:
    parser = argparse.ArgumentParser(description="final v1.7 inference")
    parser.add_argument("--features_path", required=True, help="path to features csv")
    parser.add_argument("--artifact_dir", default="artifacts_v1_7", help="directory with trained artifacts")
    parser.add_argument("--output_path", default="submissions_v1_7/submission.csv")
    parser.add_argument("--weights_path", default=None, help="optional final weights json")
    parser.add_argument(
        "--mode",
        choices=["normal", "generic", "day18"],
        default="normal",
        help="normal/generic predicts all rows; day18 outputs rows with empty target",
    )
    parser.add_argument("--output_col", default="prediction", help="column name for normal/generic output")
    parser.add_argument("--target_col", default=DEFAULT_TARGET_COL, help="target column for day18 mode")
    parser.add_argument("--filled_table_path", default=None, help="optional full csv with filled day18 target")
    parser.add_argument("--strict_24", dest="strict_24", action="store_true", default=True)
    parser.add_argument("--no_strict_24", dest="strict_24", action="store_false")
    parser.add_argument("--header", dest="header", action="store_true", default=True)
    parser.add_argument("--no_header", dest="header", action="store_false")
    args = parser.parse_args()

    report = predict_to_file(
        features_path=Path(args.features_path),
        artifact_dir=Path(args.artifact_dir),
        output_path=Path(args.output_path),
        weights_path=Path(args.weights_path) if args.weights_path else None,
        mode=args.mode,
        header=args.header,
        output_col=args.output_col,
        target_col=args.target_col,
        filled_table_path=Path(args.filled_table_path) if args.filled_table_path else None,
        strict_24=bool(args.strict_24),
    )

    print(f"saved prediction: {args.output_path}")
    print(f"saved summary: {Path(args.output_path).with_suffix('.summary.json')}")
    print(f"mode: {report['mode']}; rows: {report['n_output_rows']}")


if __name__ == "__main__":
    main()
