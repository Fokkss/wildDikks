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


DEFAULT_WEIGHTS_PATH = Path("configs_v1_6/final_v1_6_weights.json")


def _read_weights(path: Path | None, artifact_dir: Path) -> dict[str, Any]:
    # сначала берем json из артефактов, чтобы predict повторял train
    if path is None:
        artifact_path = artifact_dir / "final_v1_6_weights.json"
        if artifact_path.exists():
            path = artifact_path
        else:
            path = DEFAULT_WEIGHTS_PATH

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_submission(path: Path, pred: np.ndarray, *, header: bool, column: str) -> None:
    # пишем один столбец, без индекса
    path.parent.mkdir(parents=True, exist_ok=True)

    if header:
        pd.DataFrame({column: pred}).to_csv(path, index=False)
    else:
        pd.DataFrame(pred).to_csv(path, index=False, header=False)


def _prepare_matrix(raw: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    # повторяем те же признаки, с которыми была обучена конкретная модель
    fe = make_features(
        raw,
        target_col=None,
        use_lags=artifact.get("use_lags", True),
        use_time=artifact.get("use_time", True),
        use_availability=artifact.get("use_availability", True),
    )

    cols = artifact["feature_cols"]

    # на новом файле могут отсутствовать некоторые train-колонки
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
    # возвращаем версию с текущим капом и версию без жесткого ремонтного капа
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
    # считаем production-ансамбль из двух xgboost и catboost
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
    # base повторяет лучший production-прогноз v1
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
        return final_pred, {"postprocess": "base_only", "n": int(len(final_pred))}

    # raw_global нужен только для маленькой поправки верхнего хвоста
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

    # это встроенный аналог лучшего csv-бленда: 0.97 * base + 0.03 * caprelax10
    uncap_fraction = float(cap_cfg.get("caprelax_uncap_fraction", 0.10))
    relaxed_pred = base_pred + uncap_fraction * (raw_global - base_pred)

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
        "delta_mean": float(np.mean(delta)) if len(delta) else 0.0,
        "delta_abs_mean": float(np.mean(np.abs(delta))) if len(delta) else 0.0,
        "delta_abs_max": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "changed_gt_0p001": int((np.abs(delta) > 0.001).sum()),
        "base_max": float(np.max(base_pred)) if len(base_pred) else 0.0,
        "raw_global_max": float(np.max(raw_global)) if len(raw_global) else 0.0,
        "final_max": float(np.max(final_pred)) if len(final_pred) else 0.0,
    }

    return final_pred, report


def _is_empty_target(series: pd.Series) -> pd.Series:
    # пустые строки и nan считаем строками для прогноза
    text = series.astype("string")
    return series.isna() | text.str.strip().fillna("").eq("")


def select_day_rows(
    raw: pd.DataFrame,
    *,
    weights: dict[str, Any],
    target_col: str | None,
    datetime_col: str | None,
    date: str | None,
    expected_rows: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    # по умолчанию берем строки, где целевая колонка пустая
    day_cfg = weights.get("day_fill", {})
    target = target_col or day_cfg.get("target_col", "Выработка. Результирующий расчет")
    dt_col = datetime_col or day_cfg.get("datetime_col", "METEOFORECASTHOUR_OPENM_Datetime")

    if target in raw.columns:
        mask = _is_empty_target(raw[target])
        selection_reason = "empty_target"
    else:
        mask = pd.Series(False, index=raw.index)
        selection_reason = "target_column_missing"

    # дата нужна как страховка, если в файле целевая колонка заполнена пробелами не везде
    if date is not None:
        if dt_col not in raw.columns:
            raise ValueError(f"datetime column not found: {dt_col}")
        dt = pd.to_datetime(raw[dt_col], errors="coerce")
        date_mask = dt.dt.strftime("%Y-%m-%d").eq(date)
        if mask.sum() == 0:
            mask = date_mask
            selection_reason = "date_filter"
        else:
            mask = mask & date_mask
            selection_reason = "empty_target_and_date_filter"

    idx = np.flatnonzero(mask.to_numpy())

    if expected_rows is not None and len(idx) != expected_rows:
        raise ValueError(
            f"selected {len(idx)} rows, expected {expected_rows}; "
            "check target/date columns or pass --expected_rows 0"
        )

    report = {
        "target_col": target,
        "datetime_col": dt_col,
        "date": date,
        "selection_reason": selection_reason,
        "selected_rows": int(len(idx)),
        "selected_indices_first": idx[:5].tolist(),
        "selected_indices_last": idx[-5:].tolist(),
    }
    return idx, report


def predict_day_target_column(
    raw: pd.DataFrame,
    *,
    artifact_dir: Path,
    weights: dict[str, Any],
    target_col: str | None = None,
    datetime_col: str | None = None,
    date: str | None = None,
    expected_rows: int | None = 24,
) -> tuple[np.ndarray, dict[str, Any]]:
    # прогноз считаем на всей таблице, чтобы 17 число участвовало в lag-признаках
    full_pred, full_report = predict_final_array(
        raw,
        artifact_dir=artifact_dir,
        weights=weights,
    )

    idx, select_report = select_day_rows(
        raw,
        weights=weights,
        target_col=target_col,
        datetime_col=datetime_col,
        date=date,
        expected_rows=expected_rows,
    )

    day_pred = full_pred[idx]
    report = {
        "full_prediction_report": full_report,
        "selection_report": select_report,
        "output_rows": int(len(day_pred)),
        "output_min": float(np.min(day_pred)) if len(day_pred) else 0.0,
        "output_max": float(np.max(day_pred)) if len(day_pred) else 0.0,
        "output_mean": float(np.mean(day_pred)) if len(day_pred) else 0.0,
    }
    return day_pred, report


def main() -> None:
    parser = argparse.ArgumentParser(description="final v1.6 inference")
    parser.add_argument("--features_path", required=True, help="path to features csv")
    parser.add_argument("--artifact_dir", default="artifacts_v1_6", help="directory with trained artifacts")
    parser.add_argument("--output_path", default="submissions_v1_6/prediction_18_05.csv")
    parser.add_argument("--weights_path", default=None, help="optional final weights json")
    parser.add_argument("--mode", choices=["all", "day_target"], default="day_target")
    parser.add_argument("--target_col", default=None, help="target column to fill")
    parser.add_argument("--datetime_col", default=None, help="datetime column for optional date filter")
    parser.add_argument("--date", default=None, help="optional date filter, format yyyy-mm-dd")
    parser.add_argument("--expected_rows", type=int, default=24, help="expected output rows; use 0 to disable check")
    parser.add_argument("--header", dest="header", action="store_true", default=True)
    parser.add_argument("--no_header", dest="header", action="store_false")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    raw = read_csv(args.features_path)
    weights = _read_weights(Path(args.weights_path) if args.weights_path else None, artifact_dir)

    if args.mode == "all":
        pred, report = predict_final_array(raw, artifact_dir=artifact_dir, weights=weights)
        column = weights.get("output", {}).get("prediction_column", "prediction")
    else:
        expected_rows = None if args.expected_rows == 0 else args.expected_rows
        pred, report = predict_day_target_column(
            raw,
            artifact_dir=artifact_dir,
            weights=weights,
            target_col=args.target_col,
            datetime_col=args.datetime_col,
            date=args.date,
            expected_rows=expected_rows,
        )
        column = (
            args.target_col
            or weights.get("day_fill", {}).get("prediction_column")
            or weights.get("day_fill", {}).get("target_col")
            or "Выработка. Результирующий расчет"
        )

    output_path = Path(args.output_path)
    _write_submission(output_path, pred, header=args.header, column=column)

    summary(
        output_path.with_suffix(".summary.json"),
        pred,
        {
            "version": weights.get("version", "V1.6-final"),
            "kind": f"final_v1_6_{args.mode}",
            "weights": weights,
            "postprocess_report": report,
            "header": bool(args.header),
            "column": column,
        },
    )

    print(f"saved prediction: {output_path}")
    print(f"saved summary: {output_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
