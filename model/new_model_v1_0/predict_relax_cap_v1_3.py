from __future__ import annotations

"""
V1.3 no-retrain candidates around current best V1.1/V1.2.

Main idea:
    Current best prediction is capped around available_capacity_mw and has max ~79.695 MW.
    This script keeps the same trained artifacts/config, but generates candidates that
    relax only the final hard repair-cap clipping while preserving global physical cap 0..90.09.

Run from model/:
    python -m new_model_v1_0.predict_relax_cap_v1_3 \
      --features_path ../data/valid_features.csv \
      --artifact_dir artifacts_v1_0 \
      --output_dir submissions_v1_3 \
      --report_dir reports_v1_3 \
      --header
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import FARM_CAPACITY_MW, make_features
from .predict import load_config, predict_with_v1_pipeline, write_submission
from .train_predict import profile_delta, read_csv, rule_delta, summary


# ---------------------------------------------------------------------
# Matrix preparation copied intentionally from predict.py logic, but here
# we do NOT call clip_pred() inside base model predictions.
# ---------------------------------------------------------------------

def _prepare_matrix(raw: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    fe = make_features(
        raw,
        target_col=None,
        use_lags=artifact.get("use_lags", True),
        use_time=artifact.get("use_time", True),
        use_availability=artifact.get("use_availability", True),
    )

    cols = artifact["feature_cols"]
    for col in cols:
        if col not in fe.columns:
            fe[col] = np.nan

    return pd.DataFrame(artifact["imputer"].transform(fe[cols]), columns=cols)


def _predict_model_unclipped(artifact_path: Path, raw: pd.DataFrame) -> np.ndarray:
    artifact = joblib.load(artifact_path)
    X = _prepare_matrix(raw, artifact)
    pred = np.asarray(artifact["model"].predict(X), dtype=float)
    return np.clip(pred, 0.0, FARM_CAPACITY_MW)


def predict_unclipped_v1_pipeline(
    raw: pd.DataFrame,
    *,
    artifact_dir: Path,
    config: dict[str, Any],
) -> np.ndarray:
    """
    Same V1 pipeline as predict.py, but without row-wise available_capacity clip.
    Only global physical clip 0..90.09 is applied.
    """
    legacy1700 = _predict_model_unclipped(artifact_dir / "legacy1700.joblib", raw)
    depth4 = _predict_model_unclipped(artifact_dir / "depth4_1892.joblib", raw)

    xgb_cfg = config["xgb_anchor"]
    anchor = (
        xgb_cfg["legacy1700_weight"] * legacy1700
        + xgb_cfg["depth4_1892_weight"] * depth4
    )

    catboost_pred = _predict_model_unclipped(artifact_dir / "catboost_companion.joblib", raw)
    blend_cfg = config["catboost_blend"]
    base_pred = (
        blend_cfg["anchor_weight"] * anchor
        + blend_cfg["catboost_weight"] * catboost_pred
    )

    rule_cfg = config["rule_layer"]
    pred = (
        base_pred
        + rule_delta(raw, base_pred, rule_cfg["rules_strength"])
        + profile_delta(raw, base_pred, rule_cfg["profile"])
        + rule_cfg["bias_mw"]
    )

    return np.clip(pred, 0.0, FARM_CAPACITY_MW)


# ---------------------------------------------------------------------
# Tail post-processing. Conservative: changes mostly rows that are already
# high generation candidates.
# ---------------------------------------------------------------------

def _safe_feature(raw: pd.DataFrame, name: str) -> pd.Series:
    fe = make_features(raw, target_col=None, use_lags=True, use_time=True, use_availability=True)
    if name in fe.columns:
        return pd.to_numeric(fe[name], errors="coerce")
    return pd.Series(np.nan, index=raw.index)


def _gate(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def tail_lift(
    pred: np.ndarray,
    raw: pd.DataFrame,
    *,
    amount_mw: float,
    pred_lo: float = 61.0,
    pred_hi: float = 78.0,
    wind_lo: float = 9.5,
    wind_hi: float = 12.5,
    only_cap_zone: bool = False,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    hub = _safe_feature(raw, "hub_ws_84m").to_numpy(dtype=float)

    pred_gate = _gate(pred, pred_lo, pred_hi)
    if np.isfinite(hub).any():
        wind_gate = _gate(np.nan_to_num(hub, nan=np.nanmedian(hub)), wind_lo, wind_hi)
    else:
        wind_gate = pred_gate

    if only_cap_zone:
        cap_gate = (pred >= 78.5).astype(float)
    else:
        cap_gate = 1.0

    lift = amount_mw * pred_gate * wind_gate * cap_gate
    return np.clip(pred + lift, 0.0, FARM_CAPACITY_MW)


def low_wind_trim(
    pred: np.ndarray,
    raw: pd.DataFrame,
    *,
    amount_mw: float,
    wind_hi: float = 4.5,
    pred_hi: float = 15.0,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    hub = _safe_feature(raw, "hub_ws_84m").to_numpy(dtype=float)

    if np.isfinite(hub).any():
        wind_gate = 1.0 - _gate(np.nan_to_num(hub, nan=np.nanmedian(hub)), 3.0, wind_hi)
    else:
        wind_gate = 1.0 - _gate(pred, 3.0, pred_hi)

    pred_gate = 1.0 - _gate(pred, 3.0, pred_hi)
    trim = amount_mw * wind_gate * pred_gate
    return np.clip(pred - trim, 0.0, FARM_CAPACITY_MW)


def make_candidates(raw: pd.DataFrame, clipped: np.ndarray, unclipped: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    # Main cap-relax candidates: blend old clipped best with globally-clipped uncap pipeline.
    for w_uncap in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        w_clip = 1.0 - w_uncap
        out[f"v1_3_caprelax_uncap{int(w_uncap*100):02d}"] = np.clip(
            w_clip * clipped + w_uncap * unclipped,
            0.0,
            FARM_CAPACITY_MW,
        )

    # Pure global cap pipeline.
    out["v1_3_unclipped_global"] = unclipped

    # Tail-lift candidates from current best clipped prediction.
    out["v1_3_tail_lift0p6"] = tail_lift(clipped, raw, amount_mw=0.6)
    out["v1_3_tail_lift1p0"] = tail_lift(clipped, raw, amount_mw=1.0)
    out["v1_3_tail_lift1p4"] = tail_lift(clipped, raw, amount_mw=1.4)
    out["v1_3_tail_lift2p0"] = tail_lift(clipped, raw, amount_mw=2.0)

    # Only rows already stuck at the repair cap get relaxed.
    out["v1_3_capzone_lift1p0"] = tail_lift(clipped, raw, amount_mw=1.0, only_cap_zone=True)
    out["v1_3_capzone_lift2p0"] = tail_lift(clipped, raw, amount_mw=2.0, only_cap_zone=True)
    out["v1_3_capzone_lift3p0"] = tail_lift(clipped, raw, amount_mw=3.0, only_cap_zone=True)

    # Low-wind trim + small tail lift: protects from common overprediction in calm hours.
    base = tail_lift(clipped, raw, amount_mw=0.8)
    out["v1_3_lowtrim0p4_tail0p8"] = low_wind_trim(base, raw, amount_mw=0.4)
    out["v1_3_lowtrim0p8_tail0p8"] = low_wind_trim(base, raw, amount_mw=0.8)

    return out


def prediction_stats(pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    return {
        "n": int(len(pred)),
        "mean": float(np.mean(pred)),
        "std": float(np.std(pred)),
        "min": float(np.min(pred)),
        "q01": float(np.quantile(pred, 0.01)),
        "q05": float(np.quantile(pred, 0.05)),
        "q50": float(np.quantile(pred, 0.50)),
        "q95": float(np.quantile(pred, 0.95)),
        "q99": float(np.quantile(pred, 0.99)),
        "max": float(np.max(pred)),
    }


def distance_stats(base: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    d = np.asarray(pred, dtype=float) - np.asarray(base, dtype=float)
    return {
        "mean_delta": float(np.mean(d)),
        "mae_delta": float(np.mean(np.abs(d))),
        "max_abs_delta": float(np.max(np.abs(d))),
        "changed_gt_0p1": int(np.sum(np.abs(d) > 0.1)),
        "changed_gt_0p5": int(np.sum(np.abs(d) > 0.5)),
        "changed_gt_1p0": int(np.sum(np.abs(d) > 1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V1.3 cap-relax/tail candidates without retraining.")
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--artifact_dir", default="artifacts_v1_0")
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--output_dir", default="submissions_v1_3")
    parser.add_argument("--report_dir", default="reports_v1_3")
    parser.add_argument("--header", action="store_true", default=True)
    parser.add_argument("--no_header", dest="header", action="store_false")
    args = parser.parse_args()

    raw = read_csv(args.features_path)
    artifact_dir = Path(args.artifact_dir)
    config = load_config(Path(args.config_path) if args.config_path else None, artifact_dir)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    clipped = predict_with_v1_pipeline(raw, artifact_dir=artifact_dir, config=config)
    unclipped = predict_unclipped_v1_pipeline(raw, artifact_dir=artifact_dir, config=config)

    candidates = make_candidates(raw, clipped, unclipped)

    report: dict[str, Any] = {
        "base_clipped": prediction_stats(clipped),
        "unclipped_global": prediction_stats(unclipped),
        "unclipped_distance_from_base": distance_stats(clipped, unclipped),
        "candidates": {},
    }

    submit_paths: list[str] = []

    # Save the exact current baseline too, useful for local blending checks.
    baseline_path = output_dir / "v1_3_baseline_current.csv"
    write_submission(baseline_path, clipped, header=args.header)
    summary(
        baseline_path.with_suffix(".summary.json"),
        clipped,
        {"candidate": "v1_3_baseline_current", "header": bool(args.header)},
    )

    for name, pred in candidates.items():
        out_path = output_dir / f"{name}.csv"
        write_submission(out_path, pred, header=args.header)
        summary(
            out_path.with_suffix(".summary.json"),
            pred,
            {"candidate": name, "header": bool(args.header)},
        )
        submit_paths.append(str(out_path))
        report["candidates"][name] = {
            "stats": prediction_stats(pred),
            "distance_from_base": distance_stats(clipped, pred),
        }

    # Recommendation order: conservative first.
    recommended = [
        output_dir / "v1_3_caprelax_uncap10.csv",
        output_dir / "v1_3_caprelax_uncap15.csv",
        output_dir / "v1_3_capzone_lift1p0.csv",
        output_dir / "v1_3_tail_lift0p6.csv",
        output_dir / "v1_3_caprelax_uncap20.csv",
        output_dir / "v1_3_lowtrim0p4_tail0p8.csv",
        output_dir / "v1_3_capzone_lift2p0.csv",
        output_dir / "v1_3_tail_lift1p0.csv",
    ]

    (output_dir / "SUBMIT_FIRST.txt").write_text(
        "\n".join(str(p) for p in recommended if p.exists()) + "\n",
        encoding="utf-8",
    )

    (report_dir / "v1_3_cap_relax_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Generated V1.3 candidates:")
    for p in submit_paths:
        print(p)

    print("\nSubmit first:")
    for p in recommended:
        if p.exists():
            print(p)
    print(f"\nReport: {report_dir / 'v1_3_cap_relax_report.json'}")


if __name__ == "__main__":
    main()
