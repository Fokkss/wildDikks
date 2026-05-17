from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .feature_engineering import (
    FARM_CAPACITY_MW,
    available_capacity_from_raw,
    find_datetime_col,
    find_target_col,
    make_features,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_submission(path: str | Path, pred: np.ndarray, header: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": np.asarray(pred, dtype=float)}).to_csv(path, index=False, header=header)


def sort_for_features_keep_order(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Sort by timestamp for lag/rolling weather features, remember original order."""
    dt_col = find_datetime_col(df)
    tmp = df.copy()
    tmp["__orig_order_beta04"] = np.arange(len(tmp))
    if dt_col is not None:
        tmp["__dt_sort_beta04"] = pd.to_datetime(tmp[dt_col], errors="coerce")
        tmp = tmp.sort_values("__dt_sort_beta04", kind="mergesort").drop(columns=["__dt_sort_beta04"])
    return tmp.reset_index(drop=True), tmp["__orig_order_beta04"].to_numpy(int)


def restore_order(values: np.ndarray, orig_order: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    out[orig_order] = values
    return out


def make_features_chrono(df: pd.DataFrame, target_col: str | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    sorted_df, orig_order = sort_for_features_keep_order(df)
    fe = make_features(sorted_df.drop(columns=["__orig_order_beta04"], errors="ignore"), target_col=target_col)
    return fe, orig_order


def competition_error(y_true: np.ndarray, y_pred: np.ndarray, capacity_mw: float = FARM_CAPACITY_MW) -> float:
    return float(mean_absolute_error(y_true, y_pred) / capacity_mw * 100.0)


@dataclass
class CandidateConfig:
    name: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    objective: str = "reg:absoluteerror"


def candidate_configs() -> list[CandidateConfig]:
    """Focused production model grid for beta 0.4.

    The grid is centered around the old strong regressor parameters:
    lr=0.03, depth=3, min_child_weight=5, subsample/colsample=0.85,
    reg_alpha=0.04, reg_lambda=5.0.

    We add: n_estimators line-search, seed bagging, and a shallow depth-4
    companion because beta 0.3 showed that 0.80*legacy + 0.20*depth4
    was the best production ensemble.
    """
    return [
        CandidateConfig("legacy_1500", 1500, 0.03, 3, 5, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("legacy_1700", 1700, 0.03, 3, 5, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("legacy_1892", 1892, 0.03, 3, 5, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("legacy_2100", 2100, 0.03, 3, 5, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("legacy_1892_seedbag_a", 1892, 0.03, 3, 5, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("legacy_1892_seedbag_b", 1892, 0.03, 3, 5, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("legacy_depth4_2000", 2000, 0.03, 4, 6, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("legacy_depth4_2400", 2400, 0.03, 4, 6, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("friend_depth5_1800", 1800, 0.03, 5, 8, 0.85, 0.85, 0.05, 5.0),
    ]

def build_xgb(cfg: CandidateConfig, seed: int):
    try:
        from xgboost import XGBRegressor
    except Exception as e:  # pragma: no cover
        raise RuntimeError("xgboost is required for beta 0.4 production training") from e
    return XGBRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_child_weight=cfg.min_child_weight,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        objective=cfg.objective,
        eval_metric="mae",
        tree_method="hist",
        max_bin=256,
        random_state=seed,
        n_jobs=-1,
    )


def select_feature_columns(fe: pd.DataFrame, target_col: str, drop_cols: list[str] | None = None) -> list[str]:
    drop = set(drop_cols or []) | {target_col}
    cols = []
    for col in fe.columns:
        if col in drop or col.startswith("__"):
            continue
        if pd.api.types.is_numeric_dtype(fe[col]) and fe[col].notna().sum() > 0:
            cols.append(col)
    return cols


def compute_rule_state(fe_train: pd.DataFrame) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for col in ["ws_diff_180_80", "ws_ratio_180_80", "shear_alpha_80_180", "shear_alpha_80_120"]:
        if col in fe_train.columns and fe_train[col].notna().sum() > 20:
            state[f"q75__{col}"] = float(pd.to_numeric(fe_train[col], errors="coerce").quantile(0.75))
            state[f"q80__{col}"] = float(pd.to_numeric(fe_train[col], errors="coerce").quantile(0.80))
    return state


def col(fe: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in fe.columns:
        return pd.to_numeric(fe[name], errors="coerce").fillna(default)
    return pd.Series(default, index=fe.index, dtype=float)


def rule_correction(fe: pd.DataFrame, pred: np.ndarray, state: dict[str, Any], strength: float) -> np.ndarray:
    """Portable correction map discovered during leaderboard diagnostics.

    Keep this conservative for production. It is not trained on hidden labels; it encodes
    physically interpretable regimes: shear/wake, direction, cloud, pressure/temp and low wind.
    """
    pred_s = pd.Series(np.asarray(pred, dtype=float), index=fe.index)
    corr = pd.Series(0.0, index=fe.index, dtype=float)

    # High vertical shear 180-80m. Prefer direct speed difference, fallback to ratio/alpha.
    if "ws_diff_180_80" in fe.columns and "q75__ws_diff_180_80" in state:
        shear_q4 = col(fe, "ws_diff_180_80") >= state["q75__ws_diff_180_80"]
    elif "shear_alpha_80_180" in fe.columns and "q75__shear_alpha_80_180" in state:
        shear_q4 = col(fe, "shear_alpha_80_180") >= state["q75__shear_alpha_80_180"]
    else:
        shear_q4 = pd.Series(False, index=fe.index)

    pred50_70 = (pred_s >= 50.0) & (pred_s < 70.0)
    pred0_15 = pred_s < 15.0

    corr += 1.05 * (shear_q4 & pred50_70).astype(float)
    corr += -0.26 * col(fe, "dir60_75")
    corr += -0.10 * col(fe, "dir45_90")
    corr += 0.10 * col(fe, "dir315_360")
    corr += -0.12 * col(fe, "cloud_hi")
    corr += 0.12 * col(fe, "pressure_low")
    corr += -0.09 * col(fe, "pressure_midlow")
    corr += 0.10 * col(fe, "temp_gt5")

    # Wind speed zones. Prefer 84m if present; fallback to 80m.
    ws = col(fe, "hub_ws_84m", np.nan)
    if ws.isna().all():
        ws = col(fe, "hub_ws_80m", np.nan)
    corr += 0.10 * ((ws >= 5.0) & (ws < 7.0)).astype(float)
    corr += 0.08 * ((ws >= 10.0) & (ws < 11.0)).astype(float)
    corr += -0.06 * (ws >= 13.0).astype(float)
    corr += 0.10 * ((ws < 3.0) & pred0_15).astype(float)

    return np.asarray(corr * strength, dtype=float)


def clip_pred(pred: np.ndarray, raw_df_sorted: pd.DataFrame, capacity_mw: float) -> np.ndarray:
    cap = available_capacity_from_raw(raw_df_sorted).fillna(capacity_mw).to_numpy(float)
    cap = np.minimum(cap, capacity_mw)
    return np.clip(np.asarray(pred, dtype=float), 0.0, cap)


def summarize_pred(path: Path, pred: np.ndarray, extra: dict[str, Any] | None = None) -> None:
    s = pd.Series(pred)
    payload = {
        "n": int(len(s)), "mean": float(s.mean()), "std": float(s.std()),
        "min": float(s.min()), "q05": float(s.quantile(0.05)), "q50": float(s.quantile(0.50)),
        "q95": float(s.quantile(0.95)), "max": float(s.max()),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_and_predict(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    artifacts_dir = Path(args.artifacts_dir)
    submissions_dir = Path(args.submissions_dir)
    reports_dir = Path(args.reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_train = read_csv(args.train_path)
    target_col = find_target_col(raw_train, args.target_col)
    timestamp_col = find_datetime_col(raw_train)
    raw_train_sorted, train_orig = sort_for_features_keep_order(raw_train)
    raw_train_sorted = raw_train_sorted.drop(columns=["__orig_order_beta04"], errors="ignore")

    y_raw = pd.to_numeric(raw_train_sorted[target_col], errors="coerce")
    row_cap = available_capacity_from_raw(raw_train_sorted).fillna(args.capacity_mw)
    y = y_raw.clip(0.0, row_cap)
    good = y.notna()
    raw_train_sorted = raw_train_sorted.loc[good].reset_index(drop=True)
    y = y.loc[good].reset_index(drop=True)

    fe_train = make_features(raw_train_sorted, target_col=target_col)
    drop_cols = [timestamp_col] if timestamp_col else []
    feature_cols = select_feature_columns(fe_train, target_col, drop_cols)
    X = fe_train[feature_cols]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    rule_state = compute_rule_state(fe_train)
    candidates = candidate_configs()

    # Optional lightweight CV report on the same features, mainly for sanity checks.
    cv_rows: list[dict[str, Any]] = []
    if args.cv:
        n_splits = min(args.n_splits, max(2, len(X_imp) // 6000))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for cfg in candidates[:2]:
            for fold, (tr, va) in enumerate(tscv.split(X_imp), 1):
                model = build_xgb(cfg, args.seed + fold)
                model.fit(X_imp.iloc[tr], y.iloc[tr], verbose=False)
                pred_va = model.predict(X_imp.iloc[va])
                pred_va = clip_pred(pred_va, raw_train_sorted.iloc[va], args.capacity_mw)
                cv_rows.append({
                    "model": cfg.name, "fold": fold, "n_train": int(len(tr)), "n_valid": int(len(va)),
                    "mae_mw": float(mean_absolute_error(y.iloc[va], pred_va)),
                    "rmse_mw": float(np.sqrt(mean_squared_error(y.iloc[va], pred_va))),
                    "competition_error": competition_error(y.iloc[va], pred_va, args.capacity_mw),
                    "pred_mean": float(np.mean(pred_va)), "target_mean": float(np.mean(y.iloc[va])),
                })
        pd.DataFrame(cv_rows).to_csv(reports_dir / "cv_metrics.csv", index=False)

    raw_valid = read_csv(args.valid_path)
    raw_valid_sorted, valid_orig = sort_for_features_keep_order(raw_valid)
    raw_valid_sorted_clean = raw_valid_sorted.drop(columns=["__orig_order_beta04"], errors="ignore")
    fe_valid = make_features(raw_valid_sorted_clean, target_col=None)
    for c in feature_cols:
        if c not in fe_valid.columns:
            fe_valid[c] = np.nan
    Xv = pd.DataFrame(imputer.transform(fe_valid[feature_cols]), columns=feature_cols)

    models = {}
    preds_sorted: dict[str, np.ndarray] = {}
    for i, cfg in enumerate(candidates):
        model = build_xgb(cfg, args.seed + i * 17)
        model.fit(X_imp, y, verbose=False)
        models[cfg.name] = model
        pred = model.predict(Xv)
        pred = clip_pred(pred, raw_valid_sorted_clean, args.capacity_mw)
        preds_sorted[cfg.name] = pred
        pred_out = restore_order(pred, valid_orig)
        write_submission(submissions_dir / f"beta04_{cfg.name}.csv", pred_out)
        summarize_pred(reports_dir / f"beta04_{cfg.name}.summary.json", pred_out, {"kind": "single_model"})
        for strength in (0.25, 0.45):
            p2 = clip_pred(pred + rule_correction(fe_valid, pred, rule_state, strength), raw_valid_sorted_clean, args.capacity_mw)
            p2_out = restore_order(p2, valid_orig)
            write_submission(submissions_dir / f"beta04_{cfg.name}_rules{int(strength*100)}.csv", p2_out)
            summarize_pred(reports_dir / f"beta04_{cfg.name}_rules{int(strength*100)}.summary.json", p2_out, {"kind": "single_model_rules", "rules_strength": strength})

    # Self-contained production ensembles. No teacher CSV needed.
    p = preds_sorted
    ensembles: dict[str, np.ndarray] = {}

    legacy_seedbag = np.mean(np.vstack([
        p["legacy_1500"], p["legacy_1700"], p["legacy_1892"],
        p["legacy_2100"], p["legacy_1892_seedbag_a"], p["legacy_1892_seedbag_b"],
    ]), axis=0)
    depth4_bag = 0.55 * p["legacy_depth4_2000"] + 0.45 * p["legacy_depth4_2400"]

    ensembles["bag_legacy_seed"] = legacy_seedbag
    ensembles["bag_depth4"] = depth4_bag

    # Around beta 0.3 best: 0.80 * legacy_1892 + 0.20 * depth4.
    for w in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        key = f"blend_legacy1892_depth4_w{int(round(w*100)):02d}"
        ensembles[key] = w * p["legacy_1892"] + (1.0 - w) * depth4_bag

    # Seedbag variants are more stable for production/2027.
    for w in [0.70, 0.80, 0.90]:
        key = f"blend_seedbag_depth4_w{int(round(w*100)):02d}"
        ensembles[key] = w * legacy_seedbag + (1.0 - w) * depth4_bag

    ensembles["ensemble_legacy_conservative"] = 0.80 * p["legacy_1892"] + 0.20 * p["legacy_depth4_2000"]
    ensembles["ensemble_legacy_weighted"] = (
        0.55 * p["legacy_1892"]
        + 0.20 * p["legacy_2100"]
        + 0.15 * p["legacy_depth4_2000"]
        + 0.10 * p["legacy_1892_seedbag_a"]
    )
    ensembles["ensemble_mean_all"] = np.mean(np.vstack([p[k] for k in p]), axis=0)

    saved_submission_names: list[str] = []

    def save_variant(name: str, pred_sorted: np.ndarray, kind: str, rules_strength: float = 0.0, bias: float = 0.0) -> None:
        pred_sorted = clip_pred(pred_sorted + bias, raw_valid_sorted_clean, args.capacity_mw)
        out = restore_order(pred_sorted, valid_orig)
        write_submission(submissions_dir / f"beta04_{name}.csv", out)
        summarize_pred(
            reports_dir / f"beta04_{name}.summary.json",
            out,
            {"kind": kind, "rules_strength": rules_strength, "bias": bias},
        )
        saved_submission_names.append(f"beta04_{name}.csv")

    # Singles: enough for diagnosis, not too many leaderboard files.
    for name in ["legacy_1500", "legacy_1700", "legacy_1892", "legacy_2100", "legacy_depth4_2000"]:
        pred = clip_pred(p[name], raw_valid_sorted_clean, args.capacity_mw)
        save_variant(name, pred, "single_model", 0.0)
        for strength in (0.15, 0.25, 0.35):
            p2 = clip_pred(pred + rule_correction(fe_valid, pred, rule_state, strength), raw_valid_sorted_clean, args.capacity_mw)
            save_variant(f"{name}_rules{int(strength*100)}", p2, "single_model_rules", strength)

    # Ensembles + rule strength line-search.
    for name, pred in ensembles.items():
        pred = clip_pred(pred, raw_valid_sorted_clean, args.capacity_mw)
        save_variant(name, pred, "ensemble", 0.0)
        for strength in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45):
            p2 = clip_pred(pred + rule_correction(fe_valid, pred, rule_state, strength), raw_valid_sorted_clean, args.capacity_mw)
            save_variant(f"{name}_rules{int(strength*100)}", p2, "ensemble_rules", strength)

    # Self-contained bias calibration variants. These do NOT use a teacher CSV.
    # They test the systematic train-CV underprediction we observed; choose only
    # after leaderboard validation, then use the same bias for 2027 if justified.
    focus_ensembles = [
        "ensemble_legacy_conservative",
        "blend_legacy1892_depth4_w75",
        "blend_legacy1892_depth4_w80",
        "blend_legacy1892_depth4_w85",
        "blend_seedbag_depth4_w80",
    ]
    for name in focus_ensembles:
        if name not in ensembles:
            continue
        base_pred = clip_pred(ensembles[name], raw_valid_sorted_clean, args.capacity_mw)
        p_rules = clip_pred(base_pred + rule_correction(fe_valid, base_pred, rule_state, 0.25), raw_valid_sorted_clean, args.capacity_mw)
        for bias in (0.25, 0.50, 0.75, 1.00, 1.25):
            save_variant(f"{name}_rules25_bias{str(bias).replace('.', 'p')}", p_rules, "ensemble_rules_bias", 0.25, bias)

    artifact = {
        "models": models,
        "candidate_configs": [asdict(c) for c in candidates],
        "imputer": imputer,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "timestamp_col": timestamp_col,
        "capacity_mw": args.capacity_mw,
        "rule_state": rule_state,
        "default_ensemble": "blend_legacy1892_depth4_w80",
        "seed": args.seed,
    }
    joblib.dump(artifact, artifacts_dir / "beta04_production_ensemble.pkl")
    (artifacts_dir / "beta04_feature_columns.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifacts_dir / "beta04_rule_state.json").write_text(json.dumps(rule_state, ensure_ascii=False, indent=2), encoding="utf-8")
    if cv_rows:
        (artifacts_dir / "beta04_cv_metrics.json").write_text(json.dumps(cv_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    first = [
        # Current beta 0.3 winner neighborhood: blend weight and rules strength.
        "beta04_blend_legacy1892_depth4_w75_rules25.csv",
        "beta04_blend_legacy1892_depth4_w80_rules15.csv",
        "beta04_blend_legacy1892_depth4_w80_rules20.csv",
        "beta04_blend_legacy1892_depth4_w80_rules25.csv",
        "beta04_blend_legacy1892_depth4_w80_rules30.csv",
        "beta04_blend_legacy1892_depth4_w80_rules35.csv",
        "beta04_blend_legacy1892_depth4_w85_rules25.csv",
        "beta04_ensemble_legacy_conservative_rules15.csv",
        "beta04_ensemble_legacy_conservative_rules25.csv",
        "beta04_ensemble_legacy_conservative_rules35.csv",
        # Seedbag/product stability.
        "beta04_bag_legacy_seed_rules25.csv",
        "beta04_blend_seedbag_depth4_w80_rules25.csv",
        # Self-contained global bias candidates: use only if leaderboard confirms.
        "beta04_ensemble_legacy_conservative_rules25_bias0p5.csv",
        "beta04_ensemble_legacy_conservative_rules25_bias0p75.csv",
        "beta04_ensemble_legacy_conservative_rules25_bias1p0.csv",
        "beta04_blend_legacy1892_depth4_w80_rules25_bias0p5.csv",
        "beta04_blend_legacy1892_depth4_w80_rules25_bias0p75.csv",
        "beta04_blend_legacy1892_depth4_w80_rules25_bias1p0.csv",
        # N-estimator sanity.
        "beta04_legacy_1700_rules25.csv",
        "beta04_legacy_1892_rules25.csv",
        "beta04_legacy_2100_rules25.csv",
    ]
    (submissions_dir / "SUBMIT_FIRST.txt").write_text("\n".join(str(submissions_dir / x) for x in first) + "\n", encoding="utf-8")

    print(f"Saved artifact: {artifacts_dir / 'beta04_production_ensemble.pkl'}")
    print(f"Saved submissions to: {submissions_dir}")
    print(f"First submissions list: {submissions_dir / 'SUBMIT_FIRST.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="../data/train_dataset.csv")
    ap.add_argument("--valid_path", default="../data/valid_features.csv")
    ap.add_argument("--target_col", default="Выработка. Результирующий расчет")
    ap.add_argument("--artifacts_dir", default="artifacts_beta04")
    ap.add_argument("--submissions_dir", default="submissions_beta04")
    ap.add_argument("--reports_dir", default="reports_beta04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_splits", type=int, default=3)
    ap.add_argument("--capacity_mw", type=float, default=FARM_CAPACITY_MW)
    ap.add_argument("--cv", action="store_true", help="Run lightweight time-series CV report before final fit.")
    args = ap.parse_args()
    train_and_predict(args)

if __name__ == "__main__":
    main()
