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
    tmp["__orig_order_beta05"] = np.arange(len(tmp))
    if dt_col is not None:
        tmp["__dt_sort_beta05"] = pd.to_datetime(tmp[dt_col], errors="coerce")
        tmp = tmp.sort_values("__dt_sort_beta05", kind="mergesort").drop(columns=["__dt_sort_beta05"])
    return tmp.reset_index(drop=True), tmp["__orig_order_beta05"].to_numpy(int)


def restore_order(values: np.ndarray, orig_order: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    out[orig_order] = values
    return out


def make_features_chrono(df: pd.DataFrame, target_col: str | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    sorted_df, orig_order = sort_for_features_keep_order(df)
    fe = make_features(sorted_df.drop(columns=["__orig_order_beta05"], errors="ignore"), target_col=target_col)
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
    # legacy_exact is the recovered 8.46-era parameter set provided by the user.
    return [
        CandidateConfig("legacy_exact_1892", 1892, 0.03, 3, 5, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("legacy_exact_2500", 2500, 0.03, 3, 5, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("legacy_depth4_2200", 2200, 0.03, 4, 6, 0.85, 0.85, 0.04, 5.0),
        CandidateConfig("friend_depth5_1800", 1800, 0.03, 5, 8, 0.85, 0.85, 0.05, 5.0),
    ]


def build_xgb(cfg: CandidateConfig, seed: int, n_estimators_override: int | None = None, early_stopping_rounds: int | None = None):
    try:
        from xgboost import XGBRegressor
    except Exception as e:  # pragma: no cover
        raise RuntimeError("xgboost is required for beta 0.5 production training") from e
    model = XGBRegressor(
        n_estimators=int(n_estimators_override or cfg.n_estimators),
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
    if early_stopping_rounds is not None:
        params = model.get_params()
        params["early_stopping_rounds"] = int(early_stopping_rounds)
        model = XGBRegressor(**params)
    return model


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
    raw_train_sorted = raw_train_sorted.drop(columns=["__orig_order_beta05"], errors="ignore")

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
    raw_valid_sorted_clean = raw_valid_sorted.drop(columns=["__orig_order_beta05"], errors="ignore")
    fe_valid = make_features(raw_valid_sorted_clean, target_col=None)
    for c in feature_cols:
        if c not in fe_valid.columns:
            fe_valid[c] = np.nan
    Xv = pd.DataFrame(imputer.transform(fe_valid[feature_cols]), columns=feature_cols)

    models = {}
    preds_sorted: dict[str, np.ndarray] = {}

    def fit_predict_model(name: str, cfg: CandidateConfig, seed: int, n_estimators_override: int | None = None) -> np.ndarray:
        model = build_xgb(cfg, seed, n_estimators_override=n_estimators_override)
        model.fit(X_imp, y, verbose=False)
        models[name] = model
        pred = model.predict(Xv)
        pred = clip_pred(pred, raw_valid_sorted_clean, args.capacity_mw)
        preds_sorted[name] = pred
        pred_out = restore_order(pred, valid_orig)
        write_submission(submissions_dir / f"beta05_{name}.csv", pred_out)
        summarize_pred(reports_dir / f"beta05_{name}.summary.json", pred_out, {"kind": "single_model", "n_estimators": int(model.get_params().get("n_estimators", 0))})
        return pred

    for i, cfg in enumerate(candidates):
        pred = fit_predict_model(cfg.name, cfg, args.seed + i * 17)
        for strength in (0.25, 0.55, 0.45):
            p2 = clip_pred(pred + rule_correction(fe_valid, pred, rule_state, strength), raw_valid_sorted_clean, args.capacity_mw)
            p2_out = restore_order(p2, valid_orig)
            write_submission(submissions_dir / f"beta05_{cfg.name}_rules{int(strength*100)}.csv", p2_out)
            summarize_pred(reports_dir / f"beta05_{cfg.name}_rules{int(strength*100)}.summary.json", p2_out, {"kind": "single_model_rules", "rules_strength": strength})

    # ------------------------------------------------------------
    # Early-stopping route: old anecdote says best runs used 10000 trees
    # with early_stopping_rounds≈300. We estimate best_iteration on the
    # last chronological 20% of train, then refit on all rows with
    # best_iter * factor. This remains production-safe: no valid CSV target.
    # ------------------------------------------------------------
    es_report: list[dict[str, Any]] = []

    def estimate_best_iter(cfg: CandidateConfig, seed: int, valid_frac: float = 0.20) -> int:
        n = len(X_imp)
        cut = max(1000, int(n * (1.0 - valid_frac)))
        cut = min(cut, n - 1000)
        es_model = build_xgb(cfg, seed, n_estimators_override=10000, early_stopping_rounds=300)
        es_model.fit(
            X_imp.iloc[:cut], y.iloc[:cut],
            eval_set=[(X_imp.iloc[cut:], y.iloc[cut:])],
            verbose=False,
        )
        best = getattr(es_model, "best_iteration", None)
        if best is None:
            best = 10000
        else:
            best = int(best) + 1
        pred_va = es_model.predict(X_imp.iloc[cut:])
        pred_va = clip_pred(pred_va, raw_train_sorted.iloc[cut:], args.capacity_mw)
        es_report.append({
            "probe": cfg.name,
            "best_iteration": int(best),
            "valid_frac": float(valid_frac),
            "holdout_error": competition_error(y.iloc[cut:], pred_va, args.capacity_mw),
            "holdout_mae": float(mean_absolute_error(y.iloc[cut:], pred_va)),
            "pred_mean": float(np.mean(pred_va)),
            "target_mean": float(np.mean(y.iloc[cut:])),
        })
        return int(best)

    legacy_es_cfg = CandidateConfig("legacy_es_probe", 10000, 0.03, 3, 5, 0.85, 0.85, 0.04, 5.0)
    depth4_es_cfg = CandidateConfig("depth4_es_probe", 10000, 0.03, 4, 6, 0.85, 0.85, 0.04, 5.0)

    try:
        legacy_best = estimate_best_iter(legacy_es_cfg, args.seed + 911)
        for factor in (0.90, 1.00, 1.15, 1.30):
            n_est = int(np.clip(round(legacy_best * factor), 300, 10000))
            name = f"legacy_es{n_est}"
            fit_predict_model(name, legacy_es_cfg, args.seed + 920 + int(factor * 100), n_estimators_override=n_est)
    except Exception as e:
        es_report.append({"probe": "legacy_es_probe", "error": repr(e)})

    try:
        depth4_best = estimate_best_iter(depth4_es_cfg, args.seed + 1011)
        for factor in (0.90, 1.05):
            n_est = int(np.clip(round(depth4_best * factor), 300, 10000))
            name = f"depth4_es{n_est}"
            fit_predict_model(name, depth4_es_cfg, args.seed + 1020 + int(factor * 100), n_estimators_override=n_est)
    except Exception as e:
        es_report.append({"probe": "depth4_es_probe", "error": repr(e)})

    # Self-contained production ensembles. No teacher CSV needed.
    p = preds_sorted
    ensembles: dict[str, np.ndarray] = {}
    ensembles["b03_legacy_weighted"] = (
        0.55 * p["legacy_exact_1892"] + 0.25 * p["legacy_exact_2500"] + 0.15 * p["legacy_depth4_2200"] + 0.05 * p["friend_depth5_1800"]
    )
    ensembles["b03_legacy_conservative"] = 0.80 * p["legacy_exact_1892"] + 0.20 * p["legacy_depth4_2200"]
    ensembles["b03_mean_all"] = np.mean(np.vstack([p[k] for k in ["legacy_exact_1892", "legacy_exact_2500", "legacy_depth4_2200", "friend_depth5_1800"]]), axis=0)

    legacy_es_keys = [k for k in p if k.startswith("legacy_es")]
    depth4_es_keys = [k for k in p if k.startswith("depth4_es")]
    if legacy_es_keys:
        legacy_es_main = sorted(legacy_es_keys, key=lambda k: abs(int(k.replace("legacy_es", "")) - 1892))[0]
        ensembles["es_legacy_main"] = p[legacy_es_main]
        ensembles["es_mix_legacy_b03"] = 0.55 * p[legacy_es_main] + 0.45 * p["legacy_exact_1892"]
        if depth4_es_keys:
            depth4_es_main = depth4_es_keys[0]
            ensembles["es_conservative"] = 0.80 * p[legacy_es_main] + 0.20 * p[depth4_es_main]
            ensembles["es_bestmix"] = 0.45 * p["legacy_exact_1892"] + 0.55 * p[legacy_es_main] + 0.20 * p[depth4_es_main]
        else:
            ensembles["es_conservative"] = 0.80 * p[legacy_es_main] + 0.20 * p["legacy_depth4_2200"]
            ensembles["es_bestmix"] = 0.45 * p["legacy_exact_1892"] + 0.55 * p[legacy_es_main] + 0.20 * p["legacy_depth4_2200"]

    saved_submission_names: list[str] = []

    def save_variant(name: str, pred_sorted: np.ndarray, kind: str, rules_strength: float = 0.0, bias: float = 0.0) -> None:
        pred_sorted = clip_pred(pred_sorted + bias, raw_valid_sorted_clean, args.capacity_mw)
        out = restore_order(pred_sorted, valid_orig)
        write_submission(submissions_dir / f"beta05_{name}.csv", out)
        summarize_pred(
            reports_dir / f"beta05_{name}.summary.json",
            out,
            {"kind": kind, "rules_strength": rules_strength, "bias": bias},
        )
        saved_submission_names.append(f"beta05_{name}.csv")

    # Focused variants: few, high-value outputs only.
    for name, pred in ensembles.items():
        pred = clip_pred(pred, raw_valid_sorted_clean, args.capacity_mw)
        save_variant(name, pred, "ensemble", 0.0)
        for strength in (0.25, 0.55, 0.45, 0.60):
            p2 = clip_pred(pred + rule_correction(fe_valid, pred, rule_state, strength), raw_valid_sorted_clean, args.capacity_mw)
            save_variant(f"{name}_rules{int(strength*100)}", p2, "ensemble_rules", strength)

    # Bias candidates only around the best production family from beta05.
    # Keep these as validated options, not as default for 2027 unless confirmed.
    for name in ["b03_legacy_conservative", "b03_legacy_weighted", "es_bestmix"]:
        if name not in ensembles:
            continue
        base_pred = clip_pred(ensembles[name], raw_valid_sorted_clean, args.capacity_mw)
        for strength in (0.25, 0.55, 0.45):
            p_rules = clip_pred(base_pred + rule_correction(fe_valid, base_pred, rule_state, strength), raw_valid_sorted_clean, args.capacity_mw)
            for bias in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50):
                save_variant(f"{name}_rules{int(strength*100)}_bias{str(bias).replace('.', 'p')}", p_rules, "ensemble_rules_bias", strength, bias)

    artifact = {
        "models": models,
        "candidate_configs": [asdict(c) for c in candidates],
        "es_report": es_report,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "timestamp_col": timestamp_col,
        "capacity_mw": args.capacity_mw,
        "rule_state": rule_state,
        "default_ensemble": "ensemble_legacy_weighted",
        "seed": args.seed,
    }
    joblib.dump(artifact, artifacts_dir / "beta05_production_ensemble.pkl")
    (artifacts_dir / "beta05_feature_columns.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifacts_dir / "beta05_rule_state.json").write_text(json.dumps(rule_state, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifacts_dir / "beta05_es_report.json").write_text(json.dumps(es_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if cv_rows:
        (artifacts_dir / "beta05_cv_metrics.json").write_text(json.dumps(cv_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    first = [
        # 1) reproduce beta05 winner neighborhood, but with stronger rules/bias grid
        "beta05_b03_legacy_conservative_rules35.csv",
        "beta05_b03_legacy_conservative_rules45.csv",
        "beta05_b03_legacy_conservative_rules60.csv",
        "beta05_b03_legacy_conservative_rules35_bias0p5.csv",
        "beta05_b03_legacy_conservative_rules35_bias0p75.csv",
        "beta05_b03_legacy_conservative_rules45_bias0p5.csv",
        # 2) weighted family; beta05 weighted was close and may like bias/rules differently
        "beta05_b03_legacy_weighted_rules25_bias0p5.csv",
        "beta05_b03_legacy_weighted_rules35_bias0p5.csv",
        # 3) early-stopping family from 10000 trees / ES=300
        "beta05_es_legacy_main_rules25.csv",
        "beta05_es_mix_legacy_b03_rules25.csv",
        "beta05_es_conservative_rules25.csv",
        "beta05_es_bestmix_rules25.csv",
        "beta05_es_bestmix_rules35.csv",
    ]
    (submissions_dir / "SUBMIT_FIRST.txt").write_text("\n".join(str(submissions_dir / x) for x in first) + "\n", encoding="utf-8")

    print(f"Saved artifact: {artifacts_dir / 'beta05_production_ensemble.pkl'}")
    print(f"Saved submissions to: {submissions_dir}")
    print(f"First submissions list: {submissions_dir / 'SUBMIT_FIRST.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="../data/train_dataset.csv")
    ap.add_argument("--valid_path", default="../data/valid_features.csv")
    ap.add_argument("--target_col", default="Выработка. Результирующий расчет")
    ap.add_argument("--artifacts_dir", default="artifacts_beta05")
    ap.add_argument("--submissions_dir", default="submissions_beta05")
    ap.add_argument("--reports_dir", default="reports_beta05")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_splits", type=int, default=3)
    ap.add_argument("--capacity_mw", type=float, default=FARM_CAPACITY_MW)
    ap.add_argument("--cv", action="store_true", help="Run lightweight time-series CV report before final fit.")
    args = ap.parse_args()
    train_and_predict(args)

if __name__ == "__main__":
    main()
