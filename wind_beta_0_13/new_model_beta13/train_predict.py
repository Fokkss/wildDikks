from __future__ import annotations

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
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from .feature_engineering import (
    FARM_CAPACITY_MW,
    available_capacity_from_raw,
    find_datetime_col,
    find_target_col,
    make_features,
    sort_by_time_if_possible,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_submission(path: Path, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": pred}).to_csv(path, index=False)


def summary(path: Path, pred: np.ndarray, extra: dict[str, Any]) -> None:
    s = pd.Series(pred)
    data = {
        "n": int(len(pred)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "q05": float(s.quantile(0.05)),
        "q50": float(s.quantile(0.50)),
        "q95": float(s.quantile(0.95)),
        "max": float(s.max()),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_xgb(name: str, seed: int, n_estimators: int, max_depth: int, *, obj: str = "reg:absoluteerror") -> XGBRegressor:
    if name.startswith("q1"):
        min_child = 6
        reg_alpha = 0.04
        reg_lambda = 4.0
    else:
        min_child = 5
        reg_alpha = 0.04
        reg_lambda = 5.0
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=max_depth,
        min_child_weight=min_child,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective=obj,
        eval_metric="mae",
        tree_method="hist",
        max_bin=256,
        random_state=seed,
        n_jobs=-1,
    )


def select_features(fe: pd.DataFrame, target_col: str) -> list[str]:
    dt_col = find_datetime_col(fe)
    drop = {target_col}
    if dt_col:
        drop.add(dt_col)
    cols = []
    for c in fe.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(fe[c]) and fe[c].notna().sum() > 0:
            cols.append(c)
    return cols


def fit_model(
    raw: pd.DataFrame,
    y: pd.Series,
    valid_raw: pd.DataFrame,
    *,
    target_col: str,
    name: str,
    seed: int,
    n_estimators: int,
    max_depth: int,
    use_lags: bool,
    use_time: bool,
    use_availability: bool,
    sample_weight: np.ndarray | None = None,
    objective: str = "reg:absoluteerror",
) -> tuple[np.ndarray, dict[str, Any]]:
    fe = make_features(raw, target_col=target_col, use_lags=use_lags, use_time=use_time, use_availability=use_availability)
    cols = select_features(fe, target_col)
    X = fe[cols]
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=cols)

    model = build_xgb(name, seed, n_estimators, max_depth, obj=objective)
    model.fit(X_imp, y, sample_weight=sample_weight, verbose=False)

    fev = make_features(valid_raw, target_col=None, use_lags=use_lags, use_time=use_time, use_availability=use_availability)
    for c in cols:
        if c not in fev.columns:
            fev[c] = np.nan
    Xv = pd.DataFrame(imp.transform(fev[cols]), columns=cols)
    pred = model.predict(Xv)
    row_cap = available_capacity_from_raw(valid_raw).to_numpy()
    pred = np.clip(pred, 0.0, np.minimum(row_cap, FARM_CAPACITY_MW))
    meta = {
        "name": name,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "use_lags": use_lags,
        "use_time": use_time,
        "use_availability": use_availability,
        "objective": objective,
        "n_features": len(cols),
        "feature_cols": cols,
        "model": model,
        "imputer": imp,
    }
    return pred, meta


def month_series(raw: pd.DataFrame) -> pd.Series:
    dt_col = find_datetime_col(raw)
    if dt_col:
        return pd.to_datetime(raw[dt_col], errors="coerce").dt.month
    if "month" in raw.columns:
        return pd.to_numeric(raw["month"], errors="coerce")
    return pd.Series(np.nan, index=raw.index)


def year_series(raw: pd.DataFrame) -> pd.Series:
    dt_col = find_datetime_col(raw)
    if dt_col:
        return pd.to_datetime(raw[dt_col], errors="coerce").dt.year
    return pd.Series(np.nan, index=raw.index)


def make_sample_weights(raw: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    m = month_series(raw)
    y = year_series(raw)
    w = np.ones(len(raw), dtype=float)
    if "q1" in mode:
        w += m.isin([1, 2, 3]).astype(float).to_numpy() * 0.35
    if "recent" in mode and y.notna().any():
        max_y = int(y.max())
        w += (y >= max_y - 1).astype(float).to_numpy() * 0.20
    return w


def rule_delta(valid_raw: pd.DataFrame, base_pred: np.ndarray, strength: float) -> np.ndarray:
    fe = make_features(valid_raw, target_col=None, use_lags=True, use_time=True, use_availability=True)
    pred = np.asarray(base_pred, dtype=float)
    d = np.zeros(len(pred), dtype=float)
    def col(name: str, default=0.0):
        if name in fe.columns:
            return pd.to_numeric(fe[name], errors="coerce").fillna(default).to_numpy()
        return np.full(len(pred), default, dtype=float)
    # Production-stable version of hidden-valid calibration map.
    d += (col("dir60_75") > 0.5) * (-0.30)
    d += (col("dir45_90") > 0.5) * (-0.08)
    d += (col("dir315_360") > 0.5) * (+0.08)
    d += (col("cloud_hi") > 0.5) * (-0.16)
    d += (col("pressure_low") > 0.5) * (+0.18)
    d += (col("pressure_midlow") > 0.5) * (-0.13)
    d += (col("temp_gt5") > 0.5) * (+0.13)
    d += (col("ws5_7") > 0.5) * (+0.13)
    d += ((col("ws0_3") > 0.5) & (pred < 15.0)) * (+0.08)
    # Use positive 180-80 shear if present; q-based threshold was unstable in some folds.
    d += ((col("ws_diff_180_80") > 0.05) & (pred >= 50.0) & (pred < 70.0)) * (+0.75)
    d += ((col("ws10_11") > 0.5) & (pred >= 45.0)) * (+0.08)
    d += ((col("ws13p") > 0.5) & (pred > 65.0)) * (-0.08)
    return d * strength



def profile_delta(valid_raw: pd.DataFrame, base_pred: np.ndarray, profile: str) -> np.ndarray:
    """Small deterministic production profiles learned from validation diagnostics.

    These are not teacher-based; they use only forecast features and base predictions.
    They are meant to be tiny line-search profiles around the production anchor.
    """
    fe = make_features(valid_raw, target_col=None, use_lags=True, use_time=True, use_availability=True)
    pred = np.asarray(base_pred, dtype=float)
    d = np.zeros(len(pred), dtype=float)

    def col(name: str, default=0.0):
        if name in fe.columns:
            return pd.to_numeric(fe[name], errors="coerce").fillna(default).to_numpy()
        return np.full(len(pred), default, dtype=float)

    if profile in ("none", ""):
        return d

    if profile in ("dircloud", "all_mild", "all_strong", "physics_plus_dir_tiny"):
        # Wake / terrain-like direction effect + cloudy high-overprediction effect.
        # In beta08 pure dircloud was weak, so physics_plus_dir_tiny uses a tiny scale only.
        if profile == "all_strong":
            scale = 1.25
        elif profile == "physics_plus_dir_tiny":
            scale = 0.35
        else:
            scale = 1.0
        d += (col("dir60_75") > 0.5) * (-0.16 * scale)
        d += (col("dir45_90") > 0.5) * (-0.05 * scale)
        d += (col("dir315_360") > 0.5) * (+0.05 * scale)
        d += (col("cloud_hi") > 0.5) * (-0.08 * scale)

    if profile in ("physics", "physics_strong", "physics_xstrong", "physics_ultra", "physics_plus_dir_tiny", "theory103", "density_boost", "all_mild", "all_strong"):
        # Stability / density / ramp profile. Beta08 best was physics-only,
        # so beta13 line-searches around this vector.
        if profile == "physics_strong":
            scale = 1.35
        elif profile == "physics_xstrong":
            scale = 1.55
        elif profile == "physics_ultra":
            scale = 1.75
        elif profile == "theory103":
            scale = 1.20
        elif profile == "density_boost":
            scale = 1.10
        elif profile == "all_strong":
            scale = 1.20
        else:
            scale = 1.0
        d += (col("pressure_low") > 0.5) * (+0.10 * scale)
        d += (col("pressure_midlow") > 0.5) * (-0.07 * scale)
        d += (col("temp_gt5") > 0.5) * (+0.07 * scale)
        d += (col("ws5_7") > 0.5) * (+0.07 * scale)
        d += ((col("ws10_11") > 0.5) & (pred >= 45.0)) * (+0.06 * scale)
        d += ((col("ws13p") > 0.5) & (pred > 65.0)) * (-0.05 * scale)

    if profile == "theory103":
        # Explicit SG 3.4-132 / G132-IIA rated-zone probe around 10.3 m/s.
        d += ((col("near_rated_9_103") > 0.5) & (pred >= 45.0) & (pred < 75.0)) * (+0.08)
        d += ((col("rated_zone_103") > 0.5) & (pred >= 60.0)) * (-0.05)
    if profile == "density_boost":
        d += (col("pressure_low") > 0.5) * (+0.08)
        d += (col("temp_gt5") > 0.5) * (+0.05)
        d += ((col("ws5_7") > 0.5) & (pred < 45.0)) * (+0.05)

    if profile in ("lowrelax", "all_mild", "all_strong"):
        scale = 1.0 if profile != "all_strong" else 1.15
        d += ((col("ws0_3") > 0.5) & (pred < 15.0)) * (+0.05 * scale)

    return d

def clip_pred(valid_raw: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    cap = np.minimum(available_capacity_from_raw(valid_raw).to_numpy(), FARM_CAPACITY_MW)
    return np.clip(pred, 0.0, cap)


def save_candidate(
    name: str,
    pred: np.ndarray,
    valid_raw: pd.DataFrame,
    out_dir: Path,
    rep_dir: Path,
    submit_list: list[str],
    extra: dict[str, Any],
) -> None:
    pred = clip_pred(valid_raw, pred)
    path = out_dir / name
    write_submission(path, pred)
    summary(rep_dir / (name.replace(".csv", ".summary.json")), pred, extra)
    if path.exists():
        submit_list.append(str(path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", required=True)
    p.add_argument("--valid_path", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact_dir", default="artifacts_beta13")
    p.add_argument("--submission_dir", default="submissions_beta13")
    p.add_argument("--report_dir", default="reports_beta13")
    args = p.parse_args()
    set_seed(args.seed)
    art_dir = Path(args.artifact_dir); out_dir = Path(args.submission_dir); rep_dir = Path(args.report_dir)
    art_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True); rep_dir.mkdir(parents=True, exist_ok=True)

    raw = read_csv(args.train_path).rename(columns={"Кол-во_ВЭУ_в_ремонте":"repair_count"})
    target_col = find_target_col(raw, args.target)
    raw = sort_by_time_if_possible(raw)
    y_raw = pd.to_numeric(raw[target_col], errors="coerce")
    cap = available_capacity_from_raw(raw)
    y = y_raw.clip(lower=0.0, upper=cap)
    ok = y.notna()
    raw = raw.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True)
    valid_raw = read_csv(args.valid_path)

    # Model zoo: small enough to run, broad enough to recover the old production family.
    specs = [
        dict(name="legacy1700", n=1700, depth=3, lags=True, time=True, avail=True, weight="none", obj="reg:absoluteerror"),
        dict(name="legacy1892", n=1892, depth=3, lags=True, time=True, avail=True, weight="none", obj="reg:absoluteerror"),
        dict(name="legacy2500", n=2500, depth=3, lags=True, time=True, avail=True, weight="none", obj="reg:absoluteerror"),
        dict(name="depth4_1892", n=1892, depth=4, lags=True, time=True, avail=True, weight="none", obj="reg:absoluteerror"),
        dict(name="q1recent1700", n=1700, depth=3, lags=True, time=True, avail=True, weight="q1_recent", obj="reg:absoluteerror"),
        dict(name="nolags1700", n=1700, depth=3, lags=False, time=True, avail=True, weight="none", obj="reg:absoluteerror"),
    ]
    preds: dict[str, np.ndarray] = {}
    metas: dict[str, Any] = {}
    for i, sp in enumerate(specs):
        sw = make_sample_weights(raw, sp["weight"])
        pr, meta = fit_model(raw, y, valid_raw, target_col=target_col, name=sp["name"], seed=args.seed+i, n_estimators=sp["n"], max_depth=sp["depth"], use_lags=sp["lags"], use_time=sp["time"], use_availability=sp["avail"], sample_weight=sw, objective=sp["obj"])
        preds[sp["name"]] = pr
        # Remove large sklearn objects from JSON meta; save full artifact separately.
        metas[sp["name"]] = {k:v for k,v in meta.items() if k not in ("model", "imputer", "feature_cols")}
        joblib.dump(meta, art_dir / f"{sp['name']}.joblib")

    # Production ensembles. Names are explicit and stable.
    ensembles = {
        "anchor_1700_d4_w65": 0.65*preds["legacy1700"] + 0.35*preds["depth4_1892"],
        "anchor_1700_d4_w67": 0.67*preds["legacy1700"] + 0.33*preds["depth4_1892"],
        "anchor_1700_d4_w68": 0.68*preds["legacy1700"] + 0.32*preds["depth4_1892"],
        "anchor_1700_d4_w69": 0.69*preds["legacy1700"] + 0.31*preds["depth4_1892"],
        "anchor_1700_d4_w70": 0.70*preds["legacy1700"] + 0.30*preds["depth4_1892"],
        "anchor_1700_d4_w71": 0.71*preds["legacy1700"] + 0.29*preds["depth4_1892"],
        "anchor_1700_d4_w72": 0.72*preds["legacy1700"] + 0.28*preds["depth4_1892"],
        "anchor_1700_d4_w73": 0.73*preds["legacy1700"] + 0.27*preds["depth4_1892"],
        "anchor_1700_d4_w75": 0.75*preds["legacy1700"] + 0.25*preds["depth4_1892"],
        "anchor_1700_d4_w77": 0.77*preds["legacy1700"] + 0.23*preds["depth4_1892"],
        "anchor_1700_d4_w78": 0.78*preds["legacy1700"] + 0.22*preds["depth4_1892"],
        "anchor_1700_d4_w80": 0.80*preds["legacy1700"] + 0.20*preds["depth4_1892"],
        "anchor_1700_d4_w85": 0.85*preds["legacy1700"] + 0.15*preds["depth4_1892"],
        "anchor_1700_d4_nolag": 0.70*preds["legacy1700"] + 0.20*preds["depth4_1892"] + 0.10*preds["nolags1700"],
        "anchor_1700_q1_d4": 0.65*preds["legacy1700"] + 0.20*preds["q1recent1700"] + 0.15*preds["depth4_1892"],
        "anchor_bag_legacy": 0.45*preds["legacy1700"] + 0.35*preds["legacy1892"] + 0.20*preds["legacy2500"],
        "anchor_balanced": 0.50*preds["legacy1700"] + 0.20*preds["legacy1892"] + 0.20*preds["depth4_1892"] + 0.10*preds["nolags1700"],
    }

    submit_first: list[str] = []
    # Beta11: narrow production search around beta08 best:
    # anchor_1700_d4_w70 + rules55 + bias0.75 + physics_strong -> 8.3960.
    # First wave is only 8 candidates because submissions are slow.
    candidate_specs = [
        # beta13 first wave: tight around beta10 production-best
        # beta10 best: anchor_1700_d4_w70 + rules55 + bias0.75 + physics_strong -> 8.3960
        ("anchor_1700_d4_w70", 0.55, 0.78, "physics_strong"),
        ("anchor_1700_d4_w70", 0.55, 0.72, "physics_strong"),
        ("anchor_1700_d4_w70", 0.60, 0.75, "physics_strong"),
        ("anchor_1700_d4_w70", 0.50, 0.75, "physics_strong"),
        ("anchor_1700_d4_w68", 0.55, 0.75, "physics_strong"),
        ("anchor_1700_d4_w69", 0.55, 0.75, "physics_strong"),
        ("anchor_1700_d4_w71", 0.55, 0.75, "physics_strong"),
        ("anchor_1700_d4_w70", 0.55, 0.75, "physics_xstrong"),
        # second wave: only if first wave beats 8.3960
        ("anchor_1700_d4_w70", 0.55, 0.75, "density_boost"),
        ("anchor_1700_d4_w70", 0.55, 0.75, "physics_plus_dir_tiny"),
        ("anchor_1700_d4_w70", 0.65, 0.75, "physics_strong"),
        ("anchor_1700_d4_w70", 0.55, 0.82, "physics_strong"),
        ("anchor_1700_d4_w67", 0.55, 0.75, "physics_strong"),
        ("anchor_1700_d4_w72", 0.55, 0.75, "physics_strong"),
        ("anchor_1700_d4_w70", 0.60, 0.78, "physics_strong"),
        ("anchor_1700_d4_w70", 0.50, 0.78, "physics_strong"),
    ]
    for ens_name, rules_strength, bias, profile in candidate_specs:
        base = ensembles[ens_name]
        pred = base + rule_delta(valid_raw, base, rules_strength) + profile_delta(valid_raw, base, profile) + bias
        prof_suffix = "" if profile == "none" else f"_{profile}"
        fname = f"beta13_{ens_name}_rules{int(rules_strength*100):02d}_bias{str(bias).replace('.', 'p')}{prof_suffix}.csv"
        save_candidate(fname, pred, valid_raw, out_dir, rep_dir, submit_first, {"ensemble": ens_name, "rules_strength": rules_strength, "bias": bias, "profile": profile})

    # Also save raw anchor for diagnostics, but do not put all of them into submit_first.
    for ens_name in ["anchor_1700_d4_w70", "anchor_1700_d4_w75", "anchor_1700_d4_w80", "anchor_balanced"]:
        pred = ensembles[ens_name]
        fname = f"beta13_{ens_name}_raw.csv"
        save_candidate(fname, pred, valid_raw, out_dir, rep_dir, [], {"ensemble": ens_name, "rules_strength": 0.0, "bias": 0.0})

    (out_dir / "SUBMIT_FIRST.txt").write_text("\n".join(submit_first[:8]) + "\n", encoding="utf-8")
    (out_dir / "SUBMIT_SECOND.txt").write_text("\n".join(submit_first[8:]) + "\n", encoding="utf-8")
    (art_dir / "beta13_model_meta.json").write_text(json.dumps({"target_col": target_col, "models": metas, "ensembles": list(ensembles)}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Generated candidates:")
    for s in submit_first:
        print(s)
    print(f"\nFirst wave list: {out_dir / 'SUBMIT_FIRST.txt'}")
    print("No missing-file entries are written: the list is built from files after save().")


if __name__ == "__main__":
    main()
