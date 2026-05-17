from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .feature_engineering import FARM_CAPACITY_MW, find_datetime_col, make_features
from .train_production import write_submission, restore_order, rule_correction, clip_pred, summarize_pred


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def sort_for_features_keep_order(df: pd.DataFrame):
    dt_col = find_datetime_col(df)
    tmp = df.copy()
    tmp["__orig_order_beta04"] = np.arange(len(tmp))
    if dt_col is not None:
        tmp["__dt_sort_beta04"] = pd.to_datetime(tmp[dt_col], errors="coerce")
        tmp = tmp.sort_values("__dt_sort_beta04", kind="mergesort").drop(columns=["__dt_sort_beta04"])
    return tmp.reset_index(drop=True), tmp["__orig_order_beta04"].to_numpy(int)


def make_ensemble(preds: dict[str, np.ndarray], name: str) -> np.ndarray:
    legacy_seedbag = np.mean(np.vstack([
        preds["legacy_1500"], preds["legacy_1700"], preds["legacy_1892"],
        preds["legacy_2100"], preds["legacy_1892_seedbag_a"], preds["legacy_1892_seedbag_b"],
    ]), axis=0)
    depth4_bag = 0.55 * preds["legacy_depth4_2000"] + 0.45 * preds["legacy_depth4_2400"]

    if name == "bag_legacy_seed":
        return legacy_seedbag
    if name == "bag_depth4":
        return depth4_bag
    if name.startswith("blend_legacy1892_depth4_w"):
        w = float(name.rsplit("w", 1)[1]) / 100.0
        return w * preds["legacy_1892"] + (1.0 - w) * depth4_bag
    if name.startswith("blend_seedbag_depth4_w"):
        w = float(name.rsplit("w", 1)[1]) / 100.0
        return w * legacy_seedbag + (1.0 - w) * depth4_bag
    if name == "ensemble_legacy_conservative":
        return 0.80 * preds["legacy_1892"] + 0.20 * preds["legacy_depth4_2000"]
    if name == "ensemble_legacy_weighted":
        return (
            0.55 * preds["legacy_1892"]
            + 0.20 * preds["legacy_2100"]
            + 0.15 * preds["legacy_depth4_2000"]
            + 0.10 * preds["legacy_1892_seedbag_a"]
        )
    if name == "ensemble_mean_all":
        return np.mean(np.vstack(list(preds.values())), axis=0)
    if name in preds:
        return preds[name]
    raise ValueError(f"Unknown ensemble/model name: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_path", default="../data/valid_features.csv")
    ap.add_argument("--model_path", default="artifacts_beta04/beta04_production_ensemble.pkl")
    ap.add_argument("--output_path", default="submissions_beta04/submission_2027_beta04_production.csv")
    ap.add_argument("--ensemble", default="blend_legacy1892_depth4_w80")
    ap.add_argument("--rules_strength", type=float, default=0.25)
    ap.add_argument("--bias", type=float, default=0.0, help="Optional global MW bias; keep 0 unless validated.")
    ap.add_argument("--capacity_mw", type=float, default=FARM_CAPACITY_MW)
    args = ap.parse_args()

    artifact = joblib.load(args.model_path)
    raw = read_csv(args.features_path)
    raw_sorted, orig_order = sort_for_features_keep_order(raw)
    raw_sorted_clean = raw_sorted.drop(columns=["__orig_order_beta04"], errors="ignore")
    fe = make_features(raw_sorted_clean, target_col=None)
    for c in artifact["feature_cols"]:
        if c not in fe.columns:
            fe[c] = np.nan
    X = pd.DataFrame(artifact["imputer"].transform(fe[artifact["feature_cols"]]), columns=artifact["feature_cols"])

    preds = {name: clip_pred(model.predict(X), raw_sorted_clean, args.capacity_mw) for name, model in artifact["models"].items()}
    pred = make_ensemble(preds, args.ensemble)
    pred = clip_pred(pred + rule_correction(fe, pred, artifact.get("rule_state", {}), args.rules_strength) + args.bias, raw_sorted_clean, args.capacity_mw)
    pred_out = restore_order(pred, orig_order)
    write_submission(args.output_path, pred_out, header=True)
    summarize_pred(Path(args.output_path).with_suffix(".summary.json"), pred_out, {"ensemble": args.ensemble, "rules_strength": args.rules_strength, "bias": args.bias})
    print(f"Saved {len(pred_out)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()
