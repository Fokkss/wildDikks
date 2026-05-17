from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .feature_engineering import FARM_CAPACITY_MW, available_capacity_from_raw, find_datetime_col, make_features
from .train_production import write_submission, restore_order, rule_correction, clip_pred, summarize_pred


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def sort_for_features_keep_order(df: pd.DataFrame):
    dt_col = find_datetime_col(df)
    tmp = df.copy()
    tmp["__orig_order_beta03"] = np.arange(len(tmp))
    if dt_col is not None:
        tmp["__dt_sort_beta03"] = pd.to_datetime(tmp[dt_col], errors="coerce")
        tmp = tmp.sort_values("__dt_sort_beta03", kind="mergesort").drop(columns=["__dt_sort_beta03"])
    return tmp.reset_index(drop=True), tmp["__orig_order_beta03"].to_numpy(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_path", default="../data/valid_features.csv")
    ap.add_argument("--model_path", default="artifacts_beta03/beta03_production_ensemble.pkl")
    ap.add_argument("--output_path", default="submissions_beta03/submission_2027_beta03_production.csv")
    ap.add_argument("--ensemble", default="ensemble_legacy_weighted", choices=["ensemble_legacy_weighted", "ensemble_legacy_conservative", "ensemble_mean_all"])
    ap.add_argument("--rules_strength", type=float, default=0.25)
    ap.add_argument("--capacity_mw", type=float, default=FARM_CAPACITY_MW)
    args = ap.parse_args()

    artifact = joblib.load(args.model_path)
    raw = read_csv(args.features_path)
    raw_sorted, orig_order = sort_for_features_keep_order(raw)
    raw_sorted_clean = raw_sorted.drop(columns=["__orig_order_beta03"], errors="ignore")
    fe = make_features(raw_sorted_clean, target_col=None)
    for c in artifact["feature_cols"]:
        if c not in fe.columns:
            fe[c] = np.nan
    X = pd.DataFrame(artifact["imputer"].transform(fe[artifact["feature_cols"]]), columns=artifact["feature_cols"])

    preds = {name: clip_pred(model.predict(X), raw_sorted_clean, args.capacity_mw) for name, model in artifact["models"].items()}
    if args.ensemble == "ensemble_legacy_weighted":
        pred = 0.55*preds["legacy_exact_1892"] + 0.25*preds["legacy_exact_2500"] + 0.15*preds["legacy_depth4_2200"] + 0.05*preds["friend_depth5_1800"]
    elif args.ensemble == "ensemble_legacy_conservative":
        pred = 0.80*preds["legacy_exact_1892"] + 0.20*preds["legacy_depth4_2200"]
    else:
        pred = np.mean(np.vstack(list(preds.values())), axis=0)
    pred = clip_pred(pred + rule_correction(fe, pred, artifact.get("rule_state", {}), args.rules_strength), raw_sorted_clean, args.capacity_mw)
    pred_out = restore_order(pred, orig_order)
    write_submission(args.output_path, pred_out, header=True)
    summarize_pred(Path(args.output_path).with_suffix(".summary.json"), pred_out, {"ensemble": args.ensemble, "rules_strength": args.rules_strength})
    print(f"Saved {len(pred_out)} predictions to {args.output_path}")

if __name__ == "__main__":
    main()
