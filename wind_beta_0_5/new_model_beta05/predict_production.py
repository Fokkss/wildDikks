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
    tmp["__orig_order_beta05"] = np.arange(len(tmp))
    if dt_col is not None:
        tmp["__dt_sort_beta05"] = pd.to_datetime(tmp[dt_col], errors="coerce")
        tmp = tmp.sort_values("__dt_sort_beta05", kind="mergesort").drop(columns=["__dt_sort_beta05"])
    return tmp.reset_index(drop=True), tmp["__orig_order_beta05"].to_numpy(int)


def make_ensemble(preds: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "b03_legacy_weighted":
        return 0.55*preds["legacy_exact_1892"] + 0.25*preds["legacy_exact_2500"] + 0.15*preds["legacy_depth4_2200"] + 0.05*preds["friend_depth5_1800"]
    if name == "b03_legacy_conservative":
        return 0.80*preds["legacy_exact_1892"] + 0.20*preds["legacy_depth4_2200"]
    if name == "b03_mean_all":
        return np.mean(np.vstack([preds[k] for k in ["legacy_exact_1892", "legacy_exact_2500", "legacy_depth4_2200", "friend_depth5_1800"]]), axis=0)

    legacy_es_keys = [k for k in preds if k.startswith("legacy_es")]
    depth4_es_keys = [k for k in preds if k.startswith("depth4_es")]
    if legacy_es_keys:
        legacy_es_main = sorted(legacy_es_keys, key=lambda k: abs(int(k.replace("legacy_es", "")) - 1892))[0]
    else:
        legacy_es_main = "legacy_exact_1892"
    depth4_es_main = depth4_es_keys[0] if depth4_es_keys else "legacy_depth4_2200"

    if name == "es_legacy_main":
        return preds[legacy_es_main]
    if name == "es_mix_legacy_b03":
        return 0.55 * preds[legacy_es_main] + 0.45 * preds["legacy_exact_1892"]
    if name == "es_conservative":
        return 0.80 * preds[legacy_es_main] + 0.20 * preds[depth4_es_main]
    if name == "es_bestmix":
        return 0.45 * preds["legacy_exact_1892"] + 0.55 * preds[legacy_es_main] + 0.20 * preds[depth4_es_main]
    if name in preds:
        return preds[name]
    raise ValueError(f"Unknown ensemble: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_path", default="../data/valid_features.csv")
    ap.add_argument("--model_path", default="artifacts_beta05/beta05_production_ensemble.pkl")
    ap.add_argument("--output_path", default="submissions_beta05/submission_2027_beta05_production.csv")
    ap.add_argument("--ensemble", default="b03_legacy_conservative")
    ap.add_argument("--rules_strength", type=float, default=0.55)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--capacity_mw", type=float, default=FARM_CAPACITY_MW)
    args = ap.parse_args()

    artifact = joblib.load(args.model_path)
    raw = read_csv(args.features_path)
    raw_sorted, orig_order = sort_for_features_keep_order(raw)
    raw_sorted_clean = raw_sorted.drop(columns=["__orig_order_beta05"], errors="ignore")
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
