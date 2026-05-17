from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import FARM_CAPACITY_MW, available_capacity_from_raw, make_features
from .train_predict import rule_delta, profile_delta, clip_pred, write_submission, summary, read_csv


def load_pred_from_artifact(artifact_path: Path, raw: pd.DataFrame) -> np.ndarray:
    meta = joblib.load(artifact_path)
    fe = make_features(raw, target_col=None, use_lags=meta["use_lags"], use_time=meta["use_time"], use_availability=meta["use_availability"])
    cols = meta["feature_cols"]
    for c in cols:
        if c not in fe.columns:
            fe[c] = np.nan
    X = pd.DataFrame(meta["imputer"].transform(fe[cols]), columns=cols)
    pred = meta["model"].predict(X)
    return clip_pred(raw, pred)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True)
    p.add_argument("--artifact_dir", default="artifacts_beta12")
    p.add_argument("--output_path", default="submissions_beta12/submission_2027_beta12_production.csv")
    p.add_argument("--profile", default="anchor_1700_d4_w70")
    p.add_argument("--rules_strength", type=float, default=0.55)
    p.add_argument("--bias", type=float, default=0.75)
    p.add_argument("--extra_profile", default="physics_strong", choices=["none", "dircloud", "physics", "physics_strong", "physics_xstrong", "physics_ultra", "physics_plus_dir_tiny", "theory103", "density_boost", "all_mild", "all_strong", "lowrelax"])
    args = p.parse_args()
    raw = read_csv(args.features_path)
    art = Path(args.artifact_dir)
    p1700 = load_pred_from_artifact(art / "legacy1700.joblib", raw)
    p1892 = load_pred_from_artifact(art / "legacy1892.joblib", raw)
    p2500 = load_pred_from_artifact(art / "legacy2500.joblib", raw)
    pd4 = load_pred_from_artifact(art / "depth4_1892.joblib", raw)
    pq1 = load_pred_from_artifact(art / "q1recent1700.joblib", raw)
    pnl = load_pred_from_artifact(art / "nolags1700.joblib", raw)
    profiles = {
        "anchor_1700_d4_w65": 0.65*p1700 + 0.35*pd4,
        "anchor_1700_d4_w67": 0.67*p1700 + 0.33*pd4,
        "anchor_1700_d4_w68": 0.68*p1700 + 0.32*pd4,
        "anchor_1700_d4_w69": 0.69*p1700 + 0.31*pd4,
        "anchor_1700_d4_w70": 0.70*p1700 + 0.30*pd4,
        "anchor_1700_d4_w71": 0.71*p1700 + 0.29*pd4,
        "anchor_1700_d4_w72": 0.72*p1700 + 0.28*pd4,
        "anchor_1700_d4_w73": 0.73*p1700 + 0.27*pd4,
        "anchor_1700_d4_w75": 0.75*p1700 + 0.25*pd4,
        "anchor_1700_d4_w77": 0.77*p1700 + 0.23*pd4,
        "anchor_1700_d4_w78": 0.78*p1700 + 0.22*pd4,
        "anchor_1700_d4_w80": 0.80*p1700 + 0.20*pd4,
        "anchor_1700_d4_w85": 0.85*p1700 + 0.15*pd4,
        "anchor_1700_d4_nolag": 0.70*p1700 + 0.20*pd4 + 0.10*pnl,
        "anchor_1700_q1_d4": 0.65*p1700 + 0.20*pq1 + 0.15*pd4,
        "anchor_bag_legacy": 0.45*p1700 + 0.35*p1892 + 0.20*p2500,
        "anchor_balanced": 0.50*p1700 + 0.20*p1892 + 0.20*pd4 + 0.10*pnl,
    }
    if args.profile not in profiles:
        raise ValueError(f"Unknown profile {args.profile}. Options: {sorted(profiles)}")
    base = profiles[args.profile]
    pred = clip_pred(raw, base + rule_delta(raw, base, args.rules_strength) + profile_delta(raw, base, args.extra_profile) + args.bias)
    out = Path(args.output_path)
    write_submission(out, pred)
    summary(out.with_suffix(".summary.json"), pred, {"profile": args.profile, "rules_strength": args.rules_strength, "bias": args.bias, "extra_profile": args.extra_profile})
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
