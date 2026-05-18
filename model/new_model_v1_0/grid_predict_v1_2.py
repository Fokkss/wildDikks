from __future__ import annotations

"""
Generate a small no-retrain candidate grid around the current best V1.1 setup.

Run from repository root:
    cd model
    python -m new_model_v1_0.grid_predict_v1_2 \
      --features_path ../data/valid_features.csv \
      --artifact_dir artifacts_v1_0 \
      --output_dir submissions_v1_2 \
      --report_dir reports_v1_2 \
      --header

This script does NOT retrain models. It reuses existing artifacts:
    artifacts_v1_0/legacy1700.joblib
    artifacts_v1_0/depth4_1892.joblib
    artifacts_v1_0/catboost_companion.joblib
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from .predict import predict_with_v1_pipeline, write_submission
from .train_predict import read_csv, summary


BASE_CONFIG: dict[str, Any] = {
    "version": "V1.2-grid",
    "description": "Small no-retrain grid around V1.1 best leaderboard config.",
    "xgb_anchor": {
        "legacy1700_weight": 0.70,
        "depth4_1892_weight": 0.30,
    },
    "catboost_blend": {
        "anchor_weight": 0.60,
        "catboost_weight": 0.40,
    },
    "rule_layer": {
        "rules_strength": 0.55,
        "bias_mw": 0.85,
        "profile": "density_boost",
    },
    "output": {
        "default_header": True,
        "prediction_column": "prediction",
    },
    "random_seed": 42,
}


# name, legacy_weight, catboost_weight, rules_strength, bias_mw, profile
# This is intentionally narrow. V1.1 already works; we only probe nearby.
CANDIDATES: list[tuple[str, float, float, float, float, str]] = [
    # Bias around accepted V1.1
    ("v1_2_bias0p72_density", 0.70, 0.40, 0.55, 0.72, "density_boost"),
    ("v1_2_bias0p78_density", 0.70, 0.40, 0.55, 0.78, "density_boost"),
    ("v1_2_bias0p82_density", 0.70, 0.40, 0.55, 0.82, "density_boost"),
    ("v1_2_bias0p88_density", 0.70, 0.40, 0.55, 0.88, "density_boost"),
    ("v1_2_bias0p92_density", 0.70, 0.40, 0.55, 0.92, "density_boost"),
    ("v1_2_bias0p98_density", 0.70, 0.40, 0.55, 0.98, "density_boost"),

    # Rule strength around accepted V1.1
    ("v1_2_rules0p48_bias0p85_density", 0.70, 0.40, 0.48, 0.85, "density_boost"),
    ("v1_2_rules0p52_bias0p85_density", 0.70, 0.40, 0.52, 0.85, "density_boost"),
    ("v1_2_rules0p58_bias0p85_density", 0.70, 0.40, 0.58, 0.85, "density_boost"),
    ("v1_2_rules0p62_bias0p85_density", 0.70, 0.40, 0.62, 0.85, "density_boost"),

    # Anchor XGB ratio
    ("v1_2_xgb68_cat40_rules55_bias85_density", 0.68, 0.40, 0.55, 0.85, "density_boost"),
    ("v1_2_xgb72_cat40_rules55_bias85_density", 0.72, 0.40, 0.55, 0.85, "density_boost"),
    ("v1_2_xgb75_cat40_rules55_bias85_density", 0.75, 0.40, 0.55, 0.85, "density_boost"),

    # CatBoost weight
    ("v1_2_xgb70_cat34_rules55_bias85_density", 0.70, 0.34, 0.55, 0.85, "density_boost"),
    ("v1_2_xgb70_cat37_rules55_bias85_density", 0.70, 0.37, 0.55, 0.85, "density_boost"),
    ("v1_2_xgb70_cat43_rules55_bias85_density", 0.70, 0.43, 0.55, 0.85, "density_boost"),
    ("v1_2_xgb70_cat46_rules55_bias85_density", 0.70, 0.46, 0.55, 0.85, "density_boost"),

    # Profile alternatives
    ("v1_2_profile_physics_strong", 0.70, 0.40, 0.55, 0.85, "physics_strong"),
    ("v1_2_profile_physics_xstrong", 0.70, 0.40, 0.55, 0.85, "physics_xstrong"),
    ("v1_2_profile_theory103", 0.70, 0.40, 0.55, 0.85, "theory103"),
    ("v1_2_profile_physics_plus_dir_tiny", 0.70, 0.40, 0.55, 0.85, "physics_plus_dir_tiny"),
]


FIRST_SUBMIT_ORDER = [
    "v1_2_bias0p92_density.csv",
    "v1_2_xgb72_cat40_rules55_bias85_density.csv",
    "v1_2_xgb70_cat43_rules55_bias85_density.csv",
    "v1_2_rules0p58_bias0p85_density.csv",
    "v1_2_profile_physics_strong.csv",
]


def make_config(
    *,
    legacy_weight: float,
    catboost_weight: float,
    rules_strength: float,
    bias_mw: float,
    profile: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(BASE_CONFIG)

    legacy_weight = float(legacy_weight)
    depth4_weight = 1.0 - legacy_weight

    catboost_weight = float(catboost_weight)
    anchor_weight = 1.0 - catboost_weight

    cfg["xgb_anchor"]["legacy1700_weight"] = legacy_weight
    cfg["xgb_anchor"]["depth4_1892_weight"] = depth4_weight

    cfg["catboost_blend"]["anchor_weight"] = anchor_weight
    cfg["catboost_blend"]["catboost_weight"] = catboost_weight

    cfg["rule_layer"]["rules_strength"] = float(rules_strength)
    cfg["rule_layer"]["bias_mw"] = float(bias_mw)
    cfg["rule_layer"]["profile"] = str(profile)

    return cfg


def _prediction_distance_report(preds: dict[str, np.ndarray], baseline_name: str) -> list[dict[str, Any]]:
    base = preds[baseline_name]
    rows: list[dict[str, Any]] = []
    for name, pred in preds.items():
        rows.append(
            {
                "candidate": name,
                "mean": float(np.mean(pred)),
                "std": float(np.std(pred)),
                "min": float(np.min(pred)),
                "max": float(np.max(pred)),
                "mae_to_baseline": float(np.mean(np.abs(pred - base))),
                "corr_to_baseline": float(np.corrcoef(pred, base)[0, 1]) if len(pred) > 1 else float("nan"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--artifact_dir", default="artifacts_v1_0")
    parser.add_argument("--output_dir", default="submissions_v1_2")
    parser.add_argument("--report_dir", default="reports_v1_2")
    parser.add_argument("--header", action="store_true", default=True)
    parser.add_argument("--no_header", dest="header", action="store_false")
    args = parser.parse_args()

    raw = read_csv(args.features_path)

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    preds: dict[str, np.ndarray] = {}
    submit_paths: list[str] = []

    for name, legacy_w, cat_w, rules_s, bias, profile in CANDIDATES:
        cfg = make_config(
            legacy_weight=legacy_w,
            catboost_weight=cat_w,
            rules_strength=rules_s,
            bias_mw=bias,
            profile=profile,
        )

        pred = predict_with_v1_pipeline(raw, artifact_dir=artifact_dir, config=cfg)
        preds[name + ".csv"] = pred

        out_path = output_dir / f"{name}.csv"
        cfg_path = output_dir / f"{name}.json"

        write_submission(out_path, pred, header=args.header)
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        summary(
            report_dir / f"{name}.summary.json",
            pred,
            {"candidate": name, "config": cfg, "header": bool(args.header)},
        )

        submit_paths.append(str(out_path))

    baseline_name = "v1_2_bias0p88_density.csv" if "v1_2_bias0p88_density.csv" in preds else next(iter(preds))
    distance_rows = _prediction_distance_report(preds, baseline_name=baseline_name)
    (report_dir / "candidate_distance_report.json").write_text(
        json.dumps(distance_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (output_dir / "SUBMIT_FIRST.txt").write_text(
        "\n".join(str(output_dir / x) for x in FIRST_SUBMIT_ORDER if (output_dir / x).exists()) + "\n",
        encoding="utf-8",
    )

    (output_dir / "SUBMIT_ALL.txt").write_text("\n".join(submit_paths) + "\n", encoding="utf-8")

    print("Generated candidates:")
    for p in submit_paths:
        print(p)

    print(f"\nFirst submit list: {output_dir / 'SUBMIT_FIRST.txt'}")
    print(f"All candidates list: {output_dir / 'SUBMIT_ALL.txt'}")
    print(f"Distance report: {report_dir / 'candidate_distance_report.json'}")


if __name__ == "__main__":
    main()
