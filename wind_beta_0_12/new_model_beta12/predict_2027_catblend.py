from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .feature_engineering import make_features
from .train_predict import read_csv, write_submission, summary, rule_delta, profile_delta, clip_pred
from .predict_2027 import load_pred_from_artifact


def load_cat_pred(artifact_path: Path, raw: pd.DataFrame) -> np.ndarray:
    meta = joblib.load(artifact_path)
    fe = make_features(
        raw,
        target_col=None,
        use_lags=meta.get('use_lags', True),
        use_time=meta.get('use_time', True),
        use_availability=meta.get('use_availability', True),
    )
    cols = meta['feature_cols']
    for c in cols:
        if c not in fe.columns:
            fe[c] = np.nan
    X = pd.DataFrame(meta['imputer'].transform(fe[cols]), columns=cols)
    return clip_pred(raw, meta['model'].predict(X))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--features_path', required=True)
    p.add_argument('--artifact_dir', default='artifacts_beta12')
    p.add_argument('--output_path', default='submissions_beta12/submission_2027_beta12_catblend.csv')
    p.add_argument('--anchor_weight', type=float, default=0.75)
    p.add_argument('--anchor_profile', default='anchor70', choices=['anchor70', 'anchor68'])
    p.add_argument('--rules_strength', type=float, default=0.55)
    p.add_argument('--bias', type=float, default=0.75)
    p.add_argument('--extra_profile', default='physics_strong', choices=['none', 'dircloud', 'physics', 'physics_strong', 'physics_xstrong', 'physics_ultra', 'physics_plus_dir_tiny', 'theory103', 'density_boost', 'all_mild', 'all_strong', 'lowrelax'])
    args = p.parse_args()

    raw = read_csv(args.features_path)
    art = Path(args.artifact_dir)
    p1700 = load_pred_from_artifact(art / 'legacy1700.joblib', raw)
    pd4 = load_pred_from_artifact(art / 'depth4_1892.joblib', raw)
    if args.anchor_profile == 'anchor70':
        anchor = 0.70 * p1700 + 0.30 * pd4
    else:
        anchor = 0.68 * p1700 + 0.32 * pd4
    cat = load_cat_pred(art / 'catboost_companion.joblib', raw)
    base = args.anchor_weight * anchor + (1.0 - args.anchor_weight) * cat
    pred = clip_pred(raw, base + rule_delta(raw, base, args.rules_strength) + profile_delta(raw, base, args.extra_profile) + args.bias)
    out = Path(args.output_path)
    write_submission(out, pred)
    summary(out.with_suffix('.summary.json'), pred, {
        'kind': 'catboost_companion_blend_2027',
        'anchor_profile': args.anchor_profile,
        'anchor_weight': args.anchor_weight,
        'cat_weight': 1.0 - args.anchor_weight,
        'rules_strength': args.rules_strength,
        'bias': args.bias,
        'extra_profile': args.extra_profile,
    })
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
