from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .feature_engineering import (
    available_capacity_from_raw,
    find_datetime_col,
    find_target_col,
    make_features,
    sort_by_time_if_possible,
)
from .train_predict import read_csv, write_submission, summary, rule_delta, profile_delta, clip_pred
from .predict_2027 import load_pred_from_artifact


def select_features(fe: pd.DataFrame, target_col: str) -> list[str]:
    dt_col = find_datetime_col(fe)
    drop = {target_col}
    if dt_col:
        drop.add(dt_col)
    return [c for c in fe.columns if c not in drop and pd.api.types.is_numeric_dtype(fe[c]) and fe[c].notna().sum() > 0]


def _tag_float(x: float) -> str:
    return str(round(float(x), 4)).replace('.', 'p')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', required=True)
    p.add_argument('--valid_path', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--artifact_dir', default='artifacts_beta13')
    p.add_argument('--submission_dir', default='submissions_beta13_cat')
    p.add_argument('--report_dir', default='reports_beta13_cat')
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--iterations', type=int, default=2500)
    args = p.parse_args()

    try:
        from catboost import CatBoostRegressor
    except Exception as e:
        raise SystemExit('CatBoost is not installed. Skip companion model or install catboost. Original error: ' + repr(e))

    art = Path(args.artifact_dir)
    out_dir = Path(args.submission_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(args.report_dir); rep_dir.mkdir(parents=True, exist_ok=True)

    raw = read_csv(args.train_path).rename(columns={'Кол-во_ВЭУ_в_ремонте': 'repair_count'})
    target_col = find_target_col(raw, args.target)
    raw = sort_by_time_if_possible(raw)
    y_raw = pd.to_numeric(raw[target_col], errors='coerce')
    cap = available_capacity_from_raw(raw)
    y = y_raw.clip(lower=0.0, upper=cap)
    ok = y.notna()
    raw = raw.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True)
    valid_raw = read_csv(args.valid_path)

    fe = make_features(raw, target_col=target_col, use_lags=True, use_time=True, use_availability=True)
    cols = select_features(fe, target_col)
    imp = SimpleImputer(strategy='median')
    X = pd.DataFrame(imp.fit_transform(fe[cols]), columns=cols)

    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=7.0,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=args.seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)

    fev = make_features(valid_raw, target_col=None, use_lags=True, use_time=True, use_availability=True)
    for c in cols:
        if c not in fev.columns:
            fev[c] = np.nan
    Xv = pd.DataFrame(imp.transform(fev[cols]), columns=cols)
    cat_pred = clip_pred(valid_raw, model.predict(Xv))

    # Reconstruct current production anchor from saved XGBoost artifacts.
    p1700 = load_pred_from_artifact(art / 'legacy1700.joblib', valid_raw)
    pd4 = load_pred_from_artifact(art / 'depth4_1892.joblib', valid_raw)
    # beta10/beta11 best anchor family: 70% legacy1700 + 30% depth4_1892.
    anchor70 = 0.70 * p1700 + 0.30 * pd4
    # The pure XGB best from beta11 was slightly closer to w68; keep as alternative anchor for blends.
    anchor68 = 0.68 * p1700 + 0.32 * pd4

    entries_first: list[str] = []
    entries_second: list[str] = []

    def save(name: str, pred: np.ndarray, extra: dict, bucket: list[str]) -> None:
        pred = clip_pred(valid_raw, pred)
        path = out_dir / name
        write_submission(path, pred)
        summary(rep_dir / name.replace('.csv', '.summary.json'), pred, extra)
        if path.exists():
            bucket.append(str(path))

    def candidate(
        *,
        anchor_name: str,
        anchor: np.ndarray,
        anchor_weight: float,
        rules_strength: float,
        bias: float,
        profile: str,
        bucket: list[str],
    ) -> None:
        base = anchor_weight * anchor + (1.0 - anchor_weight) * cat_pred
        pred = base + rule_delta(valid_raw, base, rules_strength) + profile_delta(valid_raw, base, profile) + bias
        name = (
            f'beta13_catblend_{anchor_name}_cat_w{_tag_float(anchor_weight)}'
            f'_rules{int(round(rules_strength * 100)):02d}'
            f'_bias{_tag_float(bias)}_{profile}.csv'
        )
        save(name, pred, {
            'anchor': anchor_name,
            'anchor_weight': anchor_weight,
            'cat_weight': 1.0 - anchor_weight,
            'rules_strength': rules_strength,
            'bias': bias,
            'profile': profile,
            'kind': 'catboost_companion_blend',
        }, bucket)

    # FIRST WAVE: final narrow search around beta12 best.
    # Observed on leaderboard:
    #   anchor70 weight 0.80 -> 8.3606
    #   0.70 -> 8.3500
    #   0.65 -> 8.3466
    #   0.60 -> 8.3449 (best so far)
    #   0.55 -> 8.3453
    # So the minimum is likely around 0.58-0.62, with bias slightly above 0.75.
    for aw in [0.56, 0.58, 0.60, 0.62]:
        candidate(anchor_name='anchor70', anchor=anchor70, anchor_weight=aw, rules_strength=0.55, bias=0.75, profile='physics_strong', bucket=entries_first)

    # Bias refinement: beta12 showed bias0.90 helped at anchor_weight 0.70;
    # check whether the optimum near 0.60 also wants a bit more bias.
    for aw, b in [(0.58, 0.85), (0.60, 0.85), (0.62, 0.85), (0.60, 0.95)]:
        candidate(anchor_name='anchor70', anchor=anchor70, anchor_weight=aw, rules_strength=0.55, bias=b, profile='physics_strong', bucket=entries_first)

    # Alternative anchor slightly more depth4-heavy; submit only if needed.
    candidate(anchor_name='anchor68', anchor=anchor68, anchor_weight=0.58, rules_strength=0.55, bias=0.75, profile='physics_strong', bucket=entries_first)
    candidate(anchor_name='anchor68', anchor=anchor68, anchor_weight=0.60, rules_strength=0.55, bias=0.75, profile='physics_strong', bucket=entries_first)

    # SECOND WAVE: send only if first wave improves beta12 best.
    # It checks profile/rules shape around the best basin, plus slightly more CatBoost.
    for aw in [0.52, 0.54, 0.64, 0.66]:
        candidate(anchor_name='anchor70', anchor=anchor70, anchor_weight=aw, rules_strength=0.55, bias=0.75, profile='physics_strong', bucket=entries_second)
    for aw in [0.58, 0.60, 0.62]:
        candidate(anchor_name='anchor70', anchor=anchor70, anchor_weight=aw, rules_strength=0.60, bias=0.85, profile='physics_strong', bucket=entries_second)
    candidate(anchor_name='anchor70', anchor=anchor70, anchor_weight=0.60, rules_strength=0.50, bias=0.85, profile='physics_strong', bucket=entries_second)
    candidate(anchor_name='anchor70', anchor=anchor70, anchor_weight=0.60, rules_strength=0.55, bias=0.85, profile='physics_xstrong', bucket=entries_second)
    candidate(anchor_name='anchor70', anchor=anchor70, anchor_weight=0.60, rules_strength=0.55, bias=0.85, profile='density_boost', bucket=entries_second)
    candidate(anchor_name='anchor68', anchor=anchor68, anchor_weight=0.56, rules_strength=0.55, bias=0.85, profile='physics_strong', bucket=entries_second)
    candidate(anchor_name='anchor68', anchor=anchor68, anchor_weight=0.58, rules_strength=0.55, bias=0.85, profile='physics_strong', bucket=entries_second)

    # Diagnostics, not in first list by default.
    save('beta13_catboost_raw_rules55_bias0p75_physics_strong.csv',
         cat_pred + rule_delta(valid_raw, cat_pred, 0.55) + profile_delta(valid_raw, cat_pred, 'physics_strong') + 0.75,
         {'kind': 'catboost_raw', 'rules_strength': 0.55, 'bias': 0.75, 'profile': 'physics_strong'}, entries_second)

    (out_dir / 'SUBMIT_FIRST.txt').write_text('\n'.join(entries_first) + '\n', encoding='utf-8')
    (out_dir / 'SUBMIT_SECOND.txt').write_text('\n'.join(entries_second) + '\n', encoding='utf-8')
    meta = {'iterations': args.iterations, 'depth': 6, 'learning_rate': 0.03, 'n_features': len(cols), 'feature_cols': cols}
    joblib.dump({'model': model, 'imputer': imp, 'feature_cols': cols, 'target_col': target_col, 'use_lags': True, 'use_time': True, 'use_availability': True}, Path(args.artifact_dir) / 'catboost_companion.joblib')
    (Path(args.artifact_dir) / 'beta13_catboost_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Generated CatBoost companion candidates:')
    for e in entries_first:
        print(e)
    print(f'\nFirst wave: {out_dir / "SUBMIT_FIRST.txt"}')
    print(f'Second wave: {out_dir / "SUBMIT_SECOND.txt"}')


if __name__ == '__main__':
    main()
