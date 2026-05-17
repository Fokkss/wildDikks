from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .feature_engineering import (
    FARM_CAPACITY_MW,
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', required=True)
    p.add_argument('--valid_path', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--artifact_dir', default='artifacts_beta11')
    p.add_argument('--submission_dir', default='submissions_beta11_cat')
    p.add_argument('--report_dir', default='reports_beta11_cat')
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

    raw = read_csv(args.train_path).rename(columns={'Кол-во_ВЭУ_в_ремонте':'repair_count'})
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
    anchor = 0.70 * p1700 + 0.30 * pd4

    entries = []
    def save(name: str, pred: np.ndarray, extra: dict):
        pred = clip_pred(valid_raw, pred)
        path = out_dir / name
        write_submission(path, pred)
        summary(rep_dir / name.replace('.csv', '.summary.json'), pred, extra)
        entries.append(str(path))

    # The cat model is intentionally a companion, not a replacement.
    for w in [0.95, 0.90, 0.85, 0.80]:
        base = w * anchor + (1.0 - w) * cat_pred
        pred = base + rule_delta(valid_raw, base, 0.55) + profile_delta(valid_raw, base, 'physics_strong') + 0.75
        save(f'beta11_catblend_anchor70_cat_w{str(w).replace(".", "p")}_rules55_bias0p75_physics_strong.csv', pred, {
            'anchor_weight': w, 'cat_weight': 1.0-w, 'rules_strength': 0.55, 'bias': 0.75, 'profile': 'physics_strong', 'kind': 'catboost_companion_blend'
        })

    save('beta11_catboost_raw_rules55_bias0p75_physics_strong.csv', cat_pred + rule_delta(valid_raw, cat_pred, 0.55) + profile_delta(valid_raw, cat_pred, 'physics_strong') + 0.75, {
        'kind': 'catboost_raw', 'rules_strength': 0.55, 'bias': 0.75, 'profile': 'physics_strong'
    })

    (out_dir / 'SUBMIT_CATBOOST.txt').write_text('\n'.join(entries) + '\n', encoding='utf-8')
    meta = {'iterations': args.iterations, 'depth': 6, 'learning_rate': 0.03, 'n_features': len(cols), 'feature_cols': cols}
    (Path(args.artifact_dir) / 'beta11_catboost_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Generated CatBoost companion candidates:')
    for e in entries:
        print(e)

if __name__ == '__main__':
    main()
