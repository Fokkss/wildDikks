from __future__ import annotations

"""
Optional feature patch for V1.2 retrain experiments.

It modifies model/new_model_v1_0/feature_engineering.py in-place and creates a .bak backup.
It adds:
  1) hub wind EMA / inertia features
  2) density-corrected hub wind and density-corrected power-curve features

Run from repository root:
    python model/new_model_v1_0/apply_feature_patch_v1_2.py

Then retrain with current train.py.
"""

from pathlib import Path


EMA_FUNCTION = r'''

def add_ema_features(out: pd.DataFrame, value: pd.Series, prefix: str, dt: Optional[pd.Series], spans=(3, 6, 12)) -> pd.DataFrame:
    """Time-aware EMA features for turbine inertia / aerodynamic response delay."""
    v = pd.to_numeric(value, errors="coerce")
    if dt is not None:
        tmp = (
            pd.DataFrame({"dt": dt, "v": v})
            .dropna(subset=["dt"])
            .drop_duplicates("dt", keep="last")
            .set_index("dt")["v"]
            .sort_index()
        )
        if not tmp.empty:
            shifted = tmp.shift(1)
            for span in spans:
                ema = shifted.ewm(span=span, adjust=False, min_periods=1).mean()
                out[f"{prefix}_ema_{span}"] = dt.map(ema)
                out[f"{prefix}_minus_ema_{span}"] = v - out[f"{prefix}_ema_{span}"]
            return out
    shifted = v.shift(1)
    for span in spans:
        ema = shifted.ewm(span=span, adjust=False, min_periods=1).mean()
        out[f"{prefix}_ema_{span}"] = ema
        out[f"{prefix}_minus_ema_{span}"] = v - ema
    return out
'''

DENSITY_PATCH = r'''
 out["rho_ratio_proxy"] = (rho / STD_AIR_DENSITY).clip(0.5, 1.5) out["hub_ws_84m_density_corrected"] = hub84 * (out["rho_ratio_proxy"] ** (1.0 / 3.0)) out["hub_ws_84m_density_corrected_sq"] = out["hub_ws_84m_density_corrected"] ** 2 out["hub_ws_84m_density_corrected_cube"] = out["hub_ws_84m_density_corrected"] ** 3 pc84_density_corr = ((out["hub_ws_84m_density_corrected"].clip(CUT_IN_MS, RATED_MS_LEGACY)-CUT_IN_MS)/(RATED_MS_LEGACY-CUT_IN_MS))**3 out["power_curve_84m_density_corrected"] = pc84_density_corr.clip(0, 1) if "available_capacity_mw" in out.columns: out["expected_power_84m_density_corrected"] = out["power_curve_84m_density_corrected"] * out["available_capacity_mw"] else: out["expected_power_84m_density_corrected"] = out["power_curve_84m_density_corrected"] * FARM_CAPACITY_MW
'''


def patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    if "def add_ema_features(" not in text:
        anchor = "def add_past_sum_features"
        if anchor not in text:
            raise SystemExit("Could not find anchor for add_ema_features insertion.")
        text = text.replace(anchor, EMA_FUNCTION + "\n" + anchor, 1)

    old_lag_call = 'if use_lags and hub84.notna().any(): out = add_lag_features(out, hub84, "hub_ws", dt)'
    new_lag_call = 'if use_lags and hub84.notna().any(): out = add_lag_features(out, hub84, "hub_ws", dt) out = add_ema_features(out, hub84, "hub_ws", dt)'
    if 'add_ema_features(out, hub84, "hub_ws", dt)' not in text:
        if old_lag_call not in text:
            raise SystemExit("Could not find lag call anchor for EMA insertion.")
        text = text.replace(old_lag_call, new_lag_call, 1)

    density_anchor = 'out["power_curve_density"] = out["expected_power_84m_proxy"] * rho / STD_AIR_DENSITY'
    if "hub_ws_84m_density_corrected" not in text:
        if density_anchor not in text:
            raise SystemExit("Could not find density anchor for density-corrected wind insertion.")
        text = text.replace(density_anchor, density_anchor + DENSITY_PATCH, 1)

    if text == original:
        print("No changes needed; patch already applied.")
        return

    backup = path.with_suffix(path.suffix + ".v1_2.bak")
    backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"Patched: {path}")
    print(f"Backup:  {backup}")


def main() -> None:
    path = Path("model/new_model_v1_0/feature_engineering.py")
    if not path.exists():
        path = Path(__file__).with_name("feature_engineering.py")
    if not path.exists():
        raise SystemExit("feature_engineering.py not found. Run from repo root or place script near it.")
    patch_file(path)


if __name__ == "__main__":
    main()
