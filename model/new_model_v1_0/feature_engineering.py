from __future__ import annotations

import re
import math
from typing import Optional, Iterable

import numpy as np
import pandas as pd

FARM_CAPACITY_MW = 90.09
N_TURBINES = 26
TURBINE_CAPACITY_MW = FARM_CAPACITY_MW / N_TURBINES
HUB_HEIGHT_M = 84.0
CUT_IN_MS = 3.0
RATED_MS_THEORY = 10.3
RATED_MS_LEGACY = 12.0
CUT_OUT_MS = 25.0
R_DRY_AIR = 287.05
STD_AIR_DENSITY = 1.225
BETZ_CP = 16.0 / 27.0
ROTOR_DIAMETER_M = 132.0
ROTOR_AREA_M2 = math.pi * (ROTOR_DIAMETER_M / 2.0) ** 2
G_ACCEL = 9.80665
EPS = 1e-6

KIND_KEYWORDS: dict[str, tuple[str, ...]] = {
    "wind_speed": ("wind_speed", "windspeed", "wind speed", "ws", "speed", "скорость", "ветер"),
    "wind_dir": ("wind_direction", "wind_dir", "wind direction", "direction", "dir", "wd", "направ", "азимут", "румб"),
    "gust": ("gust", "gusts", "порыв"),
    "temp": ("temperature", "temp", "t2m", "температура", "темп"),
    "pressure": ("pressure", "press", "msl", "sp", "давление"),
    "precip": ("precip", "rain", "snow", "showers", "осад", "дожд", "снег", "ливн"),
    "cloud": ("cloud", "облач"),
    "repair": ("repair", "ремонт", "веу", "turbines_off", "unavailable", "offline"),
}
DATE_KEYWORDS = ("datetime", "timestamp", "date", "time", "dt", "дата", "время")
TARGET_KEYWORDS = ("результирующий", "выработка", "generation", "power", "target", "fact", "actual")

RENAME_MAP = {
    "РљРѕР»-РІРѕ_Р’Р­РЈ_РІ_СЂРµРјРѕРЅС‚Рµ": "repair_count",
    "Кол-во_ВЭУ_в_ремонте": "repair_count",
    "Количество_ВЭУ_в_ремонте": "repair_count",
}

def norm(s: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", "_", str(s).lower()).strip("_")

def find_datetime_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        n = norm(c)
        if any(k in n for k in DATE_KEYWORDS) and not any(x in n for x in ("hour", "month", "час", "месяц")):
            return c
    return None

def find_target_col(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested and requested in df.columns:
        return requested
    if requested:
        rn = norm(requested)
        for c in df.columns:
            if norm(c) == rn:
                return c
    cand = []
    for c in df.columns:
        n = norm(c)
        if any(k in n for k in TARGET_KEYWORDS):
            cand.append(c)
    if len(cand) == 1:
        return cand[0]
    for prefer in ("результирующий", "выработка", "generation", "power", "target"):
        for c in cand:
            if prefer in norm(c):
                return c
    raise ValueError("Target column not found. Pass --target explicitly.")

def _has_height(n: str, h: int) -> bool:
    return bool(re.search(rf"(^|_)({h})(m|м|meter|meters)?(_|$)", n)) or str(h) in n

def find_col(df: pd.DataFrame, kind: str, height_m: Optional[int] = None) -> Optional[str]:
    keys = KIND_KEYWORDS[kind]
    scored = []
    for c in df.columns:
        n = norm(c)
        if not any(k in n for k in keys):
            continue
        if height_m is not None and not _has_height(n, height_m):
            continue
        if kind == "wind_dir" and any(x in n for x in ("speed", "скорость")):
            continue
        if kind == "wind_speed" and any(x in n for x in ("direction", "dir", "направ")):
            continue
        score = 0
        if height_m is not None and _has_height(n, height_m):
            score += 10
        if kind == "wind_speed" and ("wind_speed" in n or "скорость" in n):
            score += 5
        if kind == "wind_dir" and ("wind_direction" in n or "направ" in n):
            score += 5
        if kind == "repair" and "repair_count" in n:
            score += 20
        score -= len(n) // 30
        scored.append((score, c))
    return sorted(scored, key=lambda x: (-x[0], norm(x[1])))[0][1] if scored else None

def find_cols(df: pd.DataFrame, kind: str) -> list[str]:
    keys = KIND_KEYWORDS[kind]
    return [c for c in df.columns if any(k in norm(c) for k in keys)]

def num(df: pd.DataFrame, col: Optional[str], default: float | None = None) -> pd.Series:
    if col is None or col not in df.columns:
        if default is None:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")

def safe_div(a, b, default=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.asarray(a, dtype=float) / np.asarray(b, dtype=float)
    return np.where(np.isfinite(x), x, default)

def temp_to_c(temp: pd.Series) -> pd.Series:
    med = temp.dropna().median() if temp.notna().any() else np.nan
    return temp - 273.15 if pd.notna(med) and med > 150 else temp

def temp_to_k(temp: pd.Series) -> pd.Series:
    med = temp.dropna().median() if temp.notna().any() else np.nan
    return temp if pd.notna(med) and med > 150 else temp + 273.15

def pressure_to_pa(p: pd.Series) -> pd.Series:
    med = p.dropna().median() if p.notna().any() else np.nan
    return p * 100.0 if pd.notna(med) and med < 2000 else p

def to_dir_deg(raw: pd.Series) -> pd.Series:
    raw = pd.to_numeric(raw, errors="coerce")
    q99 = raw.dropna().quantile(0.99) if raw.notna().any() else np.nan
    # competition data: 0.273 means 273 degrees. Friend's code did not handle this.
    if pd.notna(q99) and q99 <= 1.5:
        return (raw * 1000.0) % 360.0
    return raw % 360.0

def circular_diff(a_deg: pd.Series, b_deg: pd.Series) -> pd.Series:
    return ((a_deg - b_deg + 180.0) % 360.0) - 180.0

def add_time_features(out: pd.DataFrame, use_time: bool) -> pd.DataFrame:
    if not use_time:
        return out
    dt_col = find_datetime_col(out)
    if dt_col:
        dt = pd.to_datetime(out[dt_col], errors="coerce")
        out["year"] = dt.dt.year
        out["month"] = dt.dt.month
        out["dayofyear"] = dt.dt.dayofyear
        out["dayofweek"] = dt.dt.dayofweek
        out["hour_of_day"] = dt.dt.hour
    month = out.get("month")
    hour = out.get("hour_of_day")
    doy = out.get("dayofyear")
    if month is not None:
        month = pd.to_numeric(month, errors="coerce")
        out["month_sin"] = np.sin(2*np.pi*month/12)
        out["month_cos"] = np.cos(2*np.pi*month/12)
        out["is_q1"] = month.isin([1,2,3]).astype(float)
    if hour is not None:
        hour = pd.to_numeric(hour, errors="coerce")
        out["hour_sin"] = np.sin(2*np.pi*hour/24)
        out["hour_cos"] = np.cos(2*np.pi*hour/24)
    if doy is not None:
        doy = pd.to_numeric(doy, errors="coerce")
        out["doy_sin"] = np.sin(2*np.pi*doy/365.25)
        out["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    return out

def add_availability(out: pd.DataFrame, use_availability: bool) -> pd.DataFrame:
    if not use_availability:
        return out
    c = find_col(out, "repair")
    r = num(out, c, 0.0).fillna(0.0).clip(0, N_TURBINES)
    out["repair_count_inferred"] = r
    out["available_turbines"] = N_TURBINES - r
    out["availability_ratio"] = out["available_turbines"] / N_TURBINES
    out["available_capacity_mw"] = out["available_turbines"] * TURBINE_CAPACITY_MW
    return out

def add_lag_features(out: pd.DataFrame, value: pd.Series, prefix: str, dt: Optional[pd.Series], windows=(3,6,12)) -> pd.DataFrame:
    # Time-aware mapping; preserves original row order. Falls back to row-order shift.
    v = pd.to_numeric(value, errors="coerce")
    if dt is not None:
        tmp = pd.DataFrame({"dt": dt, "v": v}).dropna(subset=["dt"]).drop_duplicates("dt", keep="last").set_index("dt")["v"].sort_index()
        if not tmp.empty:
            for h in (1,2,3):
                out[f"{prefix}_lag{h}"] = (dt - pd.Timedelta(hours=h)).map(tmp)
            out[f"{prefix}_diff1"] = v - out[f"{prefix}_lag1"]
            shifted = tmp.shift(1)
            for w in windows:
                out[f"{prefix}_roll_mean_{w}"] = dt.map(shifted.rolling(f"{w}h", min_periods=1).mean())
                out[f"{prefix}_roll_std_{w}"] = dt.map(shifted.rolling(f"{w}h", min_periods=2).std())
            return out
    out[f"{prefix}_lag1"] = v.shift(1)
    out[f"{prefix}_diff1"] = v - v.shift(1)
    for w in windows:
        out[f"{prefix}_roll_mean_{w}"] = v.shift(1).rolling(w, min_periods=1).mean()
        out[f"{prefix}_roll_std_{w}"] = v.shift(1).rolling(w, min_periods=2).std()
    return out
    
def add_past_sum_features(out: pd.DataFrame, value: pd.Series, prefix: str, dt: Optional[pd.Series], windows=(3, 6, 12, 24)) -> pd.DataFrame:
    v = pd.to_numeric(value, errors="coerce").fillna(0.0)
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
            for w in windows:
                out[f"{prefix}_sum_prev_{w}h"] = dt.map(
                    shifted.rolling(f"{w}h", min_periods=1).sum()
                )
            return out
    shifted = v.shift(1)
    for w in windows:
        out[f"{prefix}_sum_prev_{w}h"] = shifted.rolling(
            w,
            min_periods=1,
        ).sum()
    return out

def make_features(df: pd.DataFrame, target_col: Optional[str] = None, *, use_lags: bool = True, use_time: bool = True, use_availability: bool = True) -> pd.DataFrame:
    out = df.copy().rename(columns=RENAME_MAP)
    dt_col = find_datetime_col(out)
    dt = pd.to_datetime(out[dt_col], errors="coerce") if dt_col else None
    out = add_time_features(out, use_time=use_time)
    out = add_availability(out, use_availability=use_availability)

    ws_cols = {h: find_col(out, "wind_speed", h) for h in (10,80,120,180)}
    ws = {h: num(out, ws_cols[h]) for h in (10,80,120,180)}
    for h, s in ws.items():
        if s.notna().any():
            out[f"ws_{h}m"] = s
            out[f"ws_{h}m_sq"] = s**2
            out[f"ws_{h}m_cube"] = s**3

    ws80 = ws[80]
    ws120 = ws[120]
    ws10 = ws[10]
    ws180 = ws[180]
    if ws80.notna().any() and ws120.notna().any():
        valid = (ws80 > 0.05) & (ws120 > 0.05)
        alpha = pd.Series(np.nan, index=out.index)
        alpha.loc[valid] = np.log(ws120.loc[valid] / ws80.loc[valid]) / np.log(120/80)
        alpha = alpha.clip(-0.5, 1.0)
        out["shear_alpha_80_120_for_84"] = alpha
        hub84 = ws80 * (HUB_HEIGHT_M/80.0) ** alpha.fillna(0.0)
    elif ws80.notna().any():
        hub84 = ws80.copy()
        out["shear_alpha_80_120_for_84"] = 0.0
    else:
        hub84 = pd.Series(np.nan, index=out.index)
    out["hub_ws_84m"] = hub84
    out["hub_ws_84m_sq"] = hub84**2
    out["hub_ws_84m_cube"] = hub84**3
    # Legacy 12 m/s rated proxy is kept because it was empirically strong.
    out["hub_ws_84m_piece_3_12"] = (hub84.clip(CUT_IN_MS, RATED_MS_LEGACY)-CUT_IN_MS).clip(lower=0)
    pc84 = ((hub84.clip(CUT_IN_MS, RATED_MS_LEGACY)-CUT_IN_MS)/(RATED_MS_LEGACY-CUT_IN_MS))**3
    out["power_curve_84m"] = pc84.clip(0,1)
    # Theory turbine curve from Siemens Gamesa SG 3.4-132 / G132-IIA: rated around 10.3 m/s.
    out["hub_ws_84m_piece_3_103"] = (hub84.clip(CUT_IN_MS, RATED_MS_THEORY)-CUT_IN_MS).clip(lower=0)
    pc84_103 = ((hub84.clip(CUT_IN_MS, RATED_MS_THEORY)-CUT_IN_MS)/(RATED_MS_THEORY-CUT_IN_MS))**3
    out["power_curve_84m_103"] = pc84_103.clip(0,1)
    cap = out.get("available_capacity_mw", pd.Series(FARM_CAPACITY_MW, index=out.index))
    out["expected_power_84m_proxy"] = out["power_curve_84m"] * cap
    out["expected_power_84m_103_proxy"] = out["power_curve_84m_103"] * cap

    # Shear and speed ratios
    for a,b in ((10,80),(80,120),(80,180),(10,120),(120,180)):
        s1, s2 = ws[a], ws[b]
        if s1.notna().any() and s2.notna().any():
            out[f"ws_diff_{b}_{a}"] = s2 - s1
            out[f"ws_ratio_{b}_{a}"] = pd.Series(safe_div(s2.clip(0.05), s1.clip(0.05)), index=out.index).clip(0, 10)
            out[f"shear_alpha_{a}_{b}"] = pd.Series(safe_div(np.log(s2.clip(0.05)/s1.clip(0.05)), math.log(b/a)), index=out.index).clip(-0.5, 1.0)

    # Regime bins from 84m and 80m.
    for lo, hi, name in [(0,3,"ws0_3"),(3,5,"ws3_5"),(5,7,"ws5_7"),(7,9,"ws7_9"),(9,10,"ws9_10"),(10,11,"ws10_11"),(11,13,"ws11_13"),(13,100,"ws13p")]:
        out[name] = ((hub84 >= lo) & (hub84 < hi)).astype(float)
    out["below_cut_in"] = (hub84 < CUT_IN_MS).astype(float)
    out["rated_zone"] = ((hub84 >= RATED_MS_LEGACY) & (hub84 < CUT_OUT_MS)).astype(float)
    out["rated_zone_103"] = ((hub84 >= RATED_MS_THEORY) & (hub84 < CUT_OUT_MS)).astype(float)
    out["near_rated_9_103"] = ((hub84 >= 9.0) & (hub84 < RATED_MS_THEORY)).astype(float)

    # Direction in deg/1000-aware form.
    dirs: dict[int, pd.Series] = {}
    for h in (10,80,120,180):
        dc = find_col(out, "wind_dir", h)
        d = to_dir_deg(num(out, dc)) if dc else pd.Series(np.nan, index=out.index)
        if d.notna().any():
            dirs[h] = d
            rad = np.deg2rad(d)
            out[f"wd_{h}m_sin"] = np.sin(rad)
            out[f"wd_{h}m_cos"] = np.cos(rad)
            if h == 80:
                out["hub_ws_cube_x_wd_sin"] = hub84**3 * out[f"wd_{h}m_sin"]
                out["hub_ws_cube_x_wd_cos"] = hub84**3 * out[f"wd_{h}m_cos"]
                for lo, hi in [(0,45),(45,60),(60,75),(75,90),(90,105),(105,120),(120,135),(135,180),(180,225),(225,270),(270,315),(315,360)]:
                    out[f"dir{lo}_{hi}"] = ((d >= lo) & (d < hi)).astype(float)
                out["dir45_90"] = ((d >= 45) & (d < 90)).astype(float)
                out["dir90_135"] = ((d >= 90) & (d < 135)).astype(float)
    for a,b in ((10,80),(80,120),(80,180),(10,120)):
        if a in dirs and b in dirs:
            out[f"wd_diff_{b}_{a}"] = circular_diff(dirs[b], dirs[a])

    # Vector shear / veer using u/v components.
    if 80 in dirs and 120 in dirs and ws80.notna().any() and ws120.notna().any():
        u80 = ws80 * np.sin(np.deg2rad(dirs[80])); v80 = ws80 * np.cos(np.deg2rad(dirs[80]))
        u120 = ws120 * np.sin(np.deg2rad(dirs[120])); v120 = ws120 * np.cos(np.deg2rad(dirs[120]))
        out["vector_shear_120_80"] = np.sqrt((u120-u80)**2 + (v120-v80)**2)
        out["wind_veer_120_80"] = circular_diff(dirs[120], dirs[80])
        out["abs_wind_veer_120_80"] = out["wind_veer_120_80"].abs()
    if 80 in dirs and 180 in dirs and ws180.notna().any():
        u80 = ws80 * np.sin(np.deg2rad(dirs[80])); v80 = ws80 * np.cos(np.deg2rad(dirs[80]))
        u180 = ws180 * np.sin(np.deg2rad(dirs[180])); v180 = ws180 * np.cos(np.deg2rad(dirs[180]))
        out["vector_shear_180_80"] = np.sqrt((u180-u80)**2 + (v180-v80)**2)
        out["wind_veer_180_80"] = circular_diff(dirs[180], dirs[80])
        out["abs_wind_veer_180_80"] = out["wind_veer_180_80"].abs()

    # Air density and weather.
    temp_col = find_col(out, "temp", 80) or find_col(out, "temp", 120) or find_col(out, "temp")
    press_col = find_col(out, "pressure")
    temp = num(out, temp_col); press = num(out, press_col)
    if temp.notna().any():
        out["temp_c_inferred"] = temp_to_c(temp)
        out["temp_k_inferred"] = temp_to_k(temp)
        out["temp_gt5"] = (out["temp_c_inferred"] > 5).astype(float)
        out["temp_lt_m10"] = (out["temp_c_inferred"] < -10).astype(float)
    if press.notna().any():
        out["pressure_pa_inferred"] = pressure_to_pa(press)
        # thresholds in hPa-space for readability
        press_hpa = out["pressure_pa_inferred"] / 100.0
        out["pressure_low"] = (press_hpa < 1005).astype(float)
        out["pressure_midlow"] = ((press_hpa >= 1005) & (press_hpa < 1015)).astype(float)
        out["pressure_high"] = (press_hpa >= 1025).astype(float)
    # ------------------------------------------------------------
    # Thermal stratification / boundary-layer stability proxy
    # ------------------------------------------------------------
    temp80_col = find_col(out, "temp", 80)
    temp120_col = find_col(out, "temp", 120)
    temp80_raw = num(out, temp80_col)
    temp120_raw = num(out, temp120_col)
    temp80_c = temp_to_c(temp80_raw)
    temp120_c = temp_to_c(temp120_raw)
    if temp80_c.notna().any() and temp120_c.notna().any():
        out["temp_gradient_120_80"] = temp120_c - temp80_c
        out["lapse_rate_120_80_c_per_100m"] = (
            out["temp_gradient_120_80"] / (120.0 - 80.0) * 100.0
        )
        out["stable_layer_120_80"] = (
            out["temp_gradient_120_80"] > 0.2
        ).astype(float)
        out["unstable_layer_120_80"] = (
            out["temp_gradient_120_80"] < -0.4
        ).astype(float)
        if ws80.notna().any() and ws120.notna().any():
            temp_mid_k = temp_to_k((temp80_raw + temp120_raw) / 2.0)
            dtheta_dz = out["temp_gradient_120_80"] / (120.0 - 80.0)
            du_dz = (ws120 - ws80) / (120.0 - 80.0)
            ri = (
                G_ACCEL
                / (temp_mid_k + EPS)
                * dtheta_dz
                / ((du_dz ** 2) + EPS)
            )
            out["bulk_richardson_120_80_proxy"] = ri.clip(-10, 10)
            out["bulk_richardson_abs_120_80"] = (
                out["bulk_richardson_120_80_proxy"].abs()
            )
            out["stable_low_shear_risk"] = (
                (out["bulk_richardson_120_80_proxy"] > 0.25)
                & (ws120 >= 5.0)
            ).astype(float)
            out["unstable_high_shear_risk"] = (
                (out["bulk_richardson_120_80_proxy"] < -0.25)
                & ((ws120 - ws80).abs() > 1.0)
            ).astype(float)
    if "temp_k_inferred" in out and "pressure_pa_inferred" in out:
        rho = (out["pressure_pa_inferred"] / (R_DRY_AIR * out["temp_k_inferred"])).clip(0.7,1.6)
        out["air_density_kg_m3_proxy"] = rho
        out["wind_power_density_proxy"] = 0.5 * rho * out["hub_ws_84m_cube"]
        betz_one_turbine_mw = (
            0.5
            * rho
            * ROTOR_AREA_M2
            * out["hub_ws_84m_cube"]
            * BETZ_CP
            / 1_000_000.0
        )
        out["betz_one_turbine_mw"] = betz_one_turbine_mw
        if "available_turbines" in out.columns:
            out["betz_farm_mw"] = (
                betz_one_turbine_mw * out["available_turbines"]
            )
        else:
            out["betz_farm_mw"] = betz_one_turbine_mw * N_TURBINES
        if "available_capacity_mw" in out.columns:
            out["betz_farm_clipped_mw"] = np.minimum(
                out["betz_farm_mw"],
                out["available_capacity_mw"],
            )
        else:
            out["betz_farm_clipped_mw"] = np.minimum(
                out["betz_farm_mw"],
                FARM_CAPACITY_MW,
            )
        out["pc84_to_betz_ratio"] = pd.Series(
            safe_div(
                out["expected_power_84m_proxy"],
                out["betz_farm_clipped_mw"] + EPS,
            ),
            index=out.index,
        ).clip(0, 10)
        out["power_curve_density"] = out["expected_power_84m_proxy"] * rho / STD_AIR_DENSITY
        if "available_capacity_mw" in out.columns:
            out["power_curve_density_clipped"] = np.minimum(
                out["power_curve_density"],
                out["available_capacity_mw"],
            )
        if "expected_power_84m_103_proxy" in out.columns:
            out["power_curve_103_density"] = out["expected_power_84m_103_proxy"] * rho / STD_AIR_DENSITY

    gust_col = find_col(out, "gust", 10)
    gust = num(out, gust_col)
    if gust.notna().any():
        out["gust10"] = gust
        out["gust10_excess_ws84"] = (gust - hub84).clip(lower=0)
        out["gust10_to_ws84"] = pd.Series(safe_div(gust, hub84 + 1e-6), index=out.index).clip(0,10)

    precip_cols = find_cols(out, "precip")
    if precip_cols:
        vals = out[precip_cols].apply(pd.to_numeric, errors="coerce")
        out["precip_sum_inferred"] = vals.sum(axis=1, min_count=1).fillna(0)
        out["has_precip_inferred"] = (out["precip_sum_inferred"] > 0).astype(float)
    cloud_cols = find_cols(out, "cloud")
    if cloud_cols:
        vals = out[cloud_cols].apply(pd.to_numeric, errors="coerce")
        out["cloud_mean_inferred"] = vals.mean(axis=1)
        out["cloud_max_inferred"] = vals.max(axis=1)
        cloud = out["cloud_max_inferred"].copy()
        med = cloud.dropna().median() if cloud.notna().any() else np.nan
        if pd.notna(med) and med > 1.5:
            cloud = cloud / 100.0
        out["cloud_hi"] = (cloud > 0.65).astype(float)

    if use_lags and hub84.notna().any():
        out = add_lag_features(out, hub84, "hub_ws", dt)

    # Manual interaction columns for production calibration model.
    # These are features, not direct leaderboard-fitting.
    if "ws_diff_180_80" in out:
        out["shear180_positive"] = (out["ws_diff_180_80"] > 0.05).astype(float)
        out["shear180_positive_x_pc84"] = out["shear180_positive"] * out["power_curve_84m"]
    if "dir60_75" in out:
        out["dir60_75_x_pc84"] = out["dir60_75"] * out["power_curve_84m"]
    if "cloud_hi" in out:
        out["cloud_hi_x_pc84"] = out["cloud_hi"] * out["power_curve_84m"]
    if "pressure_low" in out:
        out["pressure_low_x_pc84"] = out["pressure_low"] * out["power_curve_84m"]
    if "temp_gt5" in out:
        out["temp_gt5_x_pc84"] = out["temp_gt5"] * out["power_curve_84m"]

    # Convert non-date, non-target columns.
    dt_col = find_datetime_col(out)
    for c in list(out.columns):
        if c == target_col or c == dt_col:
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            parsed = pd.to_numeric(out[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                out[c] = parsed
            else:
                out[c] = out[c].astype("category").cat.codes.replace(-1, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)

def available_capacity_from_raw(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy().rename(columns=RENAME_MAP)
    c = find_col(tmp, "repair")
    if c is None:
        return pd.Series(FARM_CAPACITY_MW, index=df.index)
    r = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0).clip(0, N_TURBINES)
    return ((N_TURBINES - r) * TURBINE_CAPACITY_MW).clip(0.0, FARM_CAPACITY_MW)

def sort_by_time_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    c = find_datetime_col(df)
    if c is None:
        return df.reset_index(drop=True)
    tmp = df.copy()
    tmp["__dt_sort"] = pd.to_datetime(tmp[c], errors="coerce")
    return tmp.sort_values("__dt_sort", kind="mergesort").drop(columns=["__dt_sort"]).reset_index(drop=True)
