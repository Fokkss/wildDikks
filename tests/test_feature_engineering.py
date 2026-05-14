import numpy as np
import pandas as pd

from new_model.config import FARM_CAPACITY_MW, N_TURBINES
from new_model.feature_engineering import (
    available_capacity_from_raw,
    find_datetime_col,
    find_target_col,
    make_features,
    sort_by_time_if_possible,
)


def test_find_datetime_col_detects_timestamp():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00"],
            "wind_speed_80m": [8.0],
        }
    )

    assert find_datetime_col(df) == "timestamp"


def test_find_target_col_detects_russian_target():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00"],
            "Результирующий расчет": [42.0],
        }
    )

    assert find_target_col(df) == "Результирующий расчет"


def test_make_features_adds_expected_wind_features():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "wind_speed_10m": [4.0, 6.0, 9.0],
            "wind_speed_80m": [5.0, 8.0, 12.0],
            "wind_dir_80m": [0.0, 90.0, 180.0],
            "temperature_80m": [5.0, 6.0, 7.0],
            "pressure": [1013.0, 1012.0, 1011.0],
            "repair": [0, 1, 2],
        }
    )

    fe = make_features(df)

    expected_cols = {
        "hour_of_day",
        "month",
        "hour_sin",
        "hour_cos",
        "ws_10m",
        "ws_80m",
        "wind_speed_cube",
        "temp_k",
        "pressure_pa",
        "air_density",
        "available_turbines",
        "available_capacity_mw",
    }

    missing = expected_cols - set(fe.columns)
    assert not missing, f"Missing engineered columns: {missing}"

    assert fe["ws_80m"].tolist() == [5.0, 8.0, 12.0]
    assert fe["wind_speed_cube"].tolist() == [125.0, 512.0, 1728.0]
    assert fe["wind_shear"].tolist() == [1.0, 2.0, 3.0]
    assert np.isclose(fe.loc[1, "available_turbines"], N_TURBINES - 1)


def test_available_capacity_respects_repair_count():
    df = pd.DataFrame({"repair": [0, 1, 26, 30]})

    cap = available_capacity_from_raw(df)

    assert np.isclose(cap.iloc[0], FARM_CAPACITY_MW)
    assert cap.iloc[1] < FARM_CAPACITY_MW
    assert np.isclose(cap.iloc[2], 0.0)
    assert np.isclose(cap.iloc[3], 0.0)


def test_sort_by_time_if_possible_sorts_chronologically():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 02:00:00",
                "2024-01-01 00:00:00",
                "2024-01-01 01:00:00",
            ],
            "wind_speed_80m": [2.0, 0.0, 1.0],
        }
    )

    sorted_df = sort_by_time_if_possible(df)

    assert sorted_df["wind_speed_80m"].tolist() == [0.0, 1.0, 2.0]
