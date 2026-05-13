import json

import pandas as pd
import pytest

from new_model.train import main as train_main


def make_train_frame(n_rows: int = 80) -> pd.DataFrame:
    dt = pd.date_range("2024-01-01", periods=n_rows, freq="h")

    ws = 6 + 2 * (pd.Series(range(n_rows)) % 24) / 24
    repair = pd.Series([0, 1, 0, 2] * (n_rows // 4 + 1))[:n_rows]

    # Simple synthetic target: power grows with wind speed and decreases with repair.
    y = (ws**3) * 0.08
    y = y.clip(0, 90.09)
    y = y * (26 - repair) / 26

    return pd.DataFrame(
        {
            "timestamp": dt,
            "wind_speed_80m": ws,
            "wind_dir_80m": 180,
            "temperature_80m": 5,
            "pressure": 1013,
            "repair": repair,
            "Результирующий расчет": y,
        }
    )


@pytest.mark.integration
def test_train_creates_artifacts(tmp_path, monkeypatch):
    train_path = tmp_path / "train_dataset.csv"
    model_dir = tmp_path / "artifacts"

    df = make_train_frame()
    df.to_csv(train_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_path",
            str(train_path),
            "--model_dir",
            str(model_dir),
            "--target",
            "Результирующий расчет",
            "--n_splits",
            "2",
        ],
    )

    train_main()

    assert (model_dir / "wind_xgb_model.joblib").exists()
    assert (model_dir / "cv_metrics.csv").exists()
    assert (model_dir / "feature_columns.json").exists()
    assert (model_dir / "feature_importance.csv").exists()

    with open(model_dir / "feature_columns.json", encoding="utf-8") as f:
        feature_cols = json.load(f)

    assert "hub_ws_80m" in feature_cols
    assert "hub_ws_80m_cube" in feature_cols
    assert "availability_ratio" in feature_cols