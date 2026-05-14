import joblib

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
            "--cv",
        ],
    )

    train_main()

    assert (model_dir / "wind_xgb_model.joblib").exists()
    assert (model_dir / "importance.csv").exists()

    artifact = joblib.load(model_dir / "wind_xgb_model.joblib")

    assert "model" in artifact
    assert "imputer" in artifact
    assert "feature_cols" in artifact
    assert "capacity_mw" in artifact
    assert len(artifact["feature_cols"]) > 0
