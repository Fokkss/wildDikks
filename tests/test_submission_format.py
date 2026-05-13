from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
def test_submission_format_file_exists_and_is_valid():
    path = Path("output/prediction.csv")

    assert path.exists(), "Run predict.py first; output/prediction.csv was not found"

    df = pd.read_csv(path)

    assert list(df.columns) == ["prediction"]
    assert len(df) > 0
    assert df["prediction"].notna().all()
    assert np.isfinite(df["prediction"]).all()
    assert (df["prediction"] >= 0).all()
    assert (df["prediction"] <= 90.09).all()
