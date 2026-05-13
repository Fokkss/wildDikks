import pandas as pd
from for_plot.plotter import list_completer_factory


def test_list_completer():
    # auto adding of columns without UI
    valid_cols = ['wind_speed', 'target', 'timestamp']
    completer = list_completer_factory(valid_cols)

    result = completer('wind', 0)
    assert result == 'wind_speed'

def test_data_sorting_logic():
    # sort doesn't break data
    data = {'x': [3, 1, 2], 'y': [10, 20, 30]}
    df = pd.DataFrame(data)
    sorted_df = df.sort_values(by='x')

    assert sorted_df['x'].tolist() == [1, 2, 3]
    assert sorted_df['y'].tolist() == [20, 30, 10]

def test_csv_loading(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("a,b\n1,2\n3,4")

    df = pd.read_csv(p)
    assert df.shape == (2, 2)
    assert list(df.columns) == ['a', 'b']