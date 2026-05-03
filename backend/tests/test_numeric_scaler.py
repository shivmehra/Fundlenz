import numpy as np
import pandas as pd

from app.rag.numeric_scaler import NumericScaler


def test_fit_transform_zscore_basic():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    s = NumericScaler()
    s.fit(df, ["x"])
    # mean=3, std=sqrt(2) (population std)
    assert s.mean["x"] == 3.0
    assert s.std["x"] > 0
    z_first = s.transform_row(df.iloc[0])[0]
    z_last = s.transform_row(df.iloc[-1])[0]
    assert z_first < 0 and z_last > 0
    assert z_first == -z_last  # symmetry


def test_constant_column_does_not_divide_by_zero():
    df = pd.DataFrame({"x": [7.0, 7.0, 7.0]})
    s = NumericScaler()
    s.fit(df, ["x"])
    assert s.std["x"] == 1.0  # forced floor
    vec = s.transform_row(df.iloc[0])
    # value - mean = 0, divided by 1 → 0; not NaN, not inf.
    assert vec[0] == 0.0


def test_nan_input_becomes_zero_post_scale():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    s = NumericScaler()
    s.fit(df, ["x"])
    vec = s.transform_row(df.iloc[2])
    assert vec[0] == 0.0


def test_save_load_roundtrip(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    s1 = NumericScaler()
    s1.fit(df, ["x", "y"])
    p = tmp_path / "scaler.json"
    s1.save(p)

    s2 = NumericScaler()
    s2.load(p)
    assert s2.columns == ["x", "y"]
    for c in ["x", "y"]:
        assert abs(s2.mean[c] - s1.mean[c]) < 1e-9
        assert abs(s2.std[c] - s1.std[c]) < 1e-9


def test_transform_returns_float32():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    s = NumericScaler()
    s.fit(df, ["x"])
    vec = s.transform_row(df.iloc[0])
    assert vec.dtype == np.float32
