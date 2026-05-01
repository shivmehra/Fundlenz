import pandas as pd
import pytest

from app.analysis.metrics import compute


@pytest.fixture
def df():
    return pd.DataFrame({
        "fund": ["A", "B", "C", "A", "B"],
        "nav": [100.0, 200.0, 150.0, 110.0, 210.0],
        "year": [2023, 2023, 2023, 2024, 2024],
    })


def test_mean_no_groupby(df):
    out = compute(df, "mean", "nav")
    assert out["value"].iloc[0] == pytest.approx(154.0)


def test_sum_with_groupby(df):
    out = compute(df, "sum", "nav", group_by="fund")
    by = dict(zip(out["fund"], out["nav"]))
    assert by == {"A": 210.0, "B": 410.0, "C": 150.0}


def test_top_n_default_returns_all_sorted(df):
    out = compute(df, "top_n", "nav")
    assert out["nav"].tolist() == [210.0, 200.0, 150.0, 110.0, 100.0]


def test_top_n_with_n(df):
    out = compute(df, "top_n", "nav", n=2)
    assert len(out) == 2
    assert out["nav"].tolist() == [210.0, 200.0]


def test_trend_requires_groupby(df):
    with pytest.raises(ValueError, match="trend"):
        compute(df, "trend", "nav")


def test_trend_groups_and_means(df):
    out = compute(df, "trend", "nav", group_by="year")
    by = dict(zip(out["year"], out["nav"]))
    assert by[2023] == pytest.approx(150.0)
    assert by[2024] == pytest.approx(160.0)


def test_unknown_column_raises(df):
    with pytest.raises(ValueError, match="not in dataframe"):
        compute(df, "mean", "missing")


def test_unknown_groupby_for_trend_raises(df):
    with pytest.raises(ValueError, match="not found"):
        compute(df, "trend", "nav", group_by="missing")
