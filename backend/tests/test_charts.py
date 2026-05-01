import pandas as pd

from app.analysis.charts import chart_spec


def test_simple_value_renders_indicator():
    df = pd.DataFrame({"metric": ["mean"], "value": [42.5]})
    spec = chart_spec(df, "mean", "nav", group_by=None)
    assert spec["data"][0]["type"] == "indicator"
    assert spec["data"][0]["value"] == 42.5


def test_groupby_renders_bar_chart():
    df = pd.DataFrame({"fund": ["A", "B"], "nav": [100, 200]})
    spec = chart_spec(df, "sum", "nav", group_by="fund")
    assert spec["data"][0]["type"] == "bar"
    assert spec["data"][0]["x"] == ["A", "B"]
    assert spec["data"][0]["y"] == [100, 200]


def test_trend_renders_scatter_lines():
    df = pd.DataFrame({"year": [2023, 2024], "nav": [150, 160]})
    spec = chart_spec(df, "trend", "nav", group_by="year")
    assert spec["data"][0]["type"] == "scatter"
    assert spec["data"][0]["mode"] == "lines+markers"
    assert spec["data"][0]["x"] == ["2023", "2024"]


def test_top_n_picks_first_non_numeric_column_as_label():
    # Regression test for the broken-x-axis fix.
    df = pd.DataFrame({"fund": ["A", "B", "C"], "nav": [200, 150, 100]})
    spec = chart_spec(df, "top_n", "nav", group_by=None)
    assert spec["data"][0]["type"] == "bar"
    assert spec["data"][0]["x"] == ["A", "B", "C"]


def test_top_n_falls_back_to_index_when_all_numeric():
    df = pd.DataFrame({"nav": [200, 150, 100]})
    spec = chart_spec(df, "top_n", "nav", group_by=None)
    assert spec["data"][0]["x"] == ["0", "1", "2"]


def test_top_n_uses_explicit_groupby_when_present():
    df = pd.DataFrame({
        "fund": ["A", "B", "C"],
        "category": ["x", "y", "z"],
        "nav": [200, 150, 100],
    })
    spec = chart_spec(df, "top_n", "nav", group_by="category")
    assert spec["data"][0]["x"] == ["x", "y", "z"]
