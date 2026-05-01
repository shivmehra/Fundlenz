import pandas as pd
import pytest

from app.analysis.query import query_table


@pytest.fixture
def df():
    return pd.DataFrame({
        "fund": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "category": ["Equity", "Debt", "Equity", "Hybrid", "Equity"],
        "aum": [100.0, 200.0, 50.0, 300.0, 150.0],
        "return": [0.10, 0.05, 0.12, 0.08, 0.15],
    })


def test_no_filters_returns_first_n_rows(df):
    out = query_table(df, limit=3)
    assert len(out) == 3
    assert out["fund"].tolist() == ["Alpha", "Beta", "Gamma"]


def test_equality_filter(df):
    out = query_table(df, filters=[{"column": "category", "op": "==", "value": "Equity"}])
    assert len(out) == 3
    assert set(out["fund"]) == {"Alpha", "Gamma", "Epsilon"}


def test_inequality_filter(df):
    out = query_table(df, filters=[{"column": "category", "op": "!=", "value": "Equity"}])
    assert len(out) == 2
    assert set(out["fund"]) == {"Beta", "Delta"}


def test_greater_than_filter(df):
    out = query_table(df, filters=[{"column": "aum", "op": ">", "value": 100}])
    assert len(out) == 3
    assert set(out["fund"]) == {"Beta", "Delta", "Epsilon"}


def test_greater_than_coerces_string_value(df):
    """LLMs often pass numbers as strings — handler must coerce."""
    out = query_table(df, filters=[{"column": "aum", "op": ">", "value": "100"}])
    assert len(out) == 3


def test_contains_filter_is_case_insensitive(df):
    out = query_table(df, filters=[{"column": "fund", "op": "contains", "value": "alpha"}])
    assert len(out) == 1
    assert out["fund"].iloc[0] == "Alpha"


def test_in_filter_with_list(df):
    out = query_table(df, filters=[{"column": "fund", "op": "in", "value": ["Alpha", "Beta"]}])
    assert len(out) == 2
    assert set(out["fund"]) == {"Alpha", "Beta"}


def test_in_filter_with_scalar_treated_as_single_element(df):
    out = query_table(df, filters=[{"column": "fund", "op": "in", "value": "Alpha"}])
    assert len(out) == 1


def test_multiple_filters_apply_in_sequence(df):
    out = query_table(df, filters=[
        {"column": "category", "op": "==", "value": "Equity"},
        {"column": "aum", "op": ">", "value": 100},
    ])
    assert set(out["fund"]) == {"Epsilon"}


def test_sort_descending(df):
    out = query_table(df, sort_by="aum", sort_desc=True)
    assert out["fund"].tolist() == ["Delta", "Beta", "Epsilon", "Alpha", "Gamma"]


def test_sort_ascending(df):
    out = query_table(df, sort_by="aum", sort_desc=False)
    assert out["fund"].tolist() == ["Gamma", "Alpha", "Epsilon", "Beta", "Delta"]


def test_select_columns_subsets(df):
    out = query_table(df, select_columns=["fund", "aum"])
    assert list(out.columns) == ["fund", "aum"]


def test_select_columns_ignores_unknown(df):
    out = query_table(df, select_columns=["fund", "nonexistent"])
    assert list(out.columns) == ["fund"]


def test_select_columns_empty_falls_back_to_all(df):
    out = query_table(df, select_columns=[])
    assert list(out.columns) == ["fund", "category", "aum", "return"]


def test_unknown_filter_column_raises(df):
    with pytest.raises(ValueError, match="not found"):
        query_table(df, filters=[{"column": "missing", "op": "==", "value": "x"}])


def test_unknown_sort_column_raises(df):
    with pytest.raises(ValueError, match="Sort column"):
        query_table(df, sort_by="missing")


def test_unknown_filter_op_raises(df):
    with pytest.raises(ValueError, match="Unknown filter op"):
        query_table(df, filters=[{"column": "fund", "op": "regex", "value": "."}])


def test_filter_then_sort_then_select(df):
    out = query_table(
        df,
        filters=[{"column": "category", "op": "==", "value": "Equity"}],
        sort_by="aum",
        sort_desc=True,
        select_columns=["fund", "aum"],
        limit=2,
    )
    assert list(out.columns) == ["fund", "aum"]
    assert out["fund"].tolist() == ["Epsilon", "Alpha"]


def test_empty_result_returns_empty_df(df):
    out = query_table(df, filters=[{"column": "fund", "op": "==", "value": "Nope"}])
    assert out.empty


def test_limit_caps_output(df):
    out = query_table(df, limit=2)
    assert len(out) == 2


def test_negative_limit_falls_back_to_default(df):
    out = query_table(df, limit=-1)
    assert len(out) == 5
