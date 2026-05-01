import pandas as pd
import pytest

from app.parsers.tabular import (
    _clean,
    enumerations,
    parse_tabular,
    row_chunks,
    synopsis,
)


def test_strips_column_names():
    df = pd.DataFrame({"  Fund  ": ["A"], "NAV ": [100]})
    cleaned, _ = _clean(df)
    assert list(cleaned.columns) == ["Fund", "NAV"]


def test_normalizes_null_tokens():
    df = pd.DataFrame({"x": ["1", "N/A", "-", "n/a", ""]})
    cleaned, _ = _clean(df)
    assert cleaned["x"].notna().sum() == 1


def test_drops_fully_empty_rows_and_columns():
    df = pd.DataFrame({
        "a": [1, None, 3, None],
        "b": [None, None, None, None],
        "c": [10, None, 30, None],
    })
    cleaned, report = _clean(df)
    assert "b" not in cleaned.columns
    assert len(cleaned) == 2
    assert report["cols_dropped"] == 1
    assert report["rows_dropped"] == 2


def test_currency_strings_coerce_to_floats():
    df = pd.DataFrame({"amount": ["$1,234.56", "$2,345.67", "$3,456.78"]})
    cleaned, report = _clean(df)
    assert "amount" in report["numeric_coerced"]
    assert cleaned["amount"].iloc[0] == pytest.approx(1234.56)
    assert cleaned["amount"].dtype.kind == "f"


def test_percent_strings_coerce_to_fractions():
    df = pd.DataFrame({"return": ["5.2%", "3.1%", "7.4%"]})
    cleaned, report = _clean(df)
    assert "return" in report["numeric_coerced"]
    assert cleaned["return"].iloc[0] == pytest.approx(0.052)


def test_skips_coercion_when_below_80pct_parse_rate():
    df = pd.DataFrame({"label": ["alpha", "beta", "gamma", "delta", "5"]})
    cleaned, report = _clean(df)
    assert "label" not in report["numeric_coerced"]


def test_date_coercion_only_with_name_hint():
    df = pd.DataFrame({
        "as of date": ["2023-01-01", "2023-02-01", "2023-03-01"],
        "label": ["2023-01-01", "2023-02-01", "2023-03-01"],
    })
    cleaned, report = _clean(df)
    assert "as of date" in report["date_coerced"]
    assert "label" not in report["date_coerced"]
    assert cleaned["as of date"].dtype.kind == "M"


def test_csv_returns_single_record_with_logical_name(tmp_path):
    csv = tmp_path / "fund.csv"
    csv.write_text(
        "fund,nav,as of date\n"
        "A,$1234.50,2023-01-01\n"
        "B,$2000.00,2023-02-01\n",
        encoding="utf-8",
    )
    records = parse_tabular(csv)
    assert len(records) == 1
    rec = records[0]
    assert rec["logical_name"] == "fund.csv"
    assert rec["df"]["nav"].dtype.kind == "f"
    assert "fund.csv" in rec["summary"]


def test_excel_single_sheet_uses_bare_filename(tmp_path):
    xlsx = tmp_path / "fund.xlsx"
    pd.DataFrame({"fund": ["A", "B"], "nav": [100, 200]}).to_excel(xlsx, index=False)
    records = parse_tabular(xlsx)
    assert len(records) == 1
    assert records[0]["logical_name"] == "fund.xlsx"


def test_excel_multi_sheet_produces_one_record_per_sheet(tmp_path):
    xlsx = tmp_path / "fund.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({"fund": ["A", "B"], "nav": [100, 200]}).to_excel(w, sheet_name="Holdings", index=False)
        pd.DataFrame({"month": ["Jan", "Feb"], "return": [0.01, 0.02]}).to_excel(w, sheet_name="Performance", index=False)
    records = parse_tabular(xlsx)
    assert len(records) == 2
    names = {r["logical_name"] for r in records}
    assert names == {"fund.xlsx :: Holdings", "fund.xlsx :: Performance"}


def test_excel_skips_empty_sheets(tmp_path):
    xlsx = tmp_path / "fund.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({"fund": ["A"], "nav": [100]}).to_excel(w, sheet_name="Real", index=False)
        pd.DataFrame().to_excel(w, sheet_name="Empty", index=False)
    records = parse_tabular(xlsx)
    assert len(records) == 1
    assert "Real" in records[0]["logical_name"]


def test_csv_header_sniffed_when_first_row_is_title(tmp_path):
    """If the real header is on row 1 (because row 0 is a title),
    parse_tabular should detect this via Unnamed-column ratio."""
    csv = tmp_path / "with_title.csv"
    csv.write_text(
        "Fund Performance Report,,,\n"
        "fund,nav,return\n"
        "A,100,0.05\n"
        "B,200,0.07\n",
        encoding="utf-8",
    )
    records = parse_tabular(csv)
    rec = records[0]
    assert "fund" in [c.lower() for c in rec["df"].columns]
    assert "nav" in [c.lower() for c in rec["df"].columns]


def test_summary_includes_tail_rows_for_long_files(tmp_path):
    csv = tmp_path / "big.csv"
    rows = "\n".join(f"row{i},{i}" for i in range(50))
    csv.write_text("name,n\n" + rows, encoding="utf-8")
    summary = parse_tabular(csv)[0]["summary"]
    assert "Last 5 rows" in summary
    assert "Middle 5 rows" in summary
    assert "row49" in summary  # last row visible
    assert "row0" in summary   # first row visible


def test_summary_includes_value_counts_for_text_columns(tmp_path):
    csv = tmp_path / "cats.csv"
    csv.write_text(
        "fund,nav\n"
        "Alpha,100\nAlpha,110\nBeta,200\nBeta,210\nGamma,150\n",
        encoding="utf-8",
    )
    summary = parse_tabular(csv)[0]["summary"]
    assert "Top values in text columns" in summary
    assert "Alpha=2" in summary
    assert "Beta=2" in summary


def test_summary_includes_date_ranges(tmp_path):
    csv = tmp_path / "dates.csv"
    csv.write_text(
        "as of date,nav\n"
        "2020-01-01,100\n"
        "2024-12-31,200\n",
        encoding="utf-8",
    )
    summary = parse_tabular(csv)[0]["summary"]
    assert "Date ranges" in summary
    assert "2020" in summary
    assert "2024" in summary


def test_synopsis_describes_columns_and_size():
    df = pd.DataFrame({
        "fund": ["A", "B", "A", "C"],
        "nav": [100, 200, 110, 150],
    })
    syn = synopsis("fund.csv", df)
    assert "fund.csv" in syn
    assert "4 rows" in syn
    assert "2 columns" in syn
    assert "fund" in syn
    assert "nav" in syn


def test_synopsis_includes_date_range_when_present():
    df = pd.DataFrame({
        "as of date": pd.to_datetime(["2020-01-01", "2024-12-31"]),
        "nav": [100, 200],
    })
    syn = synopsis("history.csv", df)
    assert "2020" in syn
    assert "2024" in syn


def test_row_chunks_one_per_row_with_column_value_pairs():
    df = pd.DataFrame({
        "fund": ["Alpha", "Beta"],
        "nav": [100, 200],
    })
    rows = row_chunks("fund.csv", df)
    assert len(rows) == 2
    assert "fund=Alpha" in rows[0]
    assert "nav=100" in rows[0]
    assert "fund.csv" in rows[0]
    assert "fund=Beta" in rows[1]
    assert "nav=200" in rows[1]


def test_row_chunks_skips_nan_values():
    df = pd.DataFrame({"fund": ["A"], "nav": [None], "aum": [500]})
    rows = row_chunks("fund.csv", df)
    assert "nav" not in rows[0]
    assert "fund=A" in rows[0]
    assert "aum=500" in rows[0]


def test_row_chunks_caps_at_max_rows():
    df = pd.DataFrame({"x": list(range(1000))})
    rows = row_chunks("big.csv", df, max_rows=50)
    assert len(rows) == 50


def test_enumerations_lists_distinct_values_for_bounded_cardinality():
    df = pd.DataFrame({
        "category": ["Equity", "Debt", "Equity", "Hybrid", "Debt"],
        "fund": ["A", "B", "C", "D", "E"],
        "nav": [100, 200, 300, 400, 500],  # numeric — should be skipped
    })
    enums = enumerations("fund.csv", df)
    by_col = "; ".join(enums)
    assert "category" in by_col
    assert "Equity" in by_col
    assert "Debt" in by_col
    assert "Hybrid" in by_col
    assert "fund" in by_col
    # Numeric column should not be enumerated.
    assert "nav" not in by_col


def test_enumerations_skips_high_cardinality_columns():
    df = pd.DataFrame({"id": [f"id_{i}" for i in range(600)]})
    enums = enumerations("big.csv", df, max_cardinality=500)
    assert enums == []


def test_enumerations_skips_constant_columns():
    df = pd.DataFrame({"flag": ["yes", "yes", "yes"]})
    enums = enumerations("constant.csv", df)
    assert enums == []
