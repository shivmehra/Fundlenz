import pandas as pd

from app.analysis.column_match import resolve_column


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Scheme Name": ["A", "B"],
            "NAV Regular": [100.0, 200.0],
            "NAV Direct": [101.0, 201.0],
            "Daily AUM (Cr.)": [5000.0, 60000.0],
            "1-Year Return": [10.5, 15.2],
        }
    )


def test_aum_phrase_matches_daily_aum_column():
    col, score = resolve_column("Funds with AUM greater than 50000", _df())
    assert col == "Daily AUM (Cr.)"
    assert score > 0


def test_nav_regular_beats_nav_direct_when_regular_in_query():
    col, _ = resolve_column("funds where NAV regular is less than 100", _df())
    assert col == "NAV Regular"


def test_no_match_returns_none():
    col, score = resolve_column("greater than 50", _df())
    # "greater" / "than" / "50" share no tokens with any numeric column.
    assert col is None
    assert score == 0.0


def test_empty_query_returns_none():
    col, score = resolve_column("", _df())
    assert col is None
    assert score == 0.0


def test_only_considers_numeric_columns():
    # Even though "scheme name" is a column, it's text so it can't be a
    # threshold target — should never be returned.
    col, _ = resolve_column("show me funds where scheme name has stuff", _df())
    assert col != "Scheme Name"


def test_punctuation_in_column_name_doesnt_break_matching():
    df = pd.DataFrame({"3-Year Return (%)": [1.0, 2.0]})
    col, _ = resolve_column("3-year return below 5", df)
    assert col == "3-Year Return (%)"
