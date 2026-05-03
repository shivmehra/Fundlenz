"""Tests for the numeric-threshold deterministic bypass in main.py — the
`_try_numeric_threshold_bypass` helper that runs pandas filtering instead of
relying on the LLM to call query_table correctly."""
import pandas as pd
import pytest

from app import state
from app.main import _confidence_for_llm_path, _try_numeric_threshold_bypass


@pytest.fixture
def loaded_funds(monkeypatch):
    """Stub state.dataframes_by_file_id + filename_to_file_id with a small,
    deterministic table. Cleans up after the test."""
    df = pd.DataFrame(
        {
            "Scheme Name": [
                "HDFC Top 100",
                "Axis Bluechip",
                "ICICI Pru Bluechip",
                "Samco Large Cap",
            ],
            "NAV Regular": [1099.09, 57.92, 107.46, 9.27],
            "Daily AUM (Cr.)": [38382.26, 30574.61, 76195.47, 5000.0],
        }
    )
    monkeypatch.setitem(state.filename_to_file_id, "test_funds.xlsx", "fid-1")
    monkeypatch.setitem(state.dataframes_by_file_id, "fid-1", df)
    yield df
    state.filename_to_file_id.pop("test_funds.xlsx", None)
    state.dataframes_by_file_id.pop("fid-1", None)


def test_bypass_filters_funds_above_aum_threshold(loaded_funds):
    out = _try_numeric_threshold_bypass("Funds with AUM greater than 50000")
    assert out is not None
    answer, filename, column, op, value, n_rows, n_files = out
    assert filename == "test_funds.xlsx"
    assert column == "Daily AUM (Cr.)"
    assert op == ">"
    assert value == 50000.0
    assert n_files == 1
    # Only ICICI Pru Bluechip (76195.47) qualifies.
    assert n_rows == 1
    assert "ICICI Pru Bluechip" in answer
    # The redundant "**Verified:**" prefix has been removed (the badge already
    # conveys verification status); answer should not start with it.
    assert not answer.lstrip().startswith("**Verified:**")


def test_bypass_filters_funds_below_nav_threshold(loaded_funds):
    """The bug from the user's chat: LLM said no funds had NAV < 100, while
    Samco (9.27), Axis (57.92), Sundaram-shaped values are all below 100. The
    bypass must surface them."""
    out = _try_numeric_threshold_bypass(
        "Show me funds where NAV regular is less than 100"
    )
    assert out is not None
    answer, _, column, op, value, n_rows, _ = out
    assert column == "NAV Regular"
    assert op == "<"
    assert value == 100.0
    # Axis (57.92) and Samco (9.27) — 2 of 4.
    assert n_rows == 2
    assert "Samco Large Cap" in answer
    assert "Axis Bluechip" in answer


def test_bypass_returns_none_for_non_threshold_query(loaded_funds):
    assert _try_numeric_threshold_bypass("What is the NAV of HDFC Top 100?") is None


def test_bypass_returns_none_when_no_tabular_file_loaded():
    # No fixture — state is empty.
    assert _try_numeric_threshold_bypass("Funds with AUM > 50000") is None


def test_bypass_returns_none_when_column_unresolvable(loaded_funds):
    # "Sharpe ratio" doesn't exist in the test df — bypass falls through.
    assert _try_numeric_threshold_bypass("funds with sharpe ratio above 1.5") is None


def test_bypass_handles_zero_matches(loaded_funds):
    out = _try_numeric_threshold_bypass("Funds with AUM greater than 999999")
    assert out is not None
    answer, _, _, _, _, n_rows, n_files = out
    assert n_rows == 0
    assert n_files == 1
    # The "no matches" message must reference the file that was actually
    # searched so the user can see the bypass tried something.
    assert "test_funds.xlsx" in answer


def test_bypass_aggregates_across_multiple_files(monkeypatch):
    """Multiple uploads should all be searched: the user's bug was that only
    one file was checked for "1-year return above 10". When two files share a
    matching column, both must contribute to the result."""
    df_a = pd.DataFrame(
        {
            "Scheme Name": ["Alpha Fund", "Beta Fund"],
            "Return 1 Year (%) Regular": [4.2, 7.5],
        }
    )
    df_b = pd.DataFrame(
        {
            "Scheme Name": ["Gamma Fund", "Delta Fund"],
            "Return 1 Year (%) Regular": [12.3, 15.8],
        }
    )
    monkeypatch.setitem(state.filename_to_file_id, "fileA.xlsx", "fid-A")
    monkeypatch.setitem(state.filename_to_file_id, "fileB.xlsx", "fid-B")
    monkeypatch.setitem(state.dataframes_by_file_id, "fid-A", df_a)
    monkeypatch.setitem(state.dataframes_by_file_id, "fid-B", df_b)
    try:
        out = _try_numeric_threshold_bypass(
            "Schemes with 1-year return above 10"
        )
        assert out is not None
        answer, _, column, _, _, n_rows, n_files = out
        assert column == "Return 1 Year (%) Regular"
        # Both Gamma (12.3) and Delta (15.8) live in fileB.
        assert n_rows == 2
        # fileA contributed zero rows — only fileB shows up in `n_files`.
        assert n_files == 1
        assert "Gamma Fund" in answer
        assert "Delta Fund" in answer
    finally:
        state.filename_to_file_id.pop("fileA.xlsx", None)
        state.filename_to_file_id.pop("fileB.xlsx", None)
        state.dataframes_by_file_id.pop("fid-A", None)
        state.dataframes_by_file_id.pop("fid-B", None)


def test_bypass_zero_matches_lists_all_searched_files(monkeypatch):
    """The "no rows" message must enumerate every file that was searched —
    otherwise the user can't tell whether their other uploads were considered."""
    df_a = pd.DataFrame({"Scheme Name": ["A"], "Daily AUM (Cr.)": [100.0]})
    df_b = pd.DataFrame({"Scheme Name": ["B"], "Daily AUM (Cr.)": [200.0]})
    monkeypatch.setitem(state.filename_to_file_id, "fileA.xlsx", "fid-A")
    monkeypatch.setitem(state.filename_to_file_id, "fileB.xlsx", "fid-B")
    monkeypatch.setitem(state.dataframes_by_file_id, "fid-A", df_a)
    monkeypatch.setitem(state.dataframes_by_file_id, "fid-B", df_b)
    try:
        out = _try_numeric_threshold_bypass("Funds with AUM greater than 999999")
        assert out is not None
        answer, _, _, _, _, n_rows, n_files = out
        assert n_rows == 0
        assert n_files == 2
        assert "fileA.xlsx" in answer
        assert "fileB.xlsx" in answer
    finally:
        state.filename_to_file_id.pop("fileA.xlsx", None)
        state.filename_to_file_id.pop("fileB.xlsx", None)
        state.dataframes_by_file_id.pop("fid-A", None)
        state.dataframes_by_file_id.pop("fid-B", None)


def test_bypass_multi_file_emits_source_file_column(monkeypatch):
    """When matches come from 2+ files, the result table must tag each row
    with its source file so the user knows where each scheme came from."""
    df_a = pd.DataFrame(
        {"Scheme Name": ["A1"], "Return 1 Year (%) Regular": [12.0]}
    )
    df_b = pd.DataFrame(
        {"Scheme Name": ["B1"], "Return 1 Year (%) Regular": [15.0]}
    )
    monkeypatch.setitem(state.filename_to_file_id, "fileA.xlsx", "fid-A")
    monkeypatch.setitem(state.filename_to_file_id, "fileB.xlsx", "fid-B")
    monkeypatch.setitem(state.dataframes_by_file_id, "fid-A", df_a)
    monkeypatch.setitem(state.dataframes_by_file_id, "fid-B", df_b)
    try:
        out = _try_numeric_threshold_bypass(
            "Schemes with 1-year return above 10"
        )
        assert out is not None
        answer, _, _, _, _, n_rows, n_files = out
        assert n_rows == 2
        assert n_files == 2
        assert "Source File" in answer
        assert "fileA.xlsx" in answer
        assert "fileB.xlsx" in answer
    finally:
        state.filename_to_file_id.pop("fileA.xlsx", None)
        state.filename_to_file_id.pop("fileB.xlsx", None)
        state.dataframes_by_file_id.pop("fid-A", None)
        state.dataframes_by_file_id.pop("fid-B", None)


def test_confidence_deterministic_when_tool_executed():
    conf = _confidence_for_llm_path([], tool_executed=True)
    assert conf["tier"] == "deterministic"
    assert conf["value"] == 1.0


def test_confidence_deterministic_when_all_top_sources_are_exact():
    chunks = [
        {"_score_breakdown": {"exact_id": True}, "file": "f", "chunk_type": "row"},
        {"_score_breakdown": {"exact_id": True}, "file": "f", "chunk_type": "entity"},
        {"_score_breakdown": {"exact_id": True}, "file": "f", "chunk_type": "numeric_vector"},
    ]
    conf = _confidence_for_llm_path(chunks, tool_executed=False)
    assert conf["tier"] == "deterministic"


def test_confidence_grounded_when_avg_score_high():
    chunks = [
        {"_score_breakdown": {"text_sim": 0.9}, "file": "f", "chunk_type": "row"},
        {"_score_breakdown": {"text_sim": 0.88}, "file": "f", "chunk_type": "row"},
        {"_score_breakdown": {"text_sim": 0.86}, "file": "f", "chunk_type": "row"},
    ]
    conf = _confidence_for_llm_path(chunks, tool_executed=False)
    assert conf["tier"] == "grounded"


def test_confidence_semantic_when_avg_score_low():
    chunks = [
        {"_score_breakdown": {"text_sim": 0.5}, "file": "f", "chunk_type": "row"},
        {"_score_breakdown": {"text_sim": 0.4}, "file": "f", "chunk_type": "row"},
    ]
    conf = _confidence_for_llm_path(chunks, tool_executed=False)
    assert conf["tier"] == "semantic"


def test_confidence_handles_no_chunks():
    conf = _confidence_for_llm_path([], tool_executed=False)
    assert conf["tier"] == "semantic"
    assert conf["value"] == 0.0
