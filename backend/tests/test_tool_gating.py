import pytest

from app import state
from app.main import _select_tools, _should_enable_tools


@pytest.fixture(autouse=True)
def isolate_state():
    """Each test gets a clean filename_to_file_id and never leaks into the next."""
    saved = dict(state.filename_to_file_id)
    state.filename_to_file_id.clear()
    yield
    state.filename_to_file_id.clear()
    state.filename_to_file_id.update(saved)


def test_disabled_when_no_tabular_files_ingested():
    assert _should_enable_tools("plot the average NAV") is False


def test_disabled_for_qualitative_question():
    state.filename_to_file_id["fund.csv"] = "abc"
    assert _should_enable_tools("what is the fund manager's name?") is False


def test_enabled_for_quantitative_keywords():
    state.filename_to_file_id["fund.csv"] = "abc"
    assert _should_enable_tools("plot the average NAV by year") is True
    assert _should_enable_tools("Top 5 funds by AUM") is True
    assert _should_enable_tools("compute the median return") is True


def test_enabled_for_analyse_substring_both_spellings():
    state.filename_to_file_id["fund.csv"] = "abc"
    assert _should_enable_tools("analyse the file") is True
    assert _should_enable_tools("Analyze NAV trends") is True
    assert _should_enable_tools("show me the analysis") is True


def test_keyword_match_is_case_insensitive():
    state.filename_to_file_id["fund.csv"] = "abc"
    assert _should_enable_tools("MEAN of NAV") is True


def test_enabled_for_query_phrases():
    state.filename_to_file_id["fund.csv"] = "abc"
    assert _should_enable_tools("list all debt funds") is True
    assert _should_enable_tools("show me funds with AUM > 100") is True
    assert _should_enable_tools("find all rows where category = Equity") is True
    assert _should_enable_tools("which fund has the highest return") is True
    assert _should_enable_tools("rows with NAV > 50") is True


def test_bare_list_or_show_does_not_trigger():
    """Multi-word query phrases are intentional — bare 'list' / 'show' appear
    in qualitative questions like 'show me the fund manager bio'."""
    state.filename_to_file_id["fund.csv"] = "abc"
    assert _should_enable_tools("show the manager bio") is False
    assert _should_enable_tools("what's on the list of holdings on page 3") is False


# ── _select_tools: explicit-mode override behavior ──

def test_chat_mode_returns_no_tools_even_with_quant_keywords():
    state.filename_to_file_id["fund.csv"] = "abc"
    assert _select_tools("max NAV", "chat") is None


def test_aggregate_mode_returns_only_compute_metric():
    state.filename_to_file_id["fund.csv"] = "abc"
    tools = _select_tools("anything at all", "aggregate")
    assert tools is not None
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "compute_metric"


def test_query_mode_returns_only_query_table():
    state.filename_to_file_id["fund.csv"] = "abc"
    tools = _select_tools("anything at all", "query")
    assert tools is not None
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "query_table"


def test_aggregate_mode_returns_none_without_tabular_files():
    """Forced aggregate mode still requires a tabular file to operate on."""
    tools = _select_tools("max NAV", "aggregate")
    assert tools is None


def test_query_mode_returns_none_without_tabular_files():
    tools = _select_tools("list all funds", "query")
    assert tools is None


def test_auto_mode_matches_legacy_should_enable_tools_behavior():
    state.filename_to_file_id["fund.csv"] = "abc"
    # Quant keyword → both tools exposed
    tools = _select_tools("plot the average NAV", "auto")
    assert tools is not None and len(tools) == 2
    # No matching keyword → no tools
    assert _select_tools("who is the fund manager", "auto") is None
