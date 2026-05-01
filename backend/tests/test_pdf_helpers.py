from app.parsers.pdf import (
    _clean_table_rows,
    _detect_boilerplate,
    _is_valid_table,
    _strip_boilerplate,
    _table_to_markdown,
)


def test_empty_table_returns_empty_string():
    assert _table_to_markdown([]) == ""


def test_basic_table_renders_header_and_rows():
    table = [["Fund", "NAV"], ["A", "100"], ["B", "200"]]
    md = _table_to_markdown(table)
    assert "| Fund | NAV |" in md
    assert "| --- | --- |" in md
    assert "| A | 100 |" in md
    assert "| B | 200 |" in md


def test_table_strips_whitespace_and_handles_nones():
    table = [["  col1  ", None], [" v1 ", None]]
    md = _table_to_markdown(table)
    assert "| col1 |  |" in md
    assert "| v1 |  |" in md


def test_header_only_renders_two_lines():
    md = _table_to_markdown([["a", "b"]])
    lines = [l for l in md.strip().split("\n") if l]
    assert len(lines) == 2


def test_detect_boilerplate_finds_repeating_lines():
    pages = [
        "Confidential — do not distribute\nFund factsheet page 1 content",
        "Confidential — do not distribute\nFund factsheet page 2 content",
        "Confidential — do not distribute\nFund factsheet page 3 content",
    ]
    boilerplate = _detect_boilerplate(pages, min_repeat=3)
    assert "Confidential — do not distribute" in boilerplate
    assert "Fund factsheet page 1 content" not in boilerplate


def test_detect_boilerplate_short_doc_returns_empty():
    pages = ["page one", "page two"]
    assert _detect_boilerplate(pages, min_repeat=3) == set()


def test_detect_boilerplate_ignores_short_lines():
    pages = ["a\nReal content here", "a\nDifferent content", "a\nThird content"]
    boilerplate = _detect_boilerplate(pages, min_repeat=3)
    assert "a" not in boilerplate


def test_strip_boilerplate_removes_matching_lines():
    text = "Confidential\nReal answer here\nConfidential"
    out = _strip_boilerplate(text, {"Confidential"})
    assert "Confidential" not in out
    assert "Real answer here" in out


def test_strip_boilerplate_removes_page_numbers():
    text = "Page 3 of 12\nReal content\n5\n3 / 12\nMore content"
    out = _strip_boilerplate(text, set())
    assert "Page 3 of 12" not in out
    assert "Real content" in out
    assert "More content" in out
    # Bare numeric line is also a page-number pattern.
    lines = [l for l in out.splitlines() if l.strip()]
    assert "5" not in lines
    assert "3 / 12" not in out


def test_strip_boilerplate_with_no_matches_passes_through():
    text = "first line\nsecond line"
    assert _strip_boilerplate(text, set()) == text


# ── Table cleaning & validation ──

def test_clean_table_rows_strips_whitespace():
    rows = [["  Fund  ", " NAV "], [" Alpha ", "100"]]
    assert _clean_table_rows(rows) == [["Fund", "NAV"], ["Alpha", "100"]]


def test_clean_table_rows_folds_multi_line_cells():
    rows = [["Fund", "Notes"], ["Alpha", "First line\nSecond line"]]
    cleaned = _clean_table_rows(rows)
    assert cleaned[1][1] == "First line Second line"


def test_clean_table_rows_collapses_repeated_whitespace_in_cells():
    rows = [["a", "b\n  c   d"]]
    # Single-row will be rejected by validator, but cleaning should still
    # collapse internal whitespace.
    cleaned = _clean_table_rows(rows)
    assert cleaned[0][1] == "b c d"


def test_clean_table_rows_drops_empty_rows():
    rows = [["a", "b"], ["", ""], ["c", "d"]]
    assert _clean_table_rows(rows) == [["a", "b"], ["c", "d"]]


def test_clean_table_rows_drops_empty_columns():
    rows = [["a", "", "b"], ["c", "", "d"]]
    assert _clean_table_rows(rows) == [["a", "b"], ["c", "d"]]


def test_clean_table_rows_pads_ragged_rows():
    rows = [["a", "b", "c"], ["d", "e"]]
    assert _clean_table_rows(rows) == [["a", "b", "c"], ["d", "e", ""]]


def test_clean_table_rows_handles_none_cells():
    rows = [["a", None], [None, "b"]]
    assert _clean_table_rows(rows) == [["a", ""], ["", "b"]]


def test_clean_table_rows_empty_input_returns_empty_list():
    assert _clean_table_rows([]) == []


def test_is_valid_table_accepts_2x2_well_populated():
    assert _is_valid_table([["a", "b"], ["c", "d"]]) is True


def test_is_valid_table_rejects_single_row():
    assert _is_valid_table([["a", "b", "c"]]) is False


def test_is_valid_table_rejects_single_column():
    assert _is_valid_table([["a"], ["b"], ["c"]]) is False


def test_is_valid_table_rejects_mostly_empty():
    rows = [
        ["a", "", "", ""],
        ["", "", "", ""],
        ["", "", "", ""],
        ["", "", "", ""],
    ]
    assert _is_valid_table(rows) is False


def test_is_valid_table_accepts_a_few_empties():
    rows = [["a", "b"], ["c", ""], ["e", "f"]]
    assert _is_valid_table(rows) is True


def test_is_valid_table_rejects_empty_input():
    assert _is_valid_table([]) is False
