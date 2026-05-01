from app.text import normalize


def test_empty_input_returns_empty():
    assert normalize("") == ""


def test_smart_quotes_become_ascii():
    assert normalize("“hello” ‘world’") == '"hello" \'world\''


def test_em_and_en_dashes_become_hyphen():
    assert normalize("a—b–c") == "a-b-c"


def test_non_breaking_space_becomes_regular_space():
    assert normalize("a b") == "a b"


def test_zero_width_and_bom_stripped():
    assert normalize("a​b﻿c") == "abc"


def test_dehyphenates_line_wrapped_words():
    assert normalize("expense-\nratio") == "expenseratio"
    assert normalize("co-\nfounder") == "cofounder"  # accepted trade-off


def test_preserves_meaningful_hyphens_within_a_line():
    assert normalize("co-founder") == "co-founder"


def test_collapses_three_or_more_newlines_to_paragraph_break():
    assert normalize("a\n\n\n\nb") == "a\n\nb"


def test_strips_trailing_whitespace_before_newlines():
    assert normalize("hello   \nworld") == "hello\nworld"


def test_nfkc_compatibility_normalization():
    # Fullwidth digits → ASCII via NFKC
    assert normalize("１２３") == "123"


def test_preserves_paragraph_breaks():
    assert normalize("paragraph one\n\nparagraph two") == "paragraph one\n\nparagraph two"
