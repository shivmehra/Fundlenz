from app.rag.retriever import format_context


def test_empty_chunks_returns_placeholder():
    assert format_context([]) == "(no relevant context found)"


def test_includes_filename_and_page():
    chunks = [
        {"filename": "fund.pdf", "page": 3, "text": "expense ratio is 0.5%"},
        {"filename": "fund.pdf", "page": 7, "text": "AUM 100M"},
    ]
    out = format_context(chunks)
    assert "fund.pdf p.3" in out
    assert "fund.pdf p.7" in out
    assert "expense ratio is 0.5%" in out
    assert "AUM 100M" in out


def test_omits_page_when_missing():
    chunks = [{"filename": "summary.csv", "text": "col1, col2"}]
    out = format_context(chunks)
    assert "summary.csv" in out
    assert "p." not in out


def test_chunks_are_numbered():
    chunks = [
        {"filename": "a.pdf", "page": 1, "text": "first"},
        {"filename": "b.pdf", "page": 1, "text": "second"},
    ]
    out = format_context(chunks)
    assert "[1]" in out
    assert "[2]" in out
