from app.retrieval.orchestrator import format_context_v2


def test_empty_chunks_returns_placeholder():
    assert format_context_v2([]) == "(no relevant context found)"


def test_includes_filename_and_page():
    chunks = [
        {"file": "fund.pdf", "chunk_type": "text", "page": 3, "text": "expense ratio is 0.5%"},
        {"file": "fund.pdf", "chunk_type": "text", "page": 7, "text": "AUM 100M"},
    ]
    out = format_context_v2(chunks)
    assert "fund.pdf p.3" in out
    assert "fund.pdf p.7" in out
    assert "expense ratio is 0.5%" in out
    assert "AUM 100M" in out


def test_omits_page_when_missing():
    chunks = [{"file": "summary.csv", "chunk_type": "tabular_summary", "text": "col1, col2"}]
    out = format_context_v2(chunks)
    assert "summary.csv" in out
    assert "p." not in out


def test_chunks_are_numbered():
    chunks = [
        {"file": "a.pdf", "chunk_type": "text", "page": 1, "text": "first"},
        {"file": "b.pdf", "chunk_type": "text", "page": 1, "text": "second"},
    ]
    out = format_context_v2(chunks)
    assert "[1]" in out
    assert "[2]" in out
