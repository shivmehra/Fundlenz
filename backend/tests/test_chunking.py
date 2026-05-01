from app.rag.embedder import chunk_text


def test_empty_text_returns_empty_list():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   ", 100, 10) == []


def test_short_text_returns_single_chunk():
    assert chunk_text("hello world", 100, 10) == ["hello world"]


def test_long_text_chunks_with_overlap():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    chunks = chunk_text(text, 5, 2)

    # max=5, overlap=2 → step=3 → starts at 0,3,6,9,12,15,18 → 7 chunks
    assert len(chunks) == 7
    assert chunks[0] == "w0 w1 w2 w3 w4"
    assert chunks[1] == "w3 w4 w5 w6 w7"

    # Every adjacent pair shares its overlap region.
    for i in range(len(chunks) - 1):
        prev_tail = chunks[i].split()[-2:]
        next_head = chunks[i + 1].split()[:2]
        assert prev_tail == next_head


def test_step_clamped_to_one_when_overlap_exceeds_max():
    # max=2, overlap=2 → step would be 0 → must clamp to 1 to avoid infinite loop.
    chunks = chunk_text("a b c d", 2, 2)
    assert all(chunks)
    assert len(chunks) <= 4
