from app.index.inverted import InvertedIndex


def test_lookup_id(tmp_path):
    idx = InvertedIndex(tmp_path / "inv.json")
    idx.load_or_init()
    idx.add_id("hdfc top 100", "chunk-1")
    idx.add_id("hdfc top 100", "chunk-2")
    idx.add_id("axis bluechip", "chunk-3")

    assert idx.lookup_id("hdfc top 100") == ["chunk-1", "chunk-2"]
    assert idx.lookup_id("axis bluechip") == ["chunk-3"]
    assert idx.lookup_id("missing") == []


def test_lookup_cell_is_normalized(tmp_path):
    idx = InvertedIndex(tmp_path / "inv.json")
    idx.load_or_init()
    idx.add_cell("Scheme Name", "HDFC Top 100 Fund", "chunk-1")

    # Same canonical-form lookup should hit regardless of case / "Fund" suffix.
    assert idx.lookup_cell("scheme name", "HDFC Top 100") == ["chunk-1"]
    assert idx.lookup_cell("SCHEME NAME", "hdfc top 100 fund") == ["chunk-1"]


def test_lookup_enum(tmp_path):
    idx = InvertedIndex(tmp_path / "inv.json")
    idx.load_or_init()
    idx.add_enum("Category", "chunk-enum-cat")
    idx.add_enum("Benchmark", "chunk-enum-bench")

    assert idx.lookup_enum("category") == ["chunk-enum-cat"]
    assert idx.lookup_enum("CATEGORY") == ["chunk-enum-cat"]
    assert idx.lookup_enum("benchmark") == ["chunk-enum-bench"]


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "inv.json"
    a = InvertedIndex(p)
    a.load_or_init()
    a.add_id("x", "c1")
    a.add_cell("col", "Val", "c2")
    a.add_enum("col", "c3")
    a.save()

    b = InvertedIndex(p)
    b.load_or_init()
    assert b.lookup_id("x") == ["c1"]
    assert b.lookup_cell("col", "val") == ["c2"]
    assert b.lookup_enum("col") == ["c3"]


def test_remove_chunks(tmp_path):
    idx = InvertedIndex(tmp_path / "inv.json")
    idx.load_or_init()
    idx.add_id("x", "c1")
    idx.add_id("x", "c2")
    idx.add_cell("col", "v", "c1")

    idx.remove_chunks({"c1"})
    assert idx.lookup_id("x") == ["c2"]
    # Cell key whose only entry was c1 should be cleared entirely.
    assert idx.lookup_cell("col", "v") == []


def test_total_postings(tmp_path):
    idx = InvertedIndex(tmp_path / "inv.json")
    idx.load_or_init()
    idx.add_id("x", "c1")
    idx.add_id("y", "c2")
    idx.add_id("y", "c3")
    assert idx.total_postings() == 3
