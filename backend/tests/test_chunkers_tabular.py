import pandas as pd

from app.chunkers.entity_chunker import build_entity_chunks
from app.chunkers.tabular_chunker import build_tabular_chunks, pick_name_column
from app.id.canonical import AliasMap


def _fund_df():
    return pd.DataFrame(
        {
            "Scheme Name": ["HDFC Top 100 Fund", "Axis Bluechip Fund", "ICICI Pru Bluechip"],
            "Benchmark": ["BSE 100 TRI", "BSE 100 TRI", "Nifty 100 TRI"],
            "NAV Regular": [1099.09, 57.92, 107.46],
            "Daily AUM (Cr.)": [38382.26, 30574.61, 76195.47],
        }
    )


def test_pick_name_column_picks_high_cardinality_text_col():
    df = _fund_df()
    assert pick_name_column(df) == "Scheme Name"


def test_pick_name_column_returns_none_when_no_unique_text():
    df = pd.DataFrame({"Category": ["A", "A", "A"], "x": [1, 2, 3]})
    # cardinality 1, fails distinct >= 2
    assert pick_name_column(df) is None


def test_row_chunks_carry_canonical_id(tmp_path):
    df = _fund_df()
    aliases = AliasMap(tmp_path / "aliases.json")
    aliases.load()
    chunks, name_col = build_tabular_chunks(
        df, file="fund.xlsx", file_id="fid", sheet=None, aliases=aliases
    )
    assert name_col == "Scheme Name"
    row_chunks = [m for m, _ in chunks if m["chunk_type"] == "row"]
    assert len(row_chunks) == 3
    assert row_chunks[0]["canonical_id"] == "hdfc top 100"
    assert row_chunks[1]["canonical_id"] == "axis bluechip"


def test_synopsis_and_tabular_summary_present(tmp_path):
    df = _fund_df()
    aliases = AliasMap(tmp_path / "aliases.json")
    aliases.load()
    chunks, _ = build_tabular_chunks(
        df, file="fund.xlsx", file_id="fid", sheet=None, aliases=aliases
    )
    types = [m["chunk_type"] for m, _ in chunks]
    assert "synopsis" in types
    assert "tabular_summary" in types


def test_enumeration_completeness(tmp_path):
    # Benchmark column has 2 distinct values with cardinality 2 -> enumeration emitted.
    df = _fund_df()
    aliases = AliasMap(tmp_path / "aliases.json")
    aliases.load()
    chunks, _ = build_tabular_chunks(
        df, file="fund.xlsx", file_id="fid", sheet=None, aliases=aliases
    )
    enum_chunks = [body for m, body in chunks if m["chunk_type"] == "enumeration"]
    benchmark_enum = next(b for b in enum_chunks if "Benchmark" in b)
    assert "BSE 100 TRI" in benchmark_enum
    assert "Nifty 100 TRI" in benchmark_enum


def test_entity_chunks_list_aliases(tmp_path):
    df = _fund_df()
    aliases = AliasMap(tmp_path / "aliases.json")
    aliases.load()
    # Pre-register so build_entity_chunks sees the alias list.
    for name in df["Scheme Name"]:
        aliases.register(str(name))

    entity_chunks = build_entity_chunks(
        df, file="fund.xlsx", file_id="fid", sheet=None, name_col="Scheme Name", aliases=aliases
    )
    assert len(entity_chunks) == 3
    canon_ids = {m["canonical_id"] for m, _ in entity_chunks}
    assert canon_ids == {"hdfc top 100", "axis bluechip", "icici pru bluechip"}
    # Each entity body should mention its own alias.
    for m, body in entity_chunks:
        if m["canonical_id"] == "hdfc top 100":
            assert "HDFC Top 100 Fund" in body
            break
    else:
        raise AssertionError("hdfc top 100 entity chunk not found")


def test_row_chunks_have_required_metadata(tmp_path):
    df = _fund_df()
    aliases = AliasMap(tmp_path / "aliases.json")
    aliases.load()
    chunks, _ = build_tabular_chunks(
        df, file="fund.xlsx", file_id="fid", sheet=None, aliases=aliases
    )
    row_chunks = [m for m, _ in chunks if m["chunk_type"] == "row"]
    for m in row_chunks:
        # 8 mandatory fields all present
        for k in ("chunk_id", "file", "sheet", "row_number", "chunk_type",
                  "canonical_id", "ingestion_time", "version"):
            assert k in m
        assert m["version"] == 2
