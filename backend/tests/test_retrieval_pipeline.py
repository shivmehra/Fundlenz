"""Integration tests: build a small in-memory CompositeIndex via the real
ingest path and assert that the deterministic exact-match layer wins over
plain ANN ranking."""
import pandas as pd
import pytest

from app import state
from app.config import settings
from app.id.canonical import normalize_name
from app.ingest import ingest_file
from app.retrieval.orchestrator import retrieve_v2
from app.retrieval.router import classify_intent


@pytest.fixture
def fresh_index(tmp_path, monkeypatch):
    """Point all index/upload paths at tmp_path and re-instantiate the global
    composite. Preserves prior state for restoration."""
    # Override config paths
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    # Re-create the composite at the new location
    from app.index.composite import CompositeIndex
    old_composite = state.composite
    old_dataframes = dict(state.dataframes_by_file_id)
    old_filename_to_file_id = dict(state.filename_to_file_id)
    old_documents = dict(state.documents)

    state.composite = CompositeIndex(
        tmp_path / "indexes", text_dim=384, enable_numeric_ann=False
    )
    state.composite.load_or_init()
    state.dataframes_by_file_id.clear()
    state.filename_to_file_id.clear()
    state.documents.clear()

    yield

    state.composite = old_composite
    state.dataframes_by_file_id.clear()
    state.dataframes_by_file_id.update(old_dataframes)
    state.filename_to_file_id.clear()
    state.filename_to_file_id.update(old_filename_to_file_id)
    state.documents.clear()
    state.documents.update(old_documents)


def _write_fund_xlsx(path) -> None:
    df = pd.DataFrame(
        {
            "Scheme Name": [
                "HDFC Top 100 Fund",
                "Axis Bluechip Fund",
                "ICICI Pru Bluechip",
                "Kotak Bluechip Fund",
            ],
            "Benchmark": ["BSE 100 TRI", "BSE 100 TRI", "Nifty 100 TRI", "Nifty 100 TRI"],
            "NAV Regular": [1099.09, 57.92, 107.46, 558.70],
            "Daily AUM (Cr.)": [38382.26, 30574.61, 76195.47, 5000.0],
        }
    )
    df.to_excel(path, index=False)


@pytest.mark.usefixtures("fresh_index")
def test_point_lookup_top1_is_target_entity(tmp_path):
    fund_xlsx = tmp_path / "fund.xlsx"
    _write_fund_xlsx(fund_xlsx)
    ingest_file(fund_xlsx, "fund.xlsx")

    chunks = retrieve_v2("What is the NAV regular of HDFC Top 100 Fund?", state.composite, k=5)
    assert chunks, "expected at least one retrieval result"
    top = chunks[0]
    # Either the row chunk or the entity chunk for this fund should win — both
    # carry the right canonical_id and that's what determinism guarantees.
    assert top["canonical_id"] == normalize_name("HDFC Top 100 Fund")
    assert top["chunk_type"] in ("row", "entity")


@pytest.mark.usefixtures("fresh_index")
def test_list_distinct_surfaces_enumeration_chunk(tmp_path):
    fund_xlsx = tmp_path / "fund.xlsx"
    _write_fund_xlsx(fund_xlsx)
    ingest_file(fund_xlsx, "fund.xlsx")

    chunks = retrieve_v2("List all benchmarks", state.composite, k=5)
    types = [c["chunk_type"] for c in chunks]
    assert "enumeration" in types
    enum_chunks = [c for c in chunks if c["chunk_type"] == "enumeration"]
    benchmark_enum = next(
        (c for c in enum_chunks if "Benchmark" in c.get("text", "")), None
    )
    assert benchmark_enum is not None


@pytest.mark.usefixtures("fresh_index")
def test_router_classifies_intents_correctly():
    # Pure router test — no index needed.
    assert classify_intent("List all schemes").intent == "list_distinct"
    assert classify_intent("Funds with NAV greater than 100").intent == "numeric_threshold"
    assert classify_intent("Top 5 funds by 1-year return").intent == "aggregate"
    assert classify_intent("What is the NAV of HDFC Top 100?").intent == "point_lookup"
    assert classify_intent("Hello there").intent == "qualitative"


def test_router_captures_entities_in_about_phrasing():
    # The follow-up shape that originally fell through to "aggregate" on the
    # "highest" keyword without ever capturing the entity.
    plan = classify_intent(
        "What about HDFC Balanced Advantage Fund, doesnt that have the highest AUM?"
    )
    assert any("HDFC Balanced Advantage" in p for p in plan.raw_entity_phrases)


def test_router_captures_entities_in_subject_position():
    # No "of/for/about" preposition, but the entity is the grammatical subject.
    plan = classify_intent("HDFC Top 100 Fund has the highest AUM, right?")
    assert any("HDFC Top 100" in p or "HDFC Top 100 Fund" in p for p in plan.raw_entity_phrases)


@pytest.mark.usefixtures("fresh_index")
def test_numeric_threshold_post_compute_via_query_table(tmp_path):
    fund_xlsx = tmp_path / "fund.xlsx"
    _write_fund_xlsx(fund_xlsx)
    ingest_file(fund_xlsx, "fund.xlsx")

    # The retrieval layer's job is to surface relevant chunks; the deterministic
    # numeric answer comes from query_table over the in-memory DataFrame.
    from app.analysis.query import query_table

    df = next(iter(state.dataframes_by_file_id.values()))
    result = query_table(
        df,
        filters=[{"column": "Daily AUM (Cr.)", "op": ">", "value": 5000.0}],
        limit=10,
    )
    # 3 of 4 rows have AUM > 5000; the 5000.0 row is excluded by strict >.
    assert len(result) == 3
    assert "Kotak Bluechip Fund" not in set(result["Scheme Name"])


@pytest.mark.usefixtures("fresh_index")
def test_alias_map_resolves_short_form(tmp_path):
    fund_xlsx = tmp_path / "fund.xlsx"
    _write_fund_xlsx(fund_xlsx)
    ingest_file(fund_xlsx, "fund.xlsx")

    # "HDFC Top 100" without "Fund" suffix should still resolve via normalize_name
    # (which strips the noise token).
    canon = state.composite.aliases.resolve("HDFC Top 100")
    assert canon == "hdfc top 100"
