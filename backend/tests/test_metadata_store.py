import pytest

from app.index.metadata_store import MetaStore
from app.rag.schema import SCHEMA_VERSION, ChunkMeta


def _meta(chunk_id: str, **overrides) -> ChunkMeta:
    base: ChunkMeta = {
        "chunk_id": chunk_id,
        "file": "fund.xlsx",
        "sheet": None,
        "row_number": None,
        "chunk_type": "row",
        "canonical_id": None,
        "ingestion_time": "2026-05-03T00:00:00Z",
        "version": SCHEMA_VERSION,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


@pytest.fixture
def store(tmp_path):
    s = MetaStore(tmp_path / "metadata.sqlite")
    s.init_schema()
    yield s
    s.close()


def test_upsert_and_get(store):
    m = _meta("c1", canonical_id="hdfc top 100", text="row body")
    store.upsert_many([m])
    fetched = store.get("c1")
    assert fetched is not None
    assert fetched["chunk_id"] == "c1"
    assert fetched["canonical_id"] == "hdfc top 100"
    assert fetched["text"] == "row body"


def test_get_many_preserves_request_order(store):
    metas = [_meta(f"c{i}") for i in range(5)]
    store.upsert_many(metas)
    out = store.get_many(["c3", "c0", "c4"])
    assert [m["chunk_id"] for m in out] == ["c3", "c0", "c4"]


def test_by_canonical_returns_all_matches(store):
    store.upsert_many([
        _meta("c1", canonical_id="x", chunk_type="row"),
        _meta("c2", canonical_id="x", chunk_type="entity"),
        _meta("c3", canonical_id="y", chunk_type="row"),
    ])
    out = store.by_canonical("x")
    assert {m["chunk_id"] for m in out} == {"c1", "c2"}


def test_by_file_filters_by_filename(store):
    store.upsert_many([
        _meta("c1", file="a.xlsx"),
        _meta("c2", file="b.xlsx"),
        _meta("c3", file="a.xlsx"),
    ])
    out = store.by_file("a.xlsx")
    assert {m["chunk_id"] for m in out} == {"c1", "c3"}


def test_count_and_count_by_type(store):
    store.upsert_many([
        _meta("c1", chunk_type="row"),
        _meta("c2", chunk_type="row"),
        _meta("c3", chunk_type="entity"),
    ])
    assert store.count() == 3
    by_type = store.count_by_type()
    assert by_type["row"] == 2
    assert by_type["entity"] == 1


def test_delete_by_file_returns_chunk_ids(store):
    store.upsert_many([
        _meta("c1", file="a.xlsx"),
        _meta("c2", file="b.xlsx"),
        _meta("c3", file="a.xlsx"),
    ])
    deleted = store.delete_by_file("a.xlsx")
    assert set(deleted) == {"c1", "c3"}
    assert store.count() == 1


def test_upsert_replaces_existing(store):
    store.upsert_many([_meta("c1", text="first")])
    store.upsert_many([_meta("c1", text="second")])
    assert store.count() == 1
    assert store.get("c1")["text"] == "second"


def test_list_files(store):
    store.upsert_many([
        _meta("c1", file="b.xlsx"),
        _meta("c2", file="a.xlsx"),
        _meta("c3", file="b.xlsx"),
    ])
    assert store.list_files() == ["a.xlsx", "b.xlsx"]
