import numpy as np
import pytest

from app.rag.vector_store import VectorStore, _EMBED_DIM


def _vec(values: list[float]) -> np.ndarray:
    """Build a 384-dim L2-normalized vector with given leading components."""
    v = np.zeros(_EMBED_DIM, dtype="float32")
    v[: len(values)] = values
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.reshape(1, -1).astype("float32")


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path)
    s.load_or_init()
    return s


def test_search_on_empty_index_returns_nothing(store):
    assert store.search(_vec([1.0]), k=5) == []


def test_add_and_search_returns_nearest(store):
    store.add(_vec([1.0]), [{"filename": "a.pdf", "text": "hello"}])
    store.add(_vec([0.0, 1.0]), [{"filename": "b.pdf", "text": "world"}])

    results = store.search(_vec([1.0]), k=2)
    assert results[0]["filename"] == "a.pdf"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_save_and_reload_preserves_index(tmp_path):
    s1 = VectorStore(tmp_path)
    s1.load_or_init()
    s1.add(_vec([1.0]), [{"filename": "a.pdf", "text": "x"}])
    s1.save()

    s2 = VectorStore(tmp_path)
    s2.load_or_init()
    assert s2.index.ntotal == 1
    results = s2.search(_vec([1.0]), k=1)
    assert results[0]["filename"] == "a.pdf"


def test_list_filenames_dedupes_preserving_insertion_order(store):
    store.add(_vec([1.0]), [{"filename": "a.pdf"}])
    store.add(_vec([0.0, 1.0]), [{"filename": "b.pdf"}])
    store.add(_vec([0.0, 0.0, 1.0]), [{"filename": "a.pdf"}])
    assert store.list_filenames() == ["a.pdf", "b.pdf"]


def test_delete_removes_all_chunks_for_filename(store):
    store.add(_vec([1.0]), [{"filename": "a.pdf", "text": "x"}])
    store.add(_vec([0.0, 1.0]), [{"filename": "b.pdf", "text": "y"}])
    store.add(_vec([0.0, 0.0, 1.0]), [{"filename": "a.pdf", "text": "z"}])

    deleted = store.delete_by_filename("a.pdf")
    assert deleted is True
    assert store.index.ntotal == 1
    assert store.list_filenames() == ["b.pdf"]


def test_delete_returns_false_when_filename_missing(store):
    store.add(_vec([1.0]), [{"filename": "a.pdf", "text": "x"}])
    assert store.delete_by_filename("missing.pdf") is False
    assert store.index.ntotal == 1
