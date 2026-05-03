import numpy as np
import pytest

from app.index.text_ann import TextANN


_DIM = 384


def _vec(values: list[float]) -> np.ndarray:
    """Build a 384-dim L2-normalized batch (1xDIM) with given leading components."""
    v = np.zeros(_DIM, dtype="float32")
    v[: len(values)] = values
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.reshape(1, -1).astype("float32")


@pytest.fixture
def store(tmp_path):
    s = TextANN(tmp_path, dim=_DIM)
    s.load_or_init()
    return s


def test_search_on_empty_returns_empty(store):
    assert store.search(_vec([1.0]), k=5) == []


def test_add_and_search_returns_chunk_id_and_score(store):
    store.add(_vec([1.0]), ["a"])
    store.add(_vec([0.0, 1.0]), ["b"])
    results = store.search(_vec([1.0]), k=2)
    assert results[0][0] == "a"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_save_and_reload_preserves_vectors_and_ids(tmp_path):
    s1 = TextANN(tmp_path, dim=_DIM)
    s1.load_or_init()
    s1.add(_vec([1.0]), ["a"])
    s1.add(_vec([0.0, 1.0]), ["b"])
    s1.save()

    s2 = TextANN(tmp_path, dim=_DIM)
    s2.load_or_init()
    assert s2.total() == 2
    results = s2.search(_vec([1.0]), k=2)
    assert results[0][0] == "a"


def test_search_clamps_k_to_index_size(store):
    store.add(_vec([1.0]), ["a"])
    store.add(_vec([0.0, 1.0]), ["b"])
    # k > ntotal — must not error, just returns all available.
    results = store.search(_vec([1.0]), k=10)
    assert len(results) == 2


def test_add_rejects_mismatched_lengths(store):
    with pytest.raises(AssertionError):
        store.add(_vec([1.0]), ["a", "b"])


def test_rebuild_keeping_drops_others(store):
    store.add(_vec([1.0]), ["a"])
    store.add(_vec([0.0, 1.0]), ["b"])
    # Reconstruct keep-set using the stored FAISS reconstruction path.
    keep_ids = ["b"]
    keep_vecs = np.vstack([store.index.reconstruct(1)]).astype("float32")
    store.rebuild_keeping(set(keep_ids), keep_vecs)
    store.chunk_ids = keep_ids
    assert store.total() == 1
    results = store.search(_vec([0.0, 1.0]), k=5)
    assert len(results) == 1
    assert results[0][0] == "b"
