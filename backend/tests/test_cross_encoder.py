"""Cross-encoder reranker tests. Use a stub model so the suite stays
network-free and deterministic — the real model is exercised manually."""
import numpy as np
import pytest

from app.retrieval import cross_encoder as ce_mod


class _StubCE:
    """Tiny stand-in for sentence_transformers.CrossEncoder. Returns a score
    that's the count of shared whitespace tokens between query and passage —
    enough overlap to drive a deterministic ordering for tests."""

    def predict(self, pairs, convert_to_numpy=True):  # noqa: ARG002 — signature parity
        scores = []
        for q, p in pairs:
            q_toks = set(q.lower().split())
            p_toks = set(p.lower().split())
            scores.append(float(len(q_toks & p_toks)))
        return np.array(scores)


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch):
    """Patch the lazy-loaded model with the stub for every test in this module.
    Clears the lru_cache first so a real model loaded by another test doesn't
    leak in."""
    ce_mod._model.cache_clear()
    monkeypatch.setattr(ce_mod, "_model", lambda: _StubCE())


def test_empty_input_returns_empty():
    assert ce_mod.cross_rerank("anything", []) == []


def test_reorders_by_token_overlap_with_query():
    chunks = [
        {"chunk_id": "a", "text": "hdfc top 100 fund nav regular value", "chunk_type": "row"},
        {"chunk_id": "b", "text": "axis bluechip benchmark return", "chunk_type": "row"},
        {"chunk_id": "c", "text": "hdfc top 100 aum daily", "chunk_type": "row"},
    ]
    out = ce_mod.cross_rerank("hdfc top 100 nav", chunks)
    # Chunk a has more overlap (hdfc, top, 100, nav) than c (hdfc, top, 100).
    assert [c["chunk_id"] for c in out[:2]] == ["a", "c"]
    # b has zero overlap and should sink to last.
    assert out[-1]["chunk_id"] == "b"


def test_attaches_ce_score_to_each_chunk():
    chunks = [
        {"chunk_id": "x", "text": "alpha beta gamma", "chunk_type": "row"},
    ]
    out = ce_mod.cross_rerank("alpha", chunks)
    assert "_ce_score" in out[0]
    assert isinstance(out[0]["_ce_score"], float)


def test_synthesizes_text_for_numeric_vector_chunks():
    """numeric_vector chunks have no `text` body — CE still needs something to
    embed against the query, so we synthesize a description from canonical_id
    and column list."""
    chunks = [
        {
            "chunk_id": "n1",
            "chunk_type": "numeric_vector",
            "canonical_id": "hdfc top 100",
            "numeric_columns": ["nav", "aum"],
        },
    ]
    # Should not raise — the synthesized text gives the stub something to score.
    out = ce_mod.cross_rerank("nav", chunks)
    assert out[0]["chunk_id"] == "n1"


def test_synthesizes_text_for_time_window_chunks():
    chunks = [
        {
            "chunk_id": "t1",
            "chunk_type": "time_window",
            "canonical_id": "hdfc top 100",
            "window": "365d",
        },
    ]
    out = ce_mod.cross_rerank("365d", chunks)
    assert out[0]["chunk_id"] == "t1"


def test_orchestrator_preserves_exact_id_order(monkeypatch):
    """Integration check: when CE is enabled, exact-id chunks still rank
    above semantic ones regardless of CE score."""
    from app.retrieval.orchestrator import _cross_encoder_rerank

    exact_meta = {"chunk_id": "ex1", "chunk_type": "row", "text": "irrelevant text"}
    sem_meta_high = {"chunk_id": "s1", "chunk_type": "row", "text": "shared shared shared"}
    sem_meta_low = {"chunk_id": "s2", "chunk_type": "row", "text": "different content here"}

    # Linear scores: exact has 1.0 (from exact_id boost), semantic has 0.5
    scored = [
        (1.0, exact_meta, {"exact_id": True}),
        (0.5, sem_meta_low, {"exact_id": False, "text_sim": 0.5}),
        (0.4, sem_meta_high, {"exact_id": False, "text_sim": 0.4}),
    ]
    out = _cross_encoder_rerank("shared", scored)

    # Exact is first.
    assert out[0][1]["chunk_id"] == "ex1"
    # Among semantic chunks, the one that overlaps the query ("shared shared shared")
    # is now ahead of the one that doesn't.
    semantic_ids = [tup[1]["chunk_id"] for tup in out[1:]]
    assert semantic_ids == ["s1", "s2"]


def test_orchestrator_attaches_ce_score_to_sigs():
    from app.retrieval.orchestrator import _cross_encoder_rerank

    meta = {"chunk_id": "s1", "chunk_type": "row", "text": "alpha beta"}
    scored = [(0.5, meta, {"exact_id": False, "text_sim": 0.5})]
    out = _cross_encoder_rerank("alpha", scored)
    assert "ce_score" in out[0][2]
    assert isinstance(out[0][2]["ce_score"], float)
