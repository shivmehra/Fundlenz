"""Cross-encoder reranker. Runs after the deterministic linear reranker on
the semantic tier — exact-id matches bypass it to preserve the
"exact match always wins" guarantee."""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(settings.cross_encoder_model)


def _chunk_text_for_ce(c: dict) -> str:
    """Cross-encoders need a textual body. Synthesize one for chunk types
    that don't carry `text` (numeric_vector, time_window) so they can still
    participate in reranking instead of being dropped."""
    body = c.get("text")
    if body:
        return body
    ct = c.get("chunk_type")
    cid = c.get("canonical_id") or "?"
    if ct == "numeric_vector":
        cols = c.get("numeric_columns") or []
        return f"Numeric snapshot for {cid} covering: {', '.join(cols) or '(no columns)'}."
    if ct == "time_window":
        return f"Time-window {c.get('window')} statistics for {cid}."
    return cid


def cross_rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Re-score `chunks` against `query` using a cross-encoder. Mutates each
    chunk in-place to attach `_ce_score` (raw logit), then returns the list
    sorted by that score descending. No-op on empty input."""
    if not chunks:
        return chunks
    pairs = [[query, _chunk_text_for_ce(c)] for c in chunks]
    scores = _model().predict(pairs, convert_to_numpy=True)
    for c, s in zip(chunks, scores):
        c["_ce_score"] = float(s)
    return sorted(chunks, key=lambda c: c["_ce_score"], reverse=True)
