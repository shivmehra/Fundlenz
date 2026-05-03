"""Back-compat shim. The v2 pipeline routes retrieval through
`app.retrieval.orchestrator.retrieve_v2`. This module preserves the old
`retrieve(query, store, k=None)` signature so legacy callers (and a few
tests that imported it) still work, but it no longer accepts a raw
VectorStore — it expects a CompositeIndex (or anything with `text_search`
and the orchestrator-required surface)."""
from app.config import settings
from app.index.composite import CompositeIndex
from app.rag.schema import ChunkMeta
from app.retrieval.orchestrator import retrieve_v2


def retrieve(query: str, store: CompositeIndex, k: int | None = None) -> list[ChunkMeta]:
    return retrieve_v2(query, store, k or settings.top_k)


def format_context(chunks: list[ChunkMeta]) -> str:
    from app.retrieval.orchestrator import format_context_v2
    return format_context_v2(chunks)
