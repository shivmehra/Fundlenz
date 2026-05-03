from app.index.composite import CompositeIndex
from app.rag.schema import ChunkMeta
from app.retrieval.exact import extract_cell_predicates, resolve_canonical_id
from app.retrieval.rerank import deterministic_score
from app.retrieval.router import Plan, classify_intent
from app.retrieval.semantic import text_search


# How many text-ANN hits to over-fetch before reranking. Bigger = more recall,
# higher rerank cost. 4x top_k gives the rerank room to surface a row chunk
# that ANN ranked low but exact-match boosts.
_OVERFETCH_FACTOR = 4


def retrieve_v2(
    query: str,
    idx: CompositeIndex,
    k: int,
    plan: Plan | None = None,
) -> list[ChunkMeta]:
    """Top-level retrieval: classify → exact-match fan-out → text ANN over-fetch
    → score combination → top-k. Returns ChunkMeta dicts with `_score` and
    `_score_breakdown` fields injected for caller introspection."""
    if plan is None:
        plan = classify_intent(query)
    plan.canonical_id = plan.canonical_id or resolve_canonical_id(plan, idx)

    # cid -> dict of signals
    candidates: dict[str, dict] = {}

    # 1) exact identifier match — short-circuits across all chunk types
    if plan.canonical_id:
        for cid in idx.inverted.lookup_id(plan.canonical_id):
            candidates.setdefault(cid, {})["exact_id"] = True

    # 2) cell-level predicate match
    for col, val in extract_cell_predicates(query):
        for cid in idx.inverted.lookup_cell(col, val):
            candidates.setdefault(cid, {})["exact_cell"] = True

    # 3) text ANN — over-fetch for rerank headroom
    overfetch = max(k * _OVERFETCH_FACTOR, 20)
    for cid, sim in text_search(query, idx, overfetch):
        candidates.setdefault(cid, {})["text_sim"] = sim

    # Note on numeric ANN: gated by config and only meaningful when we can
    # build a numeric query vector (e.g. "fund similar to NAV=120, AUM=500").
    # For now we leave numeric retrieval to the post-compute layer
    # (query_table) which is exact and deterministic.

    if not candidates:
        return []

    # 4) hydrate metadata + rerank
    metas = idx.meta.get_many(list(candidates.keys()))
    by_id = {m["chunk_id"]: m for m in metas}
    scored: list[tuple[float, ChunkMeta, dict]] = []
    for cid, sigs in candidates.items():
        m = by_id.get(cid)
        if m is None:
            continue
        score = deterministic_score(
            exact_id=bool(sigs.get("exact_id")),
            exact_cell=bool(sigs.get("exact_cell")),
            text_sim=float(sigs.get("text_sim", 0.0)),
            numeric_dist=sigs.get("numeric_dist"),
            chunk_type=m["chunk_type"],
            intent=plan.intent,
        )
        scored.append((score, m, sigs))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[ChunkMeta] = []
    for s, m, sigs in scored[:k]:
        m_out = dict(m)
        m_out["_score"] = s
        m_out["_score_breakdown"] = {
            "exact_id": bool(sigs.get("exact_id")),
            "exact_cell": bool(sigs.get("exact_cell")),
            "text_sim": float(sigs.get("text_sim", 0.0)),
            "numeric_dist": sigs.get("numeric_dist"),
            "intent": plan.intent,
            "chunk_type": m["chunk_type"],
        }
        out.append(m_out)  # type: ignore[arg-type]
    return out


def format_context_v2(chunks: list[ChunkMeta]) -> str:
    """Render retrieved chunks as numbered citations. numeric_vector chunks
    have no `text` body — synthesize a brief description so the LLM still
    sees them as evidence (and they still appear in `Sources`)."""
    if not chunks:
        return "(no relevant context found)"
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        loc = c.get("file", "?")
        if c.get("page") is not None:
            loc += f" p.{c['page']}"
        body = c.get("text")
        if not body and c["chunk_type"] == "numeric_vector":
            cols = c.get("numeric_columns", [])
            cid = c.get("canonical_id") or "?"
            body = f"Numeric snapshot for entity '{cid}' covering: {', '.join(cols) or '(no columns)'}."
        elif not body and c["chunk_type"] == "time_window":
            body = f"Time window {c.get('window')} stats for {c.get('canonical_id')}."
        blocks.append(f"[{i}] {loc}\n{body or ''}")
    return "\n\n".join(blocks)
