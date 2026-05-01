from app.config import settings
from app.rag.embedder import embed


def retrieve(query: str, store, k: int | None = None) -> list[dict]:
    k = k or settings.top_k
    qv = embed([query])
    return store.search(qv, k)


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no relevant context found)"
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        loc = c.get("filename", "?")
        if c.get("page") is not None:
            loc += f" p.{c['page']}"
        blocks.append(f"[{i}] {loc}\n{c.get('text', '')}")
    return "\n\n".join(blocks)
