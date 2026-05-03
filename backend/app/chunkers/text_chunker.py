from app.chunkers.common import make_meta
from app.rag.embedder import chunk_text
from app.rag.schema import ChunkMeta


def build_text_chunks(
    text: str,
    file: str,
    file_id: str,
    *,
    sheet: str | None = None,
    page: int | None = None,
    max_tokens: int,
    overlap: int,
    doc_header: str | None = None,
) -> list[tuple[ChunkMeta, str]]:
    """Sliding-window prose chunker. Adds an optional doc_header prefix to each
    chunk (e.g. "[FundFactSheet] page 3:") to anchor retrieval scoring on
    page/section even when the body text doesn't repeat that context."""
    out: list[tuple[ChunkMeta, str]] = []
    parts = chunk_text(text, max_tokens=max_tokens, overlap=overlap)
    for i, body in enumerate(parts):
        framed = f"{doc_header}\n{body}" if doc_header else body
        meta = make_meta(
            "text",
            file=file,
            file_id=file_id,
            sheet=sheet,
            row_number=i,
            text=framed,
            page=page,
        )
        out.append((meta, framed))
    return out


def build_table_chunk(
    table_md: str,
    file: str,
    file_id: str,
    *,
    page: int | None = None,
) -> tuple[ChunkMeta, str]:
    """PDF tables are kept whole — chopping a markdown table mid-row breaks the
    header/cell alignment the embedder relies on."""
    meta = make_meta(
        "table",
        file=file,
        file_id=file_id,
        text=table_md,
        page=page,
    )
    return meta, table_md
