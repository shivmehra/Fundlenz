import datetime as dt
import uuid

from app.rag.schema import ChunkMeta, ChunkType, SCHEMA_VERSION


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_chunk_id() -> str:
    return uuid.uuid4().hex


def make_meta(
    chunk_type: ChunkType,
    file: str,
    file_id: str,
    sheet: str | None = None,
    row_number: int | None = None,
    canonical_id: str | None = None,
    **extra,
) -> ChunkMeta:
    """Build a ChunkMeta with the 8 mandatory fields populated. Extra kwargs
    flow through (text, page, aliases, stats, window, etc.)."""
    meta: ChunkMeta = {
        "chunk_id": new_chunk_id(),
        "file": file,
        "sheet": sheet,
        "row_number": row_number,
        "chunk_type": chunk_type,
        "canonical_id": canonical_id,
        "ingestion_time": now_iso(),
        "version": SCHEMA_VERSION,
        "file_id": file_id,
    }
    for k, v in extra.items():
        if v is not None:
            meta[k] = v  # type: ignore[literal-required]
    return meta
