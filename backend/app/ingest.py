import uuid
from pathlib import Path

from app import state
from app.config import settings
from app.parsers.docx import parse_docx
from app.parsers.pdf import parse_pdf
from app.parsers.tabular import enumerations, parse_tabular, row_chunks, synopsis
from app.rag.embedder import chunk_text, embed


def ingest_file(path: Path, original_filename: str) -> dict:
    """Parse, chunk, embed, and index a file. Returns status dict including a
    human-readable `summary` string suitable for showing in the UI.

    Tabular files (CSV/Excel) may produce multiple logical entries (one per
    sheet); each gets its own file_id, dataframe, and document-list entry.
    Non-tabular files always produce a single entry."""
    suffix = path.suffix.lower()

    chunks: list[str] = []
    metas: list[dict] = []
    primary_file_id = uuid.uuid4().hex[:12]
    summary_parts: list[str] = []

    if suffix == ".pdf":
        records = parse_pdf(path)
        n_pages = len(records)
        for record in records:
            page = record["page"]
            for chunk in chunk_text(record["text"], settings.chunk_tokens, settings.chunk_overlap):
                chunks.append(chunk)
                metas.append({
                    "file_id": primary_file_id, "filename": original_filename,
                    "page": page, "type": "text", "text": chunk,
                })
            for table_md in record["tables_md"]:
                chunks.append(table_md)
                metas.append({
                    "file_id": primary_file_id, "filename": original_filename,
                    "page": page, "type": "table", "text": table_md,
                })
        state.documents[original_filename] = {
            "file_id": primary_file_id, "type": "pdf", "chunks": len(chunks),
        }
        summary_parts.append(f"{n_pages}-page PDF.")
        first_text = next((r["text"] for r in records if r["text"].strip()), "")
        excerpt = _excerpt(first_text)
        if excerpt:
            summary_parts.append(f"First page begins:\n\n> {excerpt}")

    elif suffix == ".docx":
        text = parse_docx(path)
        for chunk in chunk_text(text, settings.chunk_tokens, settings.chunk_overlap):
            chunks.append(chunk)
            metas.append({
                "file_id": primary_file_id, "filename": original_filename,
                "page": None, "type": "text", "text": chunk,
            })
        state.documents[original_filename] = {
            "file_id": primary_file_id, "type": "docx", "chunks": len(chunks),
        }
        summary_parts.append("Word document.")
        excerpt = _excerpt(text)
        if excerpt:
            summary_parts.append(f"Begins:\n\n> {excerpt}")

    elif suffix in (".csv", ".xlsx", ".xls"):
        sheets = parse_tabular(path)
        if len(sheets) > 1:
            summary_parts.append(f"Detected {len(sheets)} sheets.")
        for entry in sheets:
            sheet_file_id = uuid.uuid4().hex[:12]
            logical_name = entry["logical_name"]
            df = entry["df"]
            tab_summary = entry["summary"]

            state.dataframes_by_file_id[sheet_file_id] = df
            state.filename_to_file_id[logical_name] = sheet_file_id

            sheet_start = len(chunks)

            # Synopsis as the first chunk — high-recall hook for "what's in X?".
            syn = synopsis(logical_name, df)
            chunks.append(syn)
            metas.append({
                "file_id": sheet_file_id, "filename": logical_name,
                "page": None, "type": "synopsis", "text": syn,
            })

            # Enumeration chunks: complete listings of distinct values for
            # bounded-cardinality text columns. Enables "list all X" answers.
            for enum_text in enumerations(logical_name, df):
                for chunk in chunk_text(enum_text, settings.chunk_tokens, settings.chunk_overlap):
                    chunks.append(chunk)
                    metas.append({
                        "file_id": sheet_file_id, "filename": logical_name,
                        "page": None, "type": "enumeration", "text": chunk,
                    })

            for chunk in chunk_text(tab_summary, settings.chunk_tokens, settings.chunk_overlap):
                chunks.append(chunk)
                metas.append({
                    "file_id": sheet_file_id, "filename": logical_name,
                    "page": None, "type": "tabular_summary", "text": chunk,
                })

            # Per-row chunks: enables point lookups like "expense ratio of Fund X".
            # The summary alone only contains head/tail/middle samples — without
            # row chunks, point lookups against non-sampled rows fail silently.
            for row_text in row_chunks(logical_name, df):
                for chunk in chunk_text(row_text, settings.chunk_tokens, settings.chunk_overlap):
                    chunks.append(chunk)
                    metas.append({
                        "file_id": sheet_file_id, "filename": logical_name,
                        "page": None, "type": "row", "text": chunk,
                    })

            state.documents[logical_name] = {
                "file_id": sheet_file_id,
                "type": suffix.lstrip("."),
                "chunks": len(chunks) - sheet_start,
            }
            summary_parts.append(syn)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if not chunks:
        return {
            "file_id": primary_file_id, "filename": original_filename,
            "chunks": 0, "summary": "",
        }

    vectors = embed(chunks)
    state.vector_store.add(vectors, metas)
    state.vector_store.save()

    return {
        "file_id": primary_file_id,
        "filename": original_filename,
        "chunks": len(chunks),
        "summary": "\n\n".join(summary_parts),
    }


def _excerpt(text: str, words: int = 120) -> str:
    """First N words of `text`, with ellipsis if truncated. Whitespace
    collapsed so the excerpt reads cleanly inside a markdown blockquote."""
    tokens = text.split()
    if not tokens:
        return ""
    return " ".join(tokens[:words]) + ("..." if len(tokens) > words else "")
