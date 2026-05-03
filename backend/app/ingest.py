import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from app import state
from app.chunkers.entity_chunker import build_entity_chunks
from app.chunkers.numeric_encoder import build_numeric_vectors
from app.chunkers.tabular_chunker import build_tabular_chunks
from app.chunkers.text_chunker import build_table_chunk, build_text_chunks
from app.chunkers.timewindow_chunker import build_time_windows, detect_timeseries
from app.config import settings
from app.id.canonical import normalize_name
from app.index.composite import IngestItem
from app.parsers.docx import parse_docx
from app.parsers.pdf import parse_pdf
from app.parsers.tabular import parse_tabular
from app.rag.embedder import embed
from app.rag.numeric_scaler import NumericScaler
from app.rag.schema import ChunkMeta


def ingest_file(path: Path, original_filename: str) -> dict:
    """Parse, chunk, embed, and index a file under the v2 multi-index pipeline.

    Returns a status dict including a `summary` string suitable for the UI.
    Tabular files (CSV/multi-sheet Excel) produce one logical entry per sheet,
    each with its own file_id, dataframe, scaler, and document-list entry.
    Non-tabular files produce a single entry."""
    suffix = path.suffix.lower()
    summary_parts: list[str] = []
    items: list[IngestItem] = []

    primary_file_id = uuid.uuid4().hex[:12]

    if suffix == ".pdf":
        records = parse_pdf(path)
        n_pages = len(records)
        for record in records:
            page = record["page"]
            header = f"[{original_filename}] page {page}"
            for meta, body in build_text_chunks(
                record["text"],
                file=original_filename,
                file_id=primary_file_id,
                page=page,
                max_tokens=settings.chunk_tokens,
                overlap=settings.chunk_overlap,
                doc_header=header,
            ):
                items.append(IngestItem(meta=meta, text=body))
            for table_md in record["tables_md"]:
                meta, body = build_table_chunk(
                    table_md, file=original_filename, file_id=primary_file_id, page=page
                )
                items.append(IngestItem(meta=meta, text=body))
        state.documents[original_filename] = {
            "file_id": primary_file_id,
            "type": "pdf",
            "chunks": sum(1 for it in items if it.meta["chunk_type"] in ("text", "table")),
        }
        summary_parts.append(f"{n_pages}-page PDF.")
        first_text = next((r["text"] for r in records if r["text"].strip()), "")
        excerpt = _excerpt(first_text)
        if excerpt:
            summary_parts.append(f"First page begins:\n\n> {excerpt}")

    elif suffix == ".docx":
        text = parse_docx(path)
        for meta, body in build_text_chunks(
            text,
            file=original_filename,
            file_id=primary_file_id,
            max_tokens=settings.chunk_tokens,
            overlap=settings.chunk_overlap,
            doc_header=f"[{original_filename}]",
        ):
            items.append(IngestItem(meta=meta, text=body))
        state.documents[original_filename] = {
            "file_id": primary_file_id,
            "type": "docx",
            "chunks": sum(1 for it in items if it.meta["chunk_type"] == "text"),
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
            sheet_label = logical_name.split(" :: ", 1)[1] if " :: " in logical_name else None

            state.dataframes_by_file_id[sheet_file_id] = df
            state.filename_to_file_id[logical_name] = sheet_file_id

            sheet_items = _ingest_tabular_sheet(
                df=df,
                file=logical_name,
                file_id=sheet_file_id,
                sheet=sheet_label,
            )
            items.extend(sheet_items)

            state.documents[logical_name] = {
                "file_id": sheet_file_id,
                "type": suffix.lstrip("."),
                "chunks": len(sheet_items),
            }
            summary_parts.append(_synopsis_one_line(df, logical_name))

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if not items:
        return {
            "file_id": primary_file_id,
            "filename": original_filename,
            "chunks": 0,
            "summary": "",
        }

    # Single batch embedding for all text-bearing chunks.
    text_bodies: list[str] = []
    text_owners: list[IngestItem] = []
    for it in items:
        if it.text and it.text_vec is None:
            text_bodies.append(it.text)
            text_owners.append(it)
    if text_bodies:
        vecs = embed(text_bodies)  # already normalized + float32
        for owner, vec in zip(text_owners, vecs):
            owner.text_vec = vec

    state.composite.add_chunks(items)
    state.composite.save()

    return {
        "file_id": primary_file_id,
        "filename": original_filename,
        "chunks": len(items),
        "summary": "\n\n".join(summary_parts),
    }


def _ingest_tabular_sheet(
    df: pd.DataFrame, file: str, file_id: str, sheet: str | None
) -> list[IngestItem]:
    """Build the full chunk family for one cleaned tabular DataFrame."""
    items: list[IngestItem] = []
    aliases = state.composite.aliases

    # 1. synopsis / tabular_summary / enumeration / row chunks
    chunks, name_col = build_tabular_chunks(
        df, file=file, file_id=file_id, sheet=sheet, aliases=aliases
    )
    for meta, body in chunks:
        keys = _inverted_keys_for_chunk(meta, df, name_col)
        items.append(IngestItem(meta=meta, text=body, inverted_keys=keys))

    # 2. entity chunks
    if name_col:
        for meta, body in build_entity_chunks(
            df, file=file, file_id=file_id, sheet=sheet, name_col=name_col, aliases=aliases
        ):
            keys: list[tuple[str, str, str]] = []
            if meta.get("canonical_id"):
                keys.append(("id", meta["canonical_id"], ""))  # type: ignore[arg-type]
            items.append(IngestItem(meta=meta, text=body, inverted_keys=keys))

    # 3. numeric_vector chunks (always built — drives numeric ANN when enabled,
    # and the metadata is useful for provenance even when ANN is off)
    num_cols = list(df.select_dtypes(include="number").columns)
    if num_cols:
        scaler = NumericScaler()
        scaler.fit(df, num_cols)
        state.composite.register_scaler(file_id, scaler)
        for meta, vec in build_numeric_vectors(
            df, file=file, file_id=file_id, sheet=sheet, name_col=name_col, scaler=scaler
        ):
            keys: list[tuple[str, str, str]] = []
            if meta.get("canonical_id"):
                keys.append(("id", meta["canonical_id"], ""))  # type: ignore[arg-type]
            items.append(IngestItem(meta=meta, numeric_vec=vec, inverted_keys=keys))

    # 4. time_window chunks (only if multi-date data is detected)
    if name_col:
        date_col = detect_timeseries(df, name_col)
        if date_col:
            for meta, body in build_time_windows(
                df,
                file=file,
                file_id=file_id,
                sheet=sheet,
                name_col=name_col,
                date_col=date_col,
                value_cols=num_cols,
            ):
                keys: list[tuple[str, str, str]] = []
                if meta.get("canonical_id"):
                    keys.append(("id", meta["canonical_id"], ""))  # type: ignore[arg-type]
                items.append(IngestItem(meta=meta, text=body, inverted_keys=keys))

    return items


def _inverted_keys_for_chunk(
    meta: ChunkMeta, df: pd.DataFrame, name_col: str | None
) -> list[tuple[str, str, str]]:
    """Build the inverted-index key list for a chunk based on its type. Cell
    keys for row chunks, enum keys for enumeration chunks, id keys when the
    chunk has a canonical_id."""
    keys: list[tuple[str, str, str]] = []
    if meta.get("canonical_id"):
        keys.append(("id", meta["canonical_id"], ""))  # type: ignore[arg-type]

    if meta["chunk_type"] == "row" and meta.get("row_number") is not None:
        i = int(meta["row_number"])  # type: ignore[arg-type]
        if 0 <= i < len(df):
            row = df.iloc[i]
            for col, val in row.items():
                if pd.isna(val):
                    continue
                # Index identifier-ish columns and the name column (already in id key).
                # We index every cell — exact-match lookups are bounded by value
                # specificity, and the JSON dict's growth is linear with chunks.
                keys.append(("cell", str(col), str(val)))
    elif meta["chunk_type"] == "enumeration":
        cols = meta.get("column_names") or []
        for col in cols:
            keys.append(("enum", col, ""))
    return keys


def _excerpt(text: str, words: int = 120) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    return " ".join(tokens[:words]) + ("..." if len(tokens) > words else "")


def _synopsis_one_line(df: pd.DataFrame, logical_name: str) -> str:
    """One-line synopsis for the per-sheet UI summary. The full synopsis chunk
    is built inside build_tabular_chunks; this is just a status snippet."""
    return f"{logical_name}: {len(df)} rows, {len(df.columns)} columns."
