import pandas as pd

from app.chunkers.common import make_meta
from app.id.canonical import AliasMap, normalize_name
from app.parsers.tabular import _summarize, enumerations, synopsis
from app.rag.schema import ChunkMeta


def pick_name_column(df: pd.DataFrame) -> str | None:
    """Heuristic: the entity-name column is the first text column whose values
    are mostly unique (>=80% distinct, >=2 distinct values). Fund-Performance
    files have "Scheme Name" first, but we don't hardcode that — any tabular
    schema where the first text column is high-cardinality gets entity chunks."""
    text_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    n = len(df)
    if n == 0:
        return None
    for col in text_cols:
        nonnull = df[col].dropna()
        if len(nonnull) == 0:
            continue
        distinct = nonnull.nunique()
        if distinct >= 2 and distinct / max(len(nonnull), 1) >= 0.8:
            return col
    return None


def _row_body(df: pd.DataFrame, i: int, file: str) -> str:
    row = df.iloc[i]
    parts: list[str] = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        parts.append(f"{col}={val}")
    return f"Row from {file}: " + "; ".join(parts) + "."


def build_tabular_chunks(
    df: pd.DataFrame,
    file: str,
    file_id: str,
    *,
    sheet: str | None,
    aliases: AliasMap,
    cleaning_report: dict | None = None,
) -> tuple[list[tuple[ChunkMeta, str]], str | None]:
    """Build synopsis / tabular_summary / enumeration / row chunks for one
    cleaned DataFrame. Returns (chunks, name_col) — the caller uses name_col
    to drive the entity / numeric_vector / time_window chunkers.

    Each chunk carries a `text` body; the caller embeds the bodies in a single
    batch for efficiency."""
    out: list[tuple[ChunkMeta, str]] = []
    name_col = pick_name_column(df)

    # 1. synopsis — high-recall narrative hook for "what's in X?"
    syn = synopsis(file, df)
    out.append((make_meta("synopsis", file=file, file_id=file_id, sheet=sheet, text=syn), syn))

    # 2. tabular_summary — head/tail/middle samples + describe
    report = cleaning_report or {
        "rows_dropped": 0,
        "cols_dropped": 0,
        "numeric_coerced": [],
        "date_coerced": [],
    }
    summ = _summarize(df, file, report)
    out.append(
        (make_meta("tabular_summary", file=file, file_id=file_id, sheet=sheet, text=summ), summ)
    )

    # 3. enumerations — complete distinct-value listings per qualifying text col
    for col in df.select_dtypes(include=["object", "string"]).columns:
        unique = df[col].dropna().unique()
        n_unique = len(unique)
        if n_unique < 2 or n_unique > 500:
            continue
        values = sorted(str(v) for v in unique)
        body = (
            f"All distinct values in column '{col}' from {file} "
            f"({n_unique} total): {', '.join(values)}."
        )
        meta = make_meta(
            "enumeration",
            file=file,
            file_id=file_id,
            sheet=sheet,
            text=body,
            column_names=[col],
        )
        out.append((meta, body))

    # 4. row chunks — one per row, with canonical_id derived from name_col
    max_rows = 500
    n = min(len(df), max_rows)
    for i in range(n):
        body = _row_body(df, i, file)
        canon = None
        if name_col:
            raw = df.iloc[i][name_col]
            if pd.notna(raw):
                canon = aliases.register(str(raw))
        meta = make_meta(
            "row",
            file=file,
            file_id=file_id,
            sheet=sheet,
            row_number=i,
            canonical_id=canon,
            text=body,
            column_names=list(df.columns),
        )
        out.append((meta, body))

    return out, name_col
