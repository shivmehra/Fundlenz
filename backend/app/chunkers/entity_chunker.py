import pandas as pd

from app.chunkers.common import make_meta
from app.id.canonical import AliasMap, normalize_name
from app.rag.schema import ChunkMeta


def build_entity_chunks(
    df: pd.DataFrame,
    file: str,
    file_id: str,
    *,
    sheet: str | None,
    name_col: str,
    aliases: AliasMap,
) -> list[tuple[ChunkMeta, str]]:
    """One chunk per canonical_id: aggregates all attributes for the entity
    plus its known aliases. For snapshot data this is one row per entity; for
    time-series data we take the most-recent row's attributes (the time-window
    chunker emits the rolling-stats view separately)."""
    out: list[tuple[ChunkMeta, str]] = []
    if name_col not in df.columns or len(df) == 0:
        return out

    work = df.copy()
    work["_canon"] = work[name_col].astype(str).map(normalize_name)
    work = work[work["_canon"] != ""]
    if work.empty:
        return out

    # Use the latest date (if any datetime col exists) to pick a representative
    # row per entity. Otherwise the first occurrence.
    date_cols = list(df.select_dtypes(include="datetime").columns)
    if date_cols:
        work = work.sort_values(date_cols[0])

    for canon, group in work.groupby("_canon", sort=True):
        rep = group.iloc[-1]
        attr_parts: list[str] = []
        for col in df.columns:
            val = rep.get(col)
            if pd.isna(val) or col == "_canon":
                continue
            attr_parts.append(f"{col}={val}")
        alias_list = aliases.aliases_for(canon)
        alias_phrase = (
            f"Aliases: {', '.join(alias_list)}." if alias_list else "Aliases: (none registered)."
        )
        body = (
            f"Entity {canon} from {file}. {alias_phrase} "
            f"Attributes: " + "; ".join(attr_parts) + "."
        )
        meta = make_meta(
            "entity",
            file=file,
            file_id=file_id,
            sheet=sheet,
            canonical_id=canon,
            text=body,
            aliases=alias_list,
        )
        out.append((meta, body))
    return out
