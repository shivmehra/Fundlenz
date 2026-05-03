import numpy as np
import pandas as pd

from app.chunkers.common import make_meta
from app.id.canonical import normalize_name
from app.rag.numeric_scaler import NumericScaler
from app.rag.schema import ChunkMeta


def build_numeric_vectors(
    df: pd.DataFrame,
    file: str,
    file_id: str,
    *,
    sheet: str | None,
    name_col: str | None,
    scaler: NumericScaler,
) -> list[tuple[ChunkMeta, np.ndarray]]:
    """Per-row z-scored numeric vectors. No text body — these chunks contribute
    only to the numeric ANN index. canonical_id is set when name_col is known
    so exact-match retrieval can fan out to the matching numeric_vector."""
    out: list[tuple[ChunkMeta, np.ndarray]] = []
    if not scaler.columns:
        return out
    for i in range(len(df)):
        row = df.iloc[i]
        canon = None
        if name_col and name_col in df.columns:
            raw = row.get(name_col)
            if pd.notna(raw):
                canon = normalize_name(str(raw))
        vec = scaler.transform_row(row)
        meta = make_meta(
            "numeric_vector",
            file=file,
            file_id=file_id,
            sheet=sheet,
            row_number=i,
            canonical_id=canon,
            numeric_columns=list(scaler.columns),
        )
        out.append((meta, vec))
    return out
