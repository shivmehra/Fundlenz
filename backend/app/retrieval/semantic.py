import numpy as np

from app.index.composite import CompositeIndex
from app.rag.embedder import embed


def text_search(query: str, idx: CompositeIndex, k: int) -> list[tuple[str, float]]:
    qv = embed([query])
    return idx.text_search(qv, k)


def numeric_search_all_files(
    qvec_by_file: dict[str, np.ndarray],
    idx: CompositeIndex,
    k: int,
) -> list[tuple[str, float]]:
    """Run numeric ANN against each per-file index whose schema matches the
    query vector dimension. Caller is responsible for building qvec_by_file
    using the appropriate scaler."""
    if not idx.enable_numeric_ann or not qvec_by_file:
        return []
    out: list[tuple[str, float]] = []
    for file_id, qv in qvec_by_file.items():
        out.extend(idx.numeric_search(file_id, qv, k))
    return out
