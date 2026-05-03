from pathlib import Path

import faiss
import numpy as np


class NumericANN:
    """One IndexFlatL2 per file_id over z-scored numeric vectors. Different
    files may have different schemas (column counts), so we don't pool them
    into a single index. Gated at the composite level by config.enable_numeric_ann."""

    def __init__(self, file_id: str, dim: int) -> None:
        self.file_id = file_id
        self.dim = dim
        self.index: faiss.Index | None = None
        self.chunk_ids: list[str] = []

    def init_empty(self) -> None:
        self.index = faiss.IndexFlatL2(self.dim)
        self.chunk_ids = []

    def add(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        assert self.index is not None
        assert vectors.shape[0] == len(chunk_ids)
        if vectors.size == 0:
            return
        self.index.add(vectors.astype("float32"))
        self.chunk_ids.extend(chunk_ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        assert self.index is not None
        if self.index.ntotal == 0 or k <= 0:
            return []
        d2, ids = self.index.search(query.astype("float32"), min(k, self.index.ntotal))
        out: list[tuple[str, float]] = []
        for dist_sq, idx in zip(d2[0], ids[0]):
            if idx == -1:
                continue
            # Return Euclidean distance, not squared.
            out.append((self.chunk_ids[idx], float(np.sqrt(max(dist_sq, 0.0)))))
        return out

    def save(self, dir_: Path) -> None:
        assert self.index is not None
        dir_.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_ / f"numeric_{self.file_id}.faiss"))
        np.save(dir_ / f"numeric_{self.file_id}_ids.npy",
                np.array(self.chunk_ids, dtype=object),
                allow_pickle=True)

    def load(self, dir_: Path) -> bool:
        idx_path = dir_ / f"numeric_{self.file_id}.faiss"
        ids_path = dir_ / f"numeric_{self.file_id}_ids.npy"
        if not (idx_path.exists() and ids_path.exists()):
            return False
        self.index = faiss.read_index(str(idx_path))
        self.chunk_ids = list(np.load(ids_path, allow_pickle=True))
        self.dim = self.index.d
        return True
