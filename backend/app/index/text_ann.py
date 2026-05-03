from pathlib import Path

import faiss
import numpy as np


class TextANN:
    """FAISS IndexFlatIP over L2-normalized text vectors, keyed by chunk_id.

    Critical change from the v1 VectorStore: we store a parallel `chunk_ids`
    array next to the index so vector position is decoupled from metadata
    position. Deletion rebuilds the index from a kept set; metadata schema
    changes don't touch the FAISS file."""

    INDEX_FILE = "text.faiss"
    IDS_FILE = "text_ids.npy"

    def __init__(self, dir_: Path, dim: int = 384) -> None:
        self.dir = dir_
        self.dim = dim
        self.index: faiss.Index | None = None
        self.chunk_ids: list[str] = []

    def load_or_init(self) -> None:
        idx_path = self.dir / self.INDEX_FILE
        ids_path = self.dir / self.IDS_FILE
        if idx_path.exists() and ids_path.exists():
            self.index = faiss.read_index(str(idx_path))
            self.chunk_ids = list(np.load(ids_path, allow_pickle=True))
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.chunk_ids = []

    def save(self) -> None:
        assert self.index is not None
        self.dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.dir / self.INDEX_FILE))
        np.save(self.dir / self.IDS_FILE, np.array(self.chunk_ids, dtype=object), allow_pickle=True)

    def add(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        assert self.index is not None
        assert vectors.shape[0] == len(chunk_ids)
        if vectors.size == 0:
            return
        self.index.add(vectors)
        self.chunk_ids.extend(chunk_ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        assert self.index is not None
        if self.index.ntotal == 0 or k <= 0:
            return []
        scores, ids = self.index.search(query, min(k, self.index.ntotal))
        out: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            out.append((self.chunk_ids[idx], float(score)))
        return out

    def total(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def rebuild_keeping(self, keep_chunk_ids: set[str], keep_vectors: np.ndarray) -> None:
        """Replace contents with a known-vectors / known-ids slice. Used after
        delete_by_file rebuilds. keep_vectors must align with the post-filter
        order of keep_chunk_ids."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunk_ids = []
        if keep_chunk_ids and keep_vectors.size > 0:
            self.index.add(keep_vectors)
            self.chunk_ids = list(keep_chunk_ids)
