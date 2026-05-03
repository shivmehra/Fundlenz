import pickle
from pathlib import Path

import faiss
import numpy as np

_EMBED_DIM = 384  # multi-qa-MiniLM-L6-cos-v1 — hardcoded so startup never loads the model


class VectorStore:
    INDEX_FILE = "index.faiss"
    META_FILE = "metadata.pkl"

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []
        self._vectors: np.ndarray | None = None  # kept so delete can rebuild without re-embedding

    def load_or_init(self) -> None:
        idx_path = self.index_dir / self.INDEX_FILE
        meta_path = self.index_dir / self.META_FILE
        if idx_path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(idx_path))
            with meta_path.open("rb") as f:
                data = pickle.load(f)
            self.metadata = data["metadata"]
            self._vectors = data.get("vectors")
        else:
            self.index = faiss.IndexFlatIP(_EMBED_DIM)
            self.metadata = []
            self._vectors = None

    def save(self) -> None:
        assert self.index is not None
        faiss.write_index(self.index, str(self.index_dir / self.INDEX_FILE))
        with (self.index_dir / self.META_FILE).open("wb") as f:
            pickle.dump({"metadata": self.metadata, "vectors": self._vectors}, f)

    def add(self, vectors: np.ndarray, metas: list[dict]) -> None:
        assert self.index is not None
        assert vectors.shape[0] == len(metas)
        self.index.add(vectors)
        self.metadata.extend(metas)
        self._vectors = vectors.copy() if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, query: np.ndarray, k: int) -> list[dict]:
        assert self.index is not None
        if self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(query, k)
        results: list[dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)
            results.append(meta)
        return results

    def list_filenames(self) -> list[str]:
        seen: list[str] = []
        for m in self.metadata:
            fn = m["filename"]
            if fn not in seen:
                seen.append(fn)
        return seen

    def delete_by_filename(self, filename: str) -> bool:
        keep = [i for i, m in enumerate(self.metadata) if m["filename"] != filename]
        if len(keep) == len(self.metadata):
            return False
        self.metadata = [self.metadata[i] for i in keep]
        self.index = faiss.IndexFlatIP(_EMBED_DIM)
        if keep and self._vectors is not None:
            kept = self._vectors[keep]
            self.index.add(kept)
            self._vectors = kept
        else:
            self._vectors = None
        return True
