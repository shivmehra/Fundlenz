from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.id.canonical import AliasMap
from app.index.inverted import InvertedIndex
from app.index.metadata_store import MetaStore
from app.index.numeric_ann import NumericANN
from app.index.text_ann import TextANN
from app.rag.numeric_scaler import NumericScaler
from app.rag.schema import ChunkMeta


@dataclass
class IngestItem:
    """Internal record handed to CompositeIndex.add_chunks(). Either text or
    numeric_vec must be set (or both, but in practice text/numeric are emitted
    by separate chunkers)."""
    meta: ChunkMeta
    text: str | None = None
    text_vec: np.ndarray | None = None       # 1-D float32, L2-normalized
    numeric_vec: np.ndarray | None = None    # 1-D float32
    inverted_keys: list[tuple[str, str, str]] = field(default_factory=list)
    # inverted_keys: list of (kind, key, value_or_empty) where kind ∈ {"id","cell","enum"}.
    # For "id" the key is the canonical_id, value is "".
    # For "cell" the key is the column name, value is the cell value.
    # For "enum" the key is the column name, value is "".


class CompositeIndex:
    """Orchestrates the four persistent sub-stores: text ANN, per-file numeric
    ANN, inverted exact-match, SQLite metadata. Plus the user-editable AliasMap
    and per-file NumericScaler artifacts."""

    def __init__(self, root: Path, *, text_dim: int = 384, enable_numeric_ann: bool = False) -> None:
        self.root = root
        self.text_dim = text_dim
        self.enable_numeric_ann = enable_numeric_ann

        self.text = TextANN(root, dim=text_dim)
        self.numeric_by_file: dict[str, NumericANN] = {}
        self.scaler_by_file: dict[str, NumericScaler] = {}
        self.inverted = InvertedIndex(root / "inverted.json")
        self.meta = MetaStore(root / "metadata.sqlite")
        self.aliases = AliasMap(root / "aliases.json")

    # ---------- lifecycle ----------

    def load_or_init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.text.load_or_init()
        self.inverted.load_or_init()
        self.meta.init_schema()
        self.aliases.load()

        # Load any persisted scalers and (if enabled) their numeric indices.
        for sp in self.root.glob("scaler_*.json"):
            file_id = sp.stem[len("scaler_") :]
            scaler = NumericScaler()
            scaler.load(sp)
            self.scaler_by_file[file_id] = scaler
            if self.enable_numeric_ann:
                ann = NumericANN(file_id, dim=len(scaler.columns))
                if ann.load(self.root):
                    self.numeric_by_file[file_id] = ann

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.text.save()
        self.inverted.save()
        self.aliases.save()
        for fid, scaler in self.scaler_by_file.items():
            scaler.save(self.root / f"scaler_{fid}.json")
        if self.enable_numeric_ann:
            for ann in self.numeric_by_file.values():
                ann.save(self.root)

    # ---------- ingest ----------

    def register_scaler(self, file_id: str, scaler: NumericScaler) -> None:
        self.scaler_by_file[file_id] = scaler

    def add_chunks(self, items: list[IngestItem]) -> None:
        if not items:
            return

        # 1) metadata first (single transaction)
        self.meta.upsert_many([it.meta for it in items])

        # 2) text ANN — collect items with text vectors
        text_vecs: list[np.ndarray] = []
        text_ids: list[str] = []
        for it in items:
            if it.text_vec is not None:
                text_vecs.append(it.text_vec)
                text_ids.append(it.meta["chunk_id"])
        if text_vecs:
            mat = np.vstack(text_vecs).astype("float32")
            self.text.add(mat, text_ids)

        # 3) numeric ANN — per file_id, only when enabled
        if self.enable_numeric_ann:
            by_file: dict[str, list[tuple[str, np.ndarray]]] = {}
            for it in items:
                if it.numeric_vec is not None and "file_id" in it.meta:
                    by_file.setdefault(it.meta["file_id"], []).append(  # type: ignore[arg-type]
                        (it.meta["chunk_id"], it.numeric_vec)
                    )
            for file_id, pairs in by_file.items():
                ann = self.numeric_by_file.get(file_id)
                if ann is None:
                    dim = pairs[0][1].shape[0]
                    ann = NumericANN(file_id, dim=dim)
                    ann.init_empty()
                    self.numeric_by_file[file_id] = ann
                vecs = np.vstack([v for _, v in pairs]).astype("float32")
                cids = [cid for cid, _ in pairs]
                ann.add(vecs, cids)

        # 4) inverted index
        for it in items:
            cid = it.meta["chunk_id"]
            for kind, key, value in it.inverted_keys:
                if kind == "id":
                    self.inverted.add_id(key, cid)
                elif kind == "cell":
                    self.inverted.add_cell(key, value, cid)
                elif kind == "enum":
                    self.inverted.add_enum(key, cid)

    # ---------- query ----------

    def text_search(self, qv: np.ndarray, k: int) -> list[tuple[str, float]]:
        return self.text.search(qv, k)

    def numeric_search(
        self, file_id: str, qv: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        if not self.enable_numeric_ann:
            return []
        ann = self.numeric_by_file.get(file_id)
        if ann is None:
            return []
        return ann.search(qv, k)

    # ---------- introspection ----------

    def stats(self) -> dict:
        return {
            "text_chunks": self.text.total(),
            "metadata_count": self.meta.count(),
            "by_type": self.meta.count_by_type(),
            "inverted_postings": self.inverted.total_postings(),
            "files": self.meta.list_files(),
            "scalers": list(self.scaler_by_file.keys()),
            "numeric_indices": list(self.numeric_by_file.keys()) if self.enable_numeric_ann else [],
            "enable_numeric_ann": self.enable_numeric_ann,
        }

    # ---------- mutation ----------

    def delete_by_file(self, file: str) -> int:
        deleted_chunk_ids = self.meta.delete_by_file(file)
        if not deleted_chunk_ids:
            return 0
        deleted_set = set(deleted_chunk_ids)

        # Rebuild text ANN keeping only non-deleted chunks. We can do this
        # without re-embedding because vectors live alongside the index in
        # FAISS; we have to re-walk the FAISS array. Easiest path: ask FAISS
        # to reconstruct each kept vector.
        if self.text.index is not None and self.text.index.ntotal:
            keep_ids: list[str] = []
            keep_vecs: list[np.ndarray] = []
            for i, cid in enumerate(self.text.chunk_ids):
                if cid in deleted_set:
                    continue
                keep_ids.append(cid)
                keep_vecs.append(self.text.index.reconstruct(i))
            mat = np.vstack(keep_vecs).astype("float32") if keep_vecs else np.zeros(
                (0, self.text_dim), dtype="float32"
            )
            self.text.rebuild_keeping(set(keep_ids), mat)
            # rebuild_keeping uses set ordering — rewrite chunk_ids in original order.
            self.text.chunk_ids = keep_ids

        # Inverted index — remove postings referencing deleted chunks.
        self.inverted.remove_chunks(deleted_set)

        # Numeric ANN — drop the per-file index entirely (file is going away)
        # and remove its scaler.
        # First find file_ids to drop from the deleted metadata.
        # We don't have payloads anymore (they were removed); instead, infer
        # from the file_id space: any scaler whose file_id is no longer
        # present in the metadata store gets dropped.
        live_file_ids = set(self.meta.list_file_ids())
        for fid in list(self.scaler_by_file.keys()):
            if fid not in live_file_ids:
                self.scaler_by_file.pop(fid, None)
                # delete persisted scaler json
                sp = self.root / f"scaler_{fid}.json"
                if sp.exists():
                    sp.unlink()
                if self.enable_numeric_ann:
                    self.numeric_by_file.pop(fid, None)
                    for ext in (".faiss", "_ids.npy"):
                        p = self.root / f"numeric_{fid}{ext}"
                        if p.exists():
                            p.unlink()

        return len(deleted_chunk_ids)
