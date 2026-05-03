import json
from collections import defaultdict
from pathlib import Path

from app.id.canonical import normalize_name


class InvertedIndex:
    """Deterministic exact-match. Three keyspaces over a flat dict[str, list[str]]:

      - id:<canonical_id>          -> [chunk_id, ...]      (entity hits)
      - cell:<col_lower>=<n_value> -> [chunk_id, ...]      (cell match in a row)
      - enum:<col_lower>           -> [chunk_id, ...]      (enumeration chunk for a column)

    Persisted as a single JSON file. Small enough that an O(MB) personal
    dataset round-trips in milliseconds. Postings preserve insertion order;
    duplicates within the same posting list are allowed (the retrieval layer
    de-dupes by chunk_id anyway)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.postings: dict[str, list[str]] = defaultdict(list)

    def load_or_init(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.postings = defaultdict(list, {k: list(v) for k, v in data.items()})
        else:
            self.postings = defaultdict(list)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(self.postings), sort_keys=True), encoding="utf-8")

    def add_id(self, canonical_id: str, chunk_id: str) -> None:
        if not canonical_id:
            return
        self.postings[f"id:{canonical_id}"].append(chunk_id)

    def add_cell(self, col: str, value: str, chunk_id: str) -> None:
        norm = normalize_name(str(value))
        if not norm:
            return
        self.postings[f"cell:{col.lower()}={norm}"].append(chunk_id)

    def add_enum(self, col: str, chunk_id: str) -> None:
        self.postings[f"enum:{col.lower()}"].append(chunk_id)

    def lookup_id(self, canonical_id: str) -> list[str]:
        return list(self.postings.get(f"id:{canonical_id}", []))

    def lookup_cell(self, col: str, value: str) -> list[str]:
        norm = normalize_name(str(value))
        return list(self.postings.get(f"cell:{col.lower()}={norm}", []))

    def lookup_enum(self, col: str) -> list[str]:
        return list(self.postings.get(f"enum:{col.lower()}", []))

    def total_postings(self) -> int:
        return sum(len(v) for v in self.postings.values())

    def remove_chunks(self, chunk_ids: set[str]) -> None:
        if not chunk_ids:
            return
        empty_keys: list[str] = []
        for k, v in self.postings.items():
            kept = [c for c in v if c not in chunk_ids]
            if kept:
                self.postings[k] = kept
            else:
                empty_keys.append(k)
        for k in empty_keys:
            self.postings.pop(k, None)
