import json
import sqlite3
from pathlib import Path

from app.rag.schema import ChunkMeta


class MetaStore:
    """SQLite-backed chunk_id -> ChunkMeta mapping. The 8 mandatory fields are
    stored as columns (with indexes on canonical_id, file, chunk_type) and the
    full ChunkMeta is duplicated in `payload` JSON for round-trip fidelity.

    SQLite is stdlib — no new dependency. Replaces the v1 metadata.pkl, which
    coupled metadata position to vector position."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False is safe here because FastAPI dispatches
        # blocking IO via asyncio.to_thread (see main.py); we serialize writes
        # through a single connection per process.
        return sqlite3.connect(str(self.path), check_same_thread=False)

    def init_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = self._connect()
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                file TEXT NOT NULL,
                sheet TEXT,
                chunk_type TEXT NOT NULL,
                canonical_id TEXT,
                row_number INTEGER,
                ingestion_time TEXT NOT NULL,
                version INTEGER NOT NULL,
                file_id TEXT,
                payload TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_canonical ON chunks(canonical_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)")

        # Per-file row counts for the sidebar stats card. Persisted here (not in
        # state.dataframes_by_file_id) so the count survives a backend restart
        # even though the DataFrames don't.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS file_stats (
                file_id        TEXT PRIMARY KEY,
                filename       TEXT NOT NULL,
                row_count      INTEGER NOT NULL,
                ingestion_time TEXT NOT NULL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_stats_filename ON file_stats(filename)"
        )
        self.conn.commit()

    def upsert_many(self, metas: list[ChunkMeta]) -> None:
        assert self.conn is not None
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO chunks
              (chunk_id, file, sheet, chunk_type, canonical_id, row_number,
               ingestion_time, version, file_id, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    m["chunk_id"],
                    m["file"],
                    m.get("sheet"),
                    m["chunk_type"],
                    m.get("canonical_id"),
                    m.get("row_number"),
                    m["ingestion_time"],
                    m["version"],
                    m.get("file_id"),
                    json.dumps(m, default=str),
                )
                for m in metas
            ],
        )
        self.conn.commit()

    def get(self, chunk_id: str) -> ChunkMeta | None:
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT payload FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def get_many(self, chunk_ids: list[str]) -> list[ChunkMeta]:
        assert self.conn is not None
        if not chunk_ids:
            return []
        # SQLite has a 999-param default limit; chunk through if needed.
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"SELECT chunk_id, payload FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        by_id = {r[0]: json.loads(r[1]) for r in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def by_canonical(self, canonical_id: str) -> list[ChunkMeta]:
        assert self.conn is not None
        rows = self.conn.execute(
            "SELECT payload FROM chunks WHERE canonical_id = ? ORDER BY chunk_id",
            (canonical_id,),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def by_file(self, file: str) -> list[ChunkMeta]:
        assert self.conn is not None
        rows = self.conn.execute(
            "SELECT payload FROM chunks WHERE file = ? ORDER BY chunk_id", (file,)
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def list_files(self) -> list[str]:
        assert self.conn is not None
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT file FROM chunks ORDER BY file"
        ).fetchall()]

    def list_file_ids(self) -> list[str]:
        assert self.conn is not None
        return [
            r[0] for r in self.conn.execute(
                "SELECT DISTINCT file_id FROM chunks WHERE file_id IS NOT NULL ORDER BY file_id"
            ).fetchall()
        ]

    def count(self) -> int:
        assert self.conn is not None
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def count_by_type(self) -> dict[str, int]:
        assert self.conn is not None
        return dict(self.conn.execute(
            "SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type"
        ).fetchall())

    def delete_by_file(self, file: str) -> list[str]:
        assert self.conn is not None
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT chunk_id FROM chunks WHERE file = ?", (file,)
        ).fetchall()
        chunk_ids = [r[0] for r in rows]
        cur.execute("DELETE FROM chunks WHERE file = ?", (file,))
        self.conn.commit()
        return chunk_ids

    # ---------- file_stats ----------

    def upsert_file_stat(
        self,
        file_id: str,
        filename: str,
        row_count: int,
        ingestion_time: str,
    ) -> None:
        assert self.conn is not None
        self.conn.execute(
            """
            INSERT OR REPLACE INTO file_stats
              (file_id, filename, row_count, ingestion_time)
            VALUES (?, ?, ?, ?)
            """,
            (file_id, filename, row_count, ingestion_time),
        )
        self.conn.commit()

    def total_rows(self) -> int:
        assert self.conn is not None
        return self.conn.execute(
            "SELECT COALESCE(SUM(row_count), 0) FROM file_stats"
        ).fetchone()[0]

    def delete_file_stats_by_filename(self, filename: str) -> None:
        assert self.conn is not None
        self.conn.execute("DELETE FROM file_stats WHERE filename = ?", (filename,))
        self.conn.commit()

    def list_file_stats(self) -> list[dict]:
        assert self.conn is not None
        rows = self.conn.execute(
            "SELECT file_id, filename, row_count, ingestion_time "
            "FROM file_stats ORDER BY filename"
        ).fetchall()
        return [
            {
                "file_id": r[0],
                "filename": r[1],
                "row_count": r[2],
                "ingestion_time": r[3],
            }
            for r in rows
        ]

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
