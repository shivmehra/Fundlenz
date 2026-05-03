"""Clean-cutover migration to the v2 multi-index pipeline.

  python -m scripts.migrate_v2

Wipes the existing FAISS+pickle index, re-initializes the new CompositeIndex
(text ANN + numeric ANN + inverted exact-match + SQLite metadata + alias map),
and re-ingests every file under data/uploads/.

Run this with the backend stopped — SQLite on Windows can fail on writes
when another process holds the file open."""

import sys
from collections import Counter
from pathlib import Path

# Make `app` importable when invoked from the backend dir.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


from app import state  # noqa: E402
from app.config import settings  # noqa: E402
from app.ingest import ingest_file  # noqa: E402


def _wipe_index_dir() -> None:
    idx_dir = settings.index_dir
    if not idx_dir.exists():
        idx_dir.mkdir(parents=True, exist_ok=True)
        return
    for p in idx_dir.iterdir():
        if p.name == ".gitkeep":
            continue
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            for child in p.iterdir():
                child.unlink()
            p.rmdir()


def main() -> int:
    print(f"index dir : {settings.index_dir}")
    print(f"upload dir: {settings.upload_dir}")

    files = sorted(p for p in settings.upload_dir.iterdir() if p.is_file() and p.suffix.lower() in (".pdf", ".docx", ".csv", ".xlsx", ".xls"))
    if not files:
        print("no files to ingest under data/uploads/")
        return 1

    print(f"\nWiping {settings.index_dir} ...")
    _wipe_index_dir()

    print("Initializing CompositeIndex ...")
    state.composite.load_or_init()
    state.composite.save()

    total_chunks = 0
    type_counts: Counter[str] = Counter()
    for f in files:
        try:
            res = ingest_file(f, f.name)
            chunks = res["chunks"]
            total_chunks += chunks
            # Pull per-file type breakdown from the composite metadata store
            metas = state.composite.meta.by_file(f.name)
            file_types = Counter(m["chunk_type"] for m in metas)
            type_counts.update(file_types)
            breakdown = ", ".join(f"{k}={v}" for k, v in sorted(file_types.items()))
            print(f"  OK  {f.name:50s}  chunks={chunks:5d}  ({breakdown})")
        except Exception as e:
            print(f"  FAIL {f.name}: {type(e).__name__}: {e}")

    print(f"\nTotal chunks: {total_chunks}")
    print("Type breakdown:")
    for t, n in sorted(type_counts.items()):
        print(f"  {t:20s} {n}")
    print("\nDone. Restart uvicorn to pick up the new index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
