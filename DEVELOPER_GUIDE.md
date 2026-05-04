# Fundlenz — Developer Guide

A local RAG chatbot that ingests fund documents (PDFs, Word, CSV, Excel) and answers questions using a multi-index retrieval pipeline (inverted exact-match + text ANN + numeric range), deterministic ranking, optional pandas-driven analysis, and Plotly charts. Everything runs locally — no cloud services required.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Project Layout](#project-layout)
4. [Data Flow](#data-flow)
5. [Chunk Schema](#chunk-schema)
6. [Retrieval Pipeline Walkthrough](#retrieval-pipeline-walkthrough)
7. [Backend — Module Reference](#backend--module-reference)
8. [Frontend — Module Reference](#frontend--module-reference)
9. [Endpoints](#endpoints)
10. [Reranking Knobs](#reranking-knobs)
11. [Config Flags Reference](#config-flags-reference)
12. [Key Design Decisions](#key-design-decisions)
13. [Pitfalls & Constraints](#pitfalls--constraints)
14. [Adding Features](#adding-features)
15. [Migration](#migration)
16. [Testing](#testing)

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) running locally with the default model pulled:
  ```
  ollama pull qwen2.5:7b
  ```
  Qwen 2.5 7B is the recommended default — its tool-calling discipline is far better than Mistral 7B's, which materially reduces hallucinated tool arguments and "describes the call as code instead of invoking it" failures. Mistral 7B and Llama 3.x are also supported via `OLLAMA_MODEL`.

### One-time setup

```bash
# Backend
cd backend
python -m venv ../.venv
../.venv/Scripts/activate      # Windows
pip install -r requirements.txt

# Copy env and optionally edit values
copy .env.example .env

# Frontend
cd ../frontend
npm install
```

### Start the app

```bat
start.bat
```

Opens two cmd windows: backend on port 8000, frontend on port 5173. Open `http://localhost:5173` in a browser.

---

## Architecture Overview

```
                INGEST                                   RETRIEVAL
+----------------------+   +-------------------+      +-------------------------+
| parsers/             |   | chunkers/         |      | retrieval/router.py     |
|   pdf.py             |   |   text_chunker    |      |   intent classifier     |
|   docx.py            |-->|   tabular_chunker |      +------------+------------+
|   tabular.py         |   |   entity_chunker  |                   |
+----------+-----------+   |   timewindow      |                   v
           |               |   numeric_encoder |      +-------------------------+
           |               +---------+---------+      | retrieval/exact.py      |
           |                         |                |   inverted lookup       |
           v                         v                +------------+------------+
+----------------------+   +-------------------+                   |
| extract_pdf_tables   |   | id/canonical.py   |                   v
|     _as_csv          |   |   normalize_name  |      +-------------------------+
| (PDF tables → CSV    |   |   AliasMap        |      | retrieval/semantic.py   |
|  → tabular pipeline) |   +---------+---------+      |   text ANN              |
+----------------------+             |                |   numeric ANN (gated)   |
                                     v                +------------+------------+
                +-------------------------------+                  |
                |     CompositeIndex            |                  v
                |  index/text_ann.py            |     +-------------------------+
                |  index/numeric_ann.py         |     | retrieval/rerank.py     |
                |  index/inverted.py            |     |   linear score          |
                |  index/metadata_store.py      |     +------------+------------+
                |                               |                  |
                |  Persistence:                 |                  v
                |    text.faiss + ids.npy       |     +-------------------------+
                |    numeric_<fid>.faiss        |     | retrieval/cross_encoder |
                |    inverted.json              |     |   semantic-tier rerank  |
                |    metadata.sqlite            |     +------------+------------+
                |    scaler_<fid>.json          |                  |
                |    aliases.json               |                  v
                +-------------------------------+     +-------------------------+
                                                      | LLM synthesis (Ollama)  |
                                                      |   provenance + cites    |
                                                      +-------------------------+
```

**Stack**

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript + Vite |
| Backend | FastAPI + uvicorn |
| Embeddings | SentenceTransformers `multi-qa-MiniLM-L6-cos-v1` (asymmetric, 384-dim) |
| Vector DB | FAISS `IndexFlatIP` (cosine via inner product on L2-normalized vectors) |
| Inverted index | JSON file (`inverted.json`) — three keyspaces (`id:`, `cell:`, `enum:`) |
| Metadata store | SQLite (`metadata.sqlite`) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional 2nd-stage rerank) |
| LLM | Qwen 2.5 7B via Ollama (configurable via `OLLAMA_MODEL`) |
| PDF parsing | pdfplumber (preserves tables; tables also extracted to CSV) |
| Tabular | pandas |
| Streaming | Server-Sent Events (SSE) |

---

## Project Layout

```
Fundlenz/
├── start.bat                   # Launch both servers
├── DEVELOPER_GUIDE.md          # This file
├── backend/
│   ├── .env                    # Runtime config (gitignored)
│   ├── .env.example            # Template — copy to .env
│   ├── requirements.txt
│   ├── data/
│   │   ├── indexes/            # CompositeIndex artifacts (persisted)
│   │   │   ├── text.faiss + text_ids.npy
│   │   │   ├── numeric_<file_id>.faiss + _ids.npy   (when enable_numeric_ann)
│   │   │   ├── inverted.json
│   │   │   ├── metadata.sqlite
│   │   │   ├── scaler_<file_id>.json
│   │   │   └── aliases.json
│   │   └── uploads/            # uploaded files + auto-generated CSVs from PDF tables
│   ├── conftest.py
│   ├── notebooks/
│   │   └── demo_v2.ipynb       # Walks each retrieval intent end-to-end
│   ├── scripts/
│   │   └── migrate_v2.py       # Wipes data/indexes/* and re-ingests data/uploads/*
│   ├── tests/                  # pytest suite (203 tests)
│   └── app/
│       ├── main.py             # FastAPI app, routes, threshold bypass, confidence tiers
│       ├── config.py           # Pydantic Settings
│       ├── state.py            # CompositeIndex + dataframes + chat history
│       ├── ingest.py           # Parser → chunker dispatch
│       ├── text.py             # Universal Unicode + whitespace normalization
│       ├── parsers/
│       │   ├── pdf.py          # pdfplumber — text + tables_md + raw tables;
│       │   │                   #   extract_pdf_tables_as_csv writes CSVs
│       │   ├── docx.py         # python-docx — heading hierarchy + tables → markdown
│       │   └── tabular.py      # CSV/Excel — header sniff, multi-sheet, cleaning
│       ├── chunkers/
│       │   ├── common.py            # make_meta() — fills mandatory ChunkMeta fields
│       │   ├── text_chunker.py      # Sliding-window text + table chunks
│       │   ├── tabular_chunker.py   # synopsis + tabular_summary + enumeration + row
│       │   ├── entity_chunker.py    # One chunk per canonical_id
│       │   ├── numeric_encoder.py   # numeric_vector chunks (no text body)
│       │   └── timewindow_chunker.py# detect_timeseries + 30/90/365d windows
│       ├── id/
│       │   └── canonical.py    # normalize_name() + AliasMap
│       ├── index/
│       │   ├── composite.py        # CompositeIndex + IngestItem
│       │   ├── text_ann.py         # FAISS IndexFlatIP, chunk_id-keyed
│       │   ├── numeric_ann.py      # Per-file IndexFlatL2 (gated)
│       │   ├── inverted.py         # id: / cell: / enum: keyspaces
│       │   └── metadata_store.py   # SQLite (chunks table, indexes on canonical_id/file/type)
│       ├── rag/
│       │   ├── schema.py            # ChunkMeta TypedDict + SCHEMA_VERSION
│       │   ├── embedder.py          # SentenceTransformer + chunk_text()
│       │   ├── numeric_scaler.py    # Per-column z-score, NaN-safe
│       │   └── retriever.py         # Thin shim over orchestrator.retrieve_v2
│       ├── retrieval/
│       │   ├── router.py            # Rule-based intent classifier (Plan dataclass)
│       │   ├── exact.py             # resolve_canonical_id + extract_cell_predicates
│       │   ├── semantic.py          # Text + numeric ANN dispatch
│       │   ├── rerank.py            # Deterministic linear scoring formula
│       │   ├── cross_encoder.py     # Stage-2 CE reranker (lazy-loaded)
│       │   └── orchestrator.py      # retrieve_v2 + format_context_v2
│       ├── llm/
│       │   ├── ollama_client.py     # build_messages, rewrite_query, stream_chat
│       │   └── tools.py             # compute_metric + query_table schemas
│       └── analysis/
│           ├── metrics.py           # compute() — top/mean/sum/count/min/max/trend
│           ├── query.py             # query_table — filter/sort/select/limit
│           ├── charts.py            # Plotly JSON spec generator
│           └── column_match.py      # Token-overlap column resolver (drives threshold bypass)
└── frontend/
    ├── vite.config.ts          # Proxies /api/* to 127.0.0.1:8000 (IPv4 only)
    └── src/
        ├── main.tsx
        ├── App.tsx             # Sidebar + chat layout, hamburger drawer for mobile
        ├── styles.css          # CSS variables, grid layout, table rendering rules
        ├── types.ts            # Source / ChartPayload / Message / Confidence
        ├── api/
        │   └── client.ts       # uploadFile, getDocuments, deleteDocument, streamChat
        └── components/
            ├── ChatWindow.tsx     # Message list + composer + mode pills
            ├── MessageBubble.tsx  # Single message (text + sources + chart + confidence)
            ├── DocumentList.tsx   # Sidebar: indexed files + delete
            ├── FileUpload.tsx     # Sidebar: file picker + status (incl. PDF table extracts)
            └── PlotlyChart.tsx    # Renders ChartPayload from compute_metric
```

---

## Data Flow

### 1. Document Ingestion (`POST /ingest`)

```
File upload
    │
    ▼
ingest.py::ingest_file(path, original_filename)
    │
    ├─ .pdf  ──► parsers/pdf.py::parse_pdf()
    │                ├─ pdfplumber: per-page text + tables (lines strategy first,
    │                │   text-alignment fallback). _is_valid_table rejects
    │                │   <2 rows / <2 cols / >70% empty.
    │                ├─ Repeating boilerplate stripped (lines on ≥3 distinct pages)
    │                ├─ Page-number patterns ("Page 3 of 12", "5 / 12") stripped
    │                ├─ text.normalize() (Unicode + dehyphenate + whitespace)
    │                └─ Returns [{page, text, tables_md, tables}, ...]
    │
    ├─ .docx ──► parsers/docx.py::parse_docx()
    │                ├─ Walks doc.element.body in document order so paragraphs
    │                │   and tables interleave correctly
    │                ├─ Heading 1..9 styles → markdown # / ## / ###
    │                ├─ Tables → markdown pipe-tables
    │                └─ text.normalize() applied at the end
    │
    └─ .csv/.xlsx ──► parsers/tabular.py::parse_tabular()
                         ├─ List of sheet records (CSV → 1; multi-sheet Excel → N)
                         ├─ Header-row sniffing (Unnamed:N detection, scans rows 1..4)
                         ├─ _clean(): strip, null sentinels, drop-empty,
                         │   currency/percent → float, date hint → datetime
                         │   (80% parse threshold)
                         └─ synopsis() + _summarize() for chunk bodies

    │
    ▼
chunker dispatch (ingest.py::_ingest_tabular_sheet for tabular files)
    │
    ├─ build_text_chunks         (PDF/DOCX only)
    ├─ build_table_chunk         (PDF: one chunk per detected table_md)
    ├─ build_tabular_chunks      (synopsis + tabular_summary + enumeration + row)
    ├─ build_entity_chunks       (one per canonical_id, attribute-aggregated)
    ├─ build_numeric_vectors     (per-row z-scored numeric vectors)
    └─ build_time_windows        (only when detect_timeseries finds multi-date data)
    │
    ▼
embed all text-bearing chunks in a single batch
    │
    ▼
state.composite.add_chunks(items)
    ├─ meta.upsert_many              (SQLite, single transaction)
    ├─ text_ann.add(vecs, chunk_ids) (FAISS IndexFlatIP)
    ├─ numeric_ann.add(...)          (per file_id; only when enable_numeric_ann)
    └─ inverted index posts          (id: / cell: / enum:)
    │
    ▼
state.composite.save() → text.faiss, inverted.json, metadata.sqlite, scaler_*.json
```

**PDF table post-processing** (added in `main.py::ingest`):

After the PDF itself is ingested, `parsers/pdf.py::extract_pdf_tables_as_csv` runs over every detected table and writes a CSV per table to `data/uploads/{pdf_stem}_page{N}_table{M}.csv`. Each CSV is then ingested independently via `ingest_file(csv_path, csv_path.name)` so the table data flows into the tabular pipeline (synopsis/enumeration/row/entity/numeric_vector chunks) and becomes queryable through `query_table` and `compute_metric`.

The table-to-CSV writer normalizes the header row first (`_normalize_table_for_csv`) in three steps:

1. **Strip sparse top rows.** PDF tables often have a row 0 that is just the entity name or a section label spanning the full width (one filled cell, the rest blank), with the real column headers in row 1. Up to 3 such rows are peeled off the top — the strip continues while row 0 is less-than-half filled AND row 1 has more filled cells than row 0. Without this, pandas reads the title row as the header, the real headers land as a string-typed data row, numeric coercion fails on the mixed-type column, and `compute_metric`/`query_table` are useless on the result.
2. **Conditionally fill blank header cells.** Only when the (possibly post-strip) row 0 is ≥50% filled, blanks become `col_1`, `col_2`, … placeholders. When row 0 is still sparse, blanks are left as-is so pandas auto-names them `Unnamed: N` and the existing `tabular.py` row-1-to-4 fallback can find the real header.
3. **De-duplicate names** so pandas doesn't silently collapse repeated columns. Degenerate tables (<2 rows / <2 cols / all empty) are skipped entirely.

The `/ingest` response surfaces what landed:

```json
{
  "filename": "report.pdf",
  "chunks": 87,
  "summary": "...",
  "extracted_tables": [
    {"filename": "report_page2_table1.csv", "chunks": 24},
    {"filename": "report_page5_table1.csv", "chunks": 16},
    {"filename": "report_page7_table1.csv", "chunks": 0,
     "error": "..."}
  ]
}
```

Per-CSV failures are caught individually so one bad table can't suppress the rest. The upload status in the UI reads e.g. `Indexed 87 chunks (+2 tables from PDF, 1 failed)`.

**Multi-sheet Excel** files become multiple logical files. `fund.xlsx` with sheets `Holdings` and `Performance` produces two file_ids, two `state.dataframes_by_file_id` entries, and two sidebar entries keyed `"fund.xlsx :: Holdings"` and `"fund.xlsx :: Performance"`. Single-sheet Excel and CSV files use the bare filename.

### 2. Chat Request (`POST /chat`)

```
{session_id, message, mode: "auto" | "chat" | "aggregate" | "query"}
    │
    ▼
ollama_client.py::rewrite_query(message, history)
    └─ folds prior turns into a self-contained search query
       (used for retrieval ONLY; original message still flows to the LLM)

    │
    ▼
router.py::classify_intent(message)
    └─ Plan {intent, raw_entity_phrases, threshold?, window?}
       Note: classified on the LITERAL message, not the rewritten one,
       so broad intents ("List all benchmarks") don't get hijacked into
       point-lookups by the rewriter. Entity phrases from the rewritten
       query are merged in afterwards for coreference resolution.

    │
    ▼
orchestrator.py::retrieve_v2(search_query, composite, k, plan)
    (full pipeline — see "Retrieval Pipeline Walkthrough" below)

    │
    ▼  back in main.py::event_stream()
    │
    ├─ yield "sources" event   (UI shows chunk cards with confidence badges)
    │
    ├─ Forced-mode short-circuit: if mode is "aggregate" or "query"
    │   and no tabular file has been ingested, yield a friendly error
    │   and stop before the LLM call.
    │
    ├─ Numeric-threshold deterministic bypass (mode != "chat"):
    │   if intent is numeric_threshold and a column resolves in any
    │   loaded tabular file, run pandas filtering directly across
    │   every matching file (aggregating results), yield a
    │   confidence: deterministic event, emit the answer as a
    │   token event, and return. The LLM is never called.
    │
    ▼
llm/ollama_client.py::build_messages(message, context, history, tabular_files)
    │
    ▼
main.py::_select_tools(message, mode)
    ├─ "chat"       → None (no tools)
    ├─ "aggregate"  → [compute_metric] iff a tabular file exists
    ├─ "query"      → [query_table]   iff a tabular file exists
    └─ "auto"       → keyword-gated (TOOLS or None)
    │
    ▼
stream_chat(messages, tools=...)
    │
    ├─ token chunks  ──► "event: token" (streamed live)
    └─ tool_call ───────► _run_compute_metric / _run_query_table
                              │
                              └─ pandas execution
                              └─ emit "event: chart" (compute_metric)
                              └─ emit "event: token" with markdown table
    │
    ▼
emit "event: confidence" with tier ∈ {deterministic, grounded, semantic}
    │
    ▼
emit "event: done"
```

The full SSE stream the browser sees:

```
event: sources
data: [{"filename":"...", "page":2, "score":0.87, "match":"semantic", ...}, ...]

event: confidence
data: {"tier":"grounded", "value":0.91, "reason":"strong source overlap (avg 91%)"}

event: token
data: The fund's expense ratio is 0.75% annually...

event: chart                 (only when compute_metric ran)
data: {"chart_spec": {...}, "filename": "...", "op": "..."}

event: done
data:
```

**Confidence tiers** (`main.py::_confidence_for_llm_path` + the threshold bypass path):

| Tier | When | Displayed |
|---|---|---|
| `deterministic` | A tool ran, the threshold bypass fired, or all top-3 sources are exact-id matches | Green pill, 100% |
| `grounded` | Average displayed source score across top-3 ≥ 0.85 | Blue pill, percentage |
| `semantic` | Otherwise | Grey pill, percentage |

### 3. Document Deletion (`DELETE /documents/{filename}`)

`composite.delete_by_file(filename)` does all four sub-stores:
1. SQLite `DELETE FROM chunks WHERE file = ?` and collect deleted chunk_ids.
2. Rebuild text FAISS index keeping only non-deleted vectors (uses `IndexFlatIP.reconstruct()` per kept vector — no re-embedding).
3. Remove postings from the inverted index that reference deleted chunks.
4. If a file_id is no longer present in any chunk, drop its `NumericScaler` and (when `enable_numeric_ann`) its numeric FAISS file.

Then `state.documents`, `state.filename_to_file_id`, and `state.dataframes_by_file_id` get the matching entries cleared.

---

## Chunk Schema

`backend/app/rag/schema.py`:

```python
SCHEMA_VERSION = 2

ChunkType = Literal[
    "text", "table",
    "synopsis", "enumeration", "tabular_summary", "row",
    "entity", "numeric_vector", "time_window",
]

class ChunkMeta(TypedDict):
    # 8 mandatory fields — every chunk has these:
    chunk_id: str            # uuid4 hex
    file: str                # logical filename, sheet-aware
    sheet: str | None
    row_number: int | None
    chunk_type: ChunkType
    canonical_id: str | None # entity key; None for prose / synopsis
    ingestion_time: str      # ISO8601 UTC
    version: int             # = SCHEMA_VERSION
    # NotRequired extras populated by specific chunkers:
    text, page, aliases, column_names, numeric_columns,
    window, stats, source_chunk_ids, file_id
```

When `SCHEMA_VERSION` bumps, the migration script wipes the index dir and re-ingests every file in `data/uploads/`. The mandatory eight survive every chunker; the rest depend on type (e.g. `numeric_vector` chunks have no `text` body — `format_context_v2` synthesizes one for the LLM).

---

## Retrieval Pipeline Walkthrough

`retrieve_v2(query, idx, k, plan)` in `retrieval/orchestrator.py`:

1. **Classify intent.** `classify_intent(query)` produces a `Plan` with `intent`, `raw_entity_phrases`, optional `threshold` (op + value), and optional `window`. Intent is a `Literal` over `point_lookup | list_distinct | numeric_threshold | trend | aggregate | qualitative`.
2. **Resolve canonical_id.** `resolve_canonical_id(plan, idx)` walks the entity phrases longest-first against `aliases.alias_to_canon` and `inverted.lookup_id`, with progressive token trimming.
3. **Exact-match fan-out.** If `canonical_id` is set, look up all chunks under `id:<canon>`. Cell predicates (`X = Y`) hit `cell:<col>:<val>`.
4. **Text ANN over-fetch.** Embed the query, retrieve `k × 4` candidates from FAISS.
5. **Linear rerank.** Score every candidate with `deterministic_score`:
   ```
   score = 1.00 × exact_id
         + 0.40 × exact_cell
         + 0.50 × text_cosine
         + 0.30 × (-min(numeric_dist, 9.9) / 10)
         + 0.20 × type_prior(chunk_type, intent)
   ```
6. **Cross-encoder rerank** (optional, on by default). Splits candidates into exact-id and semantic groups. Exact-id stays in linear-score order. The top `ce_top_n` semantic candidates are scored by `cross-encoder/ms-marco-MiniLM-L-6-v2` and re-sorted. Final order: exact-id first, then CE-ranked semantic.
7. **Return top-k.** Each result carries `_score` (linear) and `_score_breakdown` (`exact_id`, `exact_cell`, `text_sim`, `numeric_dist`, `ce_score`, `intent`, `chunk_type`).

`format_context_v2` then renders `[1] file p.N\nbody` blocks, synthesizing prose for `numeric_vector` and `time_window` chunks that have no text body.

### Source-card confidence (UI mapping)

`_source_card` in `main.py` translates `_score_breakdown` into a 0–1 confidence:

| Signal | Displayed % | Badge |
|---|---|---|
| `exact_id == True` | 100% | Green (exact) |
| `exact_cell == True` | 95% | Blue (field) |
| else | `text_sim × 100` | Grey (semantic) |

`_score` (the linear ranking score) is preserved on the card as `rank_score` for debugging — it can exceed 1.0.

### Numeric-threshold deterministic bypass

`main.py::_try_numeric_threshold_bypass` runs *before* the LLM call when the intent classifier returns `numeric_threshold` and a column resolves in any loaded tabular file. It calls `analysis/query.py::query_table` with the extracted `(op, value)` directly across every matching file, aggregates results, and returns a complete answer with tier `deterministic`. The LLM is bypassed entirely.

This was added because 7B-class models systematically miscompute numeric comparisons — e.g. claim "no funds have NAV < 100" when several do. Pandas filtering with token-overlap column resolution gives an exact, auditable answer in milliseconds.

The bypass is skipped when the user explicitly chose `mode: "chat"`.

---

## Backend — Module Reference

### `app/main.py`

FastAPI app. Routes: `/health`, `/documents`, `/documents/{filename}` (DELETE), `/ingest`, `/chat`. Hosts the threshold-bypass and confidence-tier helpers (`_try_numeric_threshold_bypass`, `_confidence_for_llm_path`, `_source_card`), plus tool gating (`_select_tools`, `_should_enable_tools`).

The `/ingest` endpoint runs PDF post-processing: after the PDF itself is ingested, `extract_pdf_tables_as_csv` writes one CSV per detected table to the upload dir, and each CSV is ingested independently with per-file try/except so one bad table can't kill the rest. Per-CSV results are surfaced in the response as `extracted_tables`.

### `app/state.py`

```python
composite: CompositeIndex      # the four-store orchestrator
dataframes_by_file_id: dict    # {file_id: pd.DataFrame} for tabular files
filename_to_file_id: dict      # {logical_name: file_id} for tool dispatch
documents: dict                # {filename: {file_id, type, chunks}} for /documents
chat_history: dict             # {session_id: deque(maxlen=history_turns*2)}
```

`composite.load_or_init()` is called once at startup in the FastAPI lifespan (in a thread, since `faiss.read_index` and `sqlite3.connect` are blocking).

### `app/text.py`

Universal Unicode + whitespace normalization run by every parser. Smart quotes/em-dashes → ASCII; NBSP/zero-width/BOM stripping; dehyphenation of line-wrapped words (`expense-\nratio` → `expenseratio`); collapse 3+ newlines to a single paragraph break. Identical content with cosmetic differences (smart vs straight quotes) embeds to the same vector after this pass.

### `app/parsers/pdf.py`

Per-page extraction with table-aware processing.

**Table extraction pipeline** (`_extract_validated_tables`):
1. **Two-strategy detection.** First try pdfplumber's `lines` strategy (ruled borders); fall back to `text` strategy (whitespace-aligned columns) when lines finds nothing. Lines wins if both find tables.
2. **Validation** (`_is_valid_table`): rejects <2 rows, <2 cols, or >70% empty cells.
3. **Cleaning** (`_clean_table_rows`): strip whitespace, fold multi-line cells, drop empty rows/cols, pad ragged rows.
4. **Safety wrappers** (`_find_tables_safe`): catches exceptions from `find_tables`/`extract` so exotic PDFs don't kill ingestion.

`parse_pdf` returns `[{page, text, tables_md, tables}]`. `tables_md` becomes a `table` chunk per page; `tables` (raw rows) is consumed by `extract_pdf_tables_as_csv`.

`extract_pdf_tables_as_csv(pdf_path, output_dir)` writes one CSV per validated table. Each table goes through `_normalize_table_for_csv` first: sparse "title / group label" rows on top get stripped (up to 3 iterations, while row 0 is <50% filled and row 1 is denser), then header blanks are conditionally filled with `col_<i>` placeholders only if the resulting row 0 is dense enough to actually be a header. When it isn't, blanks are left so pandas-side `Unnamed: N` detection in `tabular.py` can take over. Returns the list of generated CSV paths.

**Body-text processing** strips boilerplate (`_detect_boilerplate` flags lines on ≥3 distinct pages) and page numbers (`_PAGE_NUMBER_RE`) before applying `text.normalize`.

### `app/parsers/docx.py`

`_iter_body(doc)` walks `doc.element.body.iterchildren()` so paragraphs and tables stay in document order. Heading styles → markdown `#` / `##` / `###`. Tables use the same markdown renderer as PDF tables. `text.normalize` applied last.

### `app/parsers/tabular.py`

The most heavily-engineered parser, because tabular data is where the system both ingests AND analyzes.

1. **Header-row sniffing** — `_has_unnamed_columns` flags `Unnamed: N` columns at row 0; if ≥30% are unnamed, scans rows 1–4 for a clean header.
2. **Multi-sheet** — CSV → 1 record; multi-sheet Excel → 1 record per non-empty sheet, `logical_name = "{filename} :: {sheet}"`.
3. **`_clean`** — strip column names, normalize null sentinels (`""`, `"-"`, `"N/A"`, etc.), drop fully-empty rows/cols, currency/percent → float, date hint → datetime. Coercion is gated by an 80% parse-rate threshold.
4. **`synopsis`** — descriptive one-paragraph blurb, embedded as a high-recall first chunk.
5. **`_summarize`** — head + tail + middle rows (for files >30 rows), `describe(include="number")`, `value_counts().head(5)` for text columns, min/max for datetime columns. Goes into `tabular_summary` chunks.

### `app/chunkers/`

| Module | Role |
|---|---|
| `common.py` | `make_meta()` — fills the 8 mandatory `ChunkMeta` fields. |
| `text_chunker.py` | Sliding-window prose chunks; preserves `doc_header` for context. |
| `tabular_chunker.py` | Emits `synopsis` + `tabular_summary` + `enumeration` + `row` chunks. Picks the entity-name column heuristically. |
| `entity_chunker.py` | One chunk per `canonical_id` aggregating all attributes (latest date if time-series). |
| `numeric_encoder.py` | Builds `numeric_vector` chunks via the per-file `NumericScaler`. No text body. |
| `timewindow_chunker.py` | `detect_timeseries` + `build_time_windows` (30/90/365d mean/std/min/max + CAGR for 365d). |

### `app/id/canonical.py`

`normalize_name()` — idempotent canonicalization. Lowercases, drops `{fund, scheme, plan, the, an, a}`, collapses whitespace. **Keeps `Regular/Direct/Growth/IDCW`** — they distinguish real entities. `AliasMap` — JSON-persisted alias↔canonical map at `data/indexes/aliases.json`, hand-editable.

### `app/index/`

| Module | Role |
|---|---|
| `inverted.py` | Three keyspaces: `id:<canon>`, `cell:<col>:<val>`, `enum:<col>`. JSON persistence. |
| `text_ann.py` | FAISS `IndexFlatIP`; parallel `chunk_ids` array decouples vector position from metadata. `rebuild_keeping()` for delete. |
| `numeric_ann.py` | Per-file `IndexFlatL2`. Gated by `enable_numeric_ann`. |
| `metadata_store.py` | SQLite. Schema: `chunks(chunk_id PK, file, sheet, chunk_type, canonical_id, row_number, ingestion_time, version, file_id, payload)`. Indexes on `(canonical_id, file, chunk_type, file_id)`. |
| `composite.py` | `CompositeIndex` orchestrator + `IngestItem` dataclass. `add_chunks` / `delete_by_file` / `text_search` / `numeric_search` / `stats`. |

### `app/rag/`

| Module | Role |
|---|---|
| `schema.py` | `ChunkMeta` `TypedDict`. `SCHEMA_VERSION = 2`. `ChunkType` `Literal`. |
| `embedder.py` | `multi-qa-MiniLM-L6-cos-v1` — asymmetric (query, passage), 384-dim, L2-normalized output. `_model()` is `@lru_cache(maxsize=1)`. |
| `numeric_scaler.py` | Per-column z-score. Constant cols get std=1; NaN → 0 post-scale. Save/load to/from JSON. |
| `retriever.py` | Thin shim over `orchestrator.retrieve_v2` for back-compat. |

### `app/retrieval/`

| Module | Role |
|---|---|
| `router.py` | Rule-based intent classifier. `Intent` `Literal`, `Plan` dataclass. Captures entity phrases via two regexes (structured "of/for/about" + permissive capitalized phrase). |
| `exact.py` | `resolve_canonical_id` (longest-first + progressive trim) and `extract_cell_predicates`. |
| `semantic.py` | `text_search` wrapper around `composite.text_search`. |
| `rerank.py` | `deterministic_score` — hand-tuned linear formula. `_TYPE_PRIOR` table maps `(intent, chunk_type) → [0,1]`. |
| `cross_encoder.py` | Stage-2 cross-encoder reranker. Lazy-loaded. Skips exact-id chunks. |
| `orchestrator.py` | `retrieve_v2` — runs the full pipeline. `format_context_v2` — renders chunks for the LLM. |

### `app/llm/`

- **`ollama_client.py`** — Ollama async client. `build_messages`, `rewrite_query` (folds chat history into the search query), `stream_chat` (SSE-friendly).
- **`tools.py`** — `compute_metric` and `query_table` tool schemas + `SYSTEM_PROMPT`.

`rewrite_query` runs *before* retrieval; the rewritten query is used only to embed and search. The original message still flows into `build_messages` for generation. Skipped when history is empty; falls back to the original message on any error. Adds ~0.5–1.5s latency per chat turn on Qwen 7B.

`stream_chat(messages, tools=None)` is a single streaming call. When `tools=None`, no tools are sent to Ollama. When `tools=[...]`, the model may emit a tool call in the final chunk; text tokens stream live before that.

**Tool selection is gated server-side** (`main.py::_select_tools`). Local 7B-class models tend to call any registered tool with hallucinated arguments; Qwen 2.5 is markedly better but the gate stays in place. Combines:

1. **User-chosen mode** (`mode: "auto" | "chat" | "aggregate" | "query"`):
   - `chat` → no tools, regardless of message
   - `aggregate` → only `compute_metric` exposed (forces it when a tabular file exists)
   - `query` → only `query_table` exposed
   - `auto` (default) → keyword-gated
2. **Auto-mode keyword gate** — both tools exposed when the message contains a `_QUANT_KEYWORDS` term (`average`, `sum`, `top`, `trend`, `plot`, `highest`, etc.) OR a `_QUERY_KEYWORDS` phrase (`list all`, `show me`, `find all`, `which fund`, `rows with`, etc.).
3. **Tabular precondition** — even forced modes return `None` if no CSV/Excel file has been ingested. The chat endpoint short-circuits with a friendly error before the LLM call.

The frontend exposes mode as four pill buttons above the composer (`Auto` / `Chat` / `Calculate` / `Filter`).

### `app/analysis/`

| Module | Role |
|---|---|
| `metrics.py` | `compute(df, op, column, group_by, n)` — top/bottom/mean/sum/count/median/min/max grouped or flat. |
| `query.py` | `query_table` — filter/sort/select on a DataFrame. Filter ops: `==`, `!=`, `>`, `>=`, `<`, `<=`, `contains`, `in`. String→number coercion built-in. Deterministic. |
| `charts.py` | Plotly figure spec as a plain dict. `trend` → line; `top_n`/grouped → bar; scalar → indicator. |
| `column_match.py` | Token-overlap column resolver. Drives `_try_numeric_threshold_bypass` so a query phrase like "1-year return" maps to the right column even when phrased differently from the header. |

---

## Frontend — Module Reference

### `src/api/client.ts`

All network calls. The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000` (not `localhost` — see Pitfalls below).

- `uploadFile(file)` — multipart POST to `/api/ingest`. Returns `IngestResult { filename, chunks, summary, extracted_tables? }`.
- `getDocuments()` — GET `/api/documents`.
- `deleteDocument(filename)` — DELETE `/api/documents/{encoded filename}`.
- `streamChat(sessionId, message, handlers, signal?, mode?)` — POST `/api/chat`, reads response body as a stream. Parses SSE manually (not `EventSource`) because `EventSource` doesn't support POST.

**SSE parsing details (subtle):**
- Frame separator is `\r?\n\r?\n` (regex). `sse-starlette` emits HTTP-style CRLF (`\r\n\r\n`); a naive split on `\n\n` will silently never match.
- Per the SSE spec, exactly **one** leading space is stripped from `data:` values — not `trimStart()`, which would collapse leading-space token separators (`" The"`, `" provided"`) into a wall of run-on text.

### `src/types.ts`

```typescript
Source       // {filename, page, type, score, match, rank_score, canonical_id?}
ChartPayload // {text, chart_spec, filename, op}
Confidence   // {tier: "deterministic" | "grounded" | "semantic", value, reason}
Message      // {role, content, sources?, chart?, confidence?}
```

### `src/App.tsx`

Root layout. Two-column CSS grid: 220px sidebar + flexible chat area on desktop; sidebar collapses to a slide-in drawer (hamburger toggle, Escape closes) on screens ≤ 768px. `refreshTick` state is incremented on upload and passed to `DocumentList` as `refreshTrigger` to force a refetch.

### `src/components/ChatWindow.tsx`

Manages the message list, mode-pill state, and abort. On send: appends user + empty assistant messages; calls `streamChat(..., mode)` with handlers that mutate the last message in state; mode pills are disabled while streaming. Includes the upload component in the composer; a "Stop" button replaces "Send" while busy.

### `src/components/MessageBubble.tsx`

Renders one message. Shows `"…"` placeholder for empty assistant content. Renders the confidence badge above the answer (deterministic/grounded/semantic with appropriate color), `<PlotlyChart>` when `msg.chart` is set, and a collapsible `<details>` for sources with per-source match badges (exact / field / semantic).

### `src/components/DocumentList.tsx`

Fetches `/api/documents` on mount and whenever `refreshTrigger` changes. Delete button shows a `"…"` spinner during the request and re-fetches on success.

### `src/components/FileUpload.tsx`

File picker + status pill. After upload, the status reads e.g. `Indexed 87 chunks (+2 tables from PDF, 1 failed)` so PDF-table extraction success/failure is visible without opening the chat.

### `src/components/PlotlyChart.tsx`

Renders a `ChartPayload` (Plotly JSON spec) inside the bubble. Auto-resizes on viewport changes.

### `src/styles.css`

CSS variables for the dark palette, grid layout, mobile drawer, table rendering inside markdown bubbles. Tables use `display: block` + `overflow-x: auto` + `white-space: nowrap` so wide tables scroll horizontally instead of squeezing cells into vertical character-by-character text.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Returns chunk counts by type, inverted postings, file list, flags. |
| `GET` | `/documents` | Lists ingested documents with type and chunk count. |
| `DELETE` | `/documents/{filename}` | Removes a doc + all its chunks across the four sub-stores. |
| `POST` | `/ingest` | Multipart file upload → parse → chunk → embed → index. PDFs additionally extract tables as CSV and ingest each. |
| `POST` | `/chat` | SSE stream of `{event: sources \| confidence \| token \| chart \| done}`. Body: `{session_id, message, mode}` where mode ∈ `auto, chat, aggregate, query`. |

---

## Reranking Knobs

### Linear coefficients (`retrieval/rerank.py`)
Hand-tuned, deterministic. To change:
1. Edit `_TYPE_PRIOR` to alter intent ↔ chunk-type preferences.
2. Edit `deterministic_score` to alter signal weights.
3. Re-run `pytest backend/tests/test_retrieval_pipeline.py` — the integration tests pin the determinism contract (exact-id always wins).

### Cross-encoder (`retrieval/cross_encoder.py`)
Default: ON. Set `ENABLE_CROSS_ENCODER=false` in `.env` to disable.

| Knob | Default | Effect |
|---|---|---|
| `enable_cross_encoder` | `True` | Master switch. |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HF model id. Swappable for `BAAI/bge-reranker-base` (slower, more accurate). |
| `ce_top_n` | `20` | Max semantic candidates passed to CE per query. |

CE never re-orders exact-id chunks — that preserves the determinism guarantee. CE only acts on the semantic tier.

First call after server start triggers a one-time model download (~90MB) into `~/.cache/huggingface/`. Subsequent calls hit the cache.

---

## Config Flags Reference

`backend/app/config.py` (overridable via `.env` or environment variables):

| Setting | Default | Purpose |
|---|---|---|
| `ollama_host` | `http://localhost:11434` | Ollama server URL. |
| `ollama_model` | `qwen2.5:7b` | Local LLM. |
| `embed_model` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | Asymmetric (query/passage), 384-dim, max_seq_length=512. |
| `data_dir` | `./data` | Root for `uploads/` and `indexes/`. |
| `top_k` | `10` | Retrieved chunks per query. |
| `chunk_tokens` | `180` | Word-count target per text chunk (~234 tokens; under embedder's 512 cap). |
| `chunk_overlap` | `40` | Overlap between adjacent text chunks. |
| `history_turns` | `4` | Chat-history turns folded into `rewrite_query`. |
| `enable_numeric_ann` | `False` | Build per-file numeric FAISS index at ingest. Off because pandas filtering is faster at our scale. |
| `enable_cross_encoder` | `True` | Run CE second-stage rerank on the semantic tier. |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HF model id for CE. |
| `ce_top_n` | `20` | Max candidates to score with CE. |

---

## Key Design Decisions

**Why a CompositeIndex (multi-store) instead of one FAISS index?**
A single ANN index conflates exact-match identifier lookups, full-text retrieval, and numeric filtering. Splitting them lets each layer use the right primitive: an inverted index for `id:<canonical>`/`cell:<col>:<val>`/`enum:<col>` exact matches (deterministic, sub-millisecond), FAISS for text similarity, and per-file `IndexFlatL2` for numeric vectors when scale demands it.

**Why a numeric-threshold bypass?**
Local 7B-class models systematically miscompute numeric comparisons — they'll claim "no funds have NAV < 100" even when several do. Routing threshold queries to pandas via `query_table` gives an exact, auditable, instant answer. The bypass is gated by intent classification + column resolution, so non-threshold queries fall through to the LLM normally.

**Why extract PDF tables to CSV in addition to embedding them as markdown?**
Markdown table chunks make tables retrievable for prose-style questions ("what columns does the holdings table have?") but they don't support `query_table`/`compute_metric`. Writing each detected PDF table as a standalone CSV and ingesting it through the tabular pipeline lets users filter/aggregate over PDF table data the same way they would over a directly-uploaded Excel file.

**Why store raw vectors alongside the FAISS index?**
FAISS `IndexFlatIP` has no in-place delete. We use `index.reconstruct(i)` to read the kept vectors out and rebuild — no re-embedding required. The metadata SQLite separately maintains the chunk_id ↔ vector-position decoupling so deletes are O(kept) not O(total).

**Why is tool exposure gated by a keyword heuristic?**
Local 7B-class models have shaky tool-call discipline — with any tool registered, they tend to call it on virtually every question (including "summarize the file") and hallucinate the required arguments. Qwen 2.5 7B is much better than Mistral 7B but still not perfect. The tool list is stripped from the Ollama request unless the message looks quantitative *and* a tabular file has been ingested. Forced modes (`aggregate`/`query`) bypass the heuristic when the user knows what they want.

**Why manual SSE parsing instead of `EventSource`?**
`EventSource` only supports GET. The chat endpoint requires POST (session ID + message body). The SSE format is simple enough to parse in ~20 lines.

**Why does the Vite proxy point at `127.0.0.1`, not `localhost`?**
On Windows, `localhost` resolves to both `::1` (IPv6) and `127.0.0.1` (IPv4). uvicorn binds IPv4 only by default, so Node tries IPv6 first, fails noisily (`AggregateError [ECONNREFUSED]`), then retries IPv4. Hard-coding the IPv4 target skips the lookup.

**Why classify intent on the literal message but retrieve on the rewritten one?**
The rewriter folds chat history into the search query so follow-ups like "what is its NAV?" find the entity from earlier turns. But the rewriter can hijack broad-intent queries — "List all benchmarks" → "List benchmarks of HDFC Top 100" — flipping `list_distinct` to `point_lookup`. So intent comes from the literal text; entity phrases from the rewritten query are merged in afterwards.

---

## Pitfalls & Constraints

These are the non-obvious traps that the codebase already accounts for. Don't undo them without knowing why.

**Embedder has a 512-token cap.**
`multi-qa-MiniLM-L6-cos-v1` silently truncates input past `max_seq_length=512`. `chunk_tokens` is 180 words (≈234 tokens), well under the limit. If you swap the embedder, check its `max_seq_length` and update `chunk_tokens`.

**Embedder is asymmetric — re-embedding is mandatory if you swap.**
Vectors from `multi-qa-MiniLM-L6-cos-v1` are not comparable to vectors from a symmetric model like `all-MiniLM-L6-v2` even though both are 384-dim. After changing `embed_model`, run `python -m backend.scripts.migrate_v2` to wipe and re-ingest.

**FAISS dimension is set on the CompositeIndex constructor.**
`backend/app/state.py` passes `text_dim=384`. If you change `embed_model` to one with a different dim, update that argument and re-migrate. The numeric ANN dimension is per-file (set from the scaler's column count).

**SSE format is CRLF-separated.**
`sse-starlette` emits `\r\n\r\n`. Frontend uses `\r?\n\r?\n` regex. Don't switch to `indexOf("\n\n")`.

**Leading-space tokens are significant.**
The SSE parser strips one leading space (per spec) from `data:` values. `trimStart()` would collapse `" The"` + `" provided"` into "Theprovided…".

**Local LLMs over-call tools.**
Don't register a tool and assume the system prompt will prevent it from being called. Gate at the request level: don't send `tools=...` to Ollama unless the message warrants it.

**Re-uploading a file appends duplicate chunks.**
`ingest_file` always assigns a new `file_id`. Use the delete button before re-uploading the same file, or run `migrate_v2` to wipe-and-rebuild from `data/uploads/`. For multi-sheet Excel files, each sheet is a separate sidebar entry — delete each before re-uploading.

**PDF table extraction may produce CSVs that ingest to zero chunks.**
A pdfplumber-detected "table" can be a sparse layout where every cell is a null-token. After `parse_tabular::_clean` strips nulls, the dataframe collapses to empty and `parse_tabular` returns `[]`. The endpoint surfaces this in `extracted_tables` with `chunks: 0` — visible in the upload status pill — but the file isn't queryable. `_normalize_table_for_csv` already filters the worst cases (<2 rows / <2 cols / all empty) before writing, and strips up to 3 sparse top rows (entity-name spans, group labels) so the real header lands at row 0 of the CSV.

**Header-row sniffing in `tabular.py` only kicks in when row 0 looks bad.**
The fallback that scans rows 1–4 only fires when ≥30% of columns came back as `Unnamed: N`. For PDF-extracted CSVs, `_normalize_table_for_csv` works *with* this fallback rather than around it: it fills blanks (suppressing the fallback) only when row 0 is dense enough to be a header, and otherwise leaves blanks (letting the fallback fire and find the real header). For directly-uploaded CSV/Excel files where row 0 is wrong but pandas didn't generate Unnamed columns (e.g. a numeric title row pandas accepts as column names), the fallback won't trigger — manually re-save with a real header row in that case.

**PDF boilerplate detection needs ≥3 pages.**
Single-page or 2-page PDFs short-circuit `_detect_boilerplate` — there's no way to distinguish content from boilerplate without repetition.

**`_DEHYPHEN_RE` over-merges in rare cases.**
Words like `co-\nfounder` become `cofounder` because the regex can't tell whether `-\n` was a hard hyphen or a wrap artifact. For fund factsheets this is almost always a wrap and the merge is correct.

**SQLite metadata store — Windows file locking.**
Stop the uvicorn server before running `migrate_v2`. Otherwise the migration can fail to acquire a write lock on `metadata.sqlite`.

**Snapshot data has no time-window chunks.**
`detect_timeseries` requires multiple distinct dates per `canonical_id`. Trend queries on single-snapshot data return empty time-window results — by design. Don't fabricate trends from snapshots.

**Cross-encoder is English-only.**
Fine for current data; revisit if multilingual fund data lands.

**Canonical-id keeps `Regular/Direct/Growth/IDCW`.**
If users frequently omit those qualifiers, register synonyms in `data/indexes/aliases.json` (the file is hand-editable JSON).

**Numeric ANN is OFF by default.**
At a few-thousand-row scale, pandas filtering via `query_table` is faster and exact. The schema slot and code path exist for when you grow past ~50k rows; flip `enable_numeric_ann=True` and re-migrate.

---

## Adding Features

### Support a new file type

1. Add a parser in `backend/app/parsers/yourformat.py` returning text or a list of records.
2. Add a branch in `backend/app/ingest.py::ingest_file()` for the new extension.
3. Add the extension to the `accept` attribute in `frontend/src/components/FileUpload.tsx`.

### Add a new LLM tool

1. Add the tool schema to `TOOLS` in `backend/app/llm/tools.py`.
2. Handle the tool name in `main.py::event_stream()` (the `if tool_call and tool_call["name"] == ...` block).
3. Update `_select_tools` and the system prompt if the tool needs gating.

### Add a new chunk type

1. **Schema.** Add the label to `ChunkType` in `rag/schema.py`.
2. **Chunker.** Create a new file in `app/chunkers/` exporting `build_<type>_chunks(...)` that returns `list[tuple[ChunkMeta, str]]`. Use `make_meta(...)` from `chunkers/common.py`.
3. **Ingest dispatch.** Wire it into `ingest.py` (`_ingest_tabular_sheet` for tabular files, the top-level `ingest_file` body for prose).
4. **Inverted keys.** If the chunk should be findable by id/cell/enum, add tuples to `IngestItem.inverted_keys`.
5. **Type prior.** Add an entry for the new chunk type to `_TYPE_PRIOR` in `rerank.py` for each intent that should surface it.
6. **Format hook.** If the chunk has no text body, extend `format_context_v2` to synthesize one for the LLM.
7. **Test.** Add a unit test under `tests/test_chunkers_<type>.py` and an integration assertion in `test_retrieval_pipeline.py`.
8. **Migrate.** Bump `SCHEMA_VERSION` if existing chunks need re-ingestion. Run `python -m backend.scripts.migrate_v2`.

### Swap the embedding model

Change `embed_model` in `.env`. Update `text_dim` in `backend/app/state.py` to match the new model's output dim. Run `python -m backend.scripts.migrate_v2` to wipe and re-ingest.

### Swap the LLM

Change `OLLAMA_MODEL` in `.env` to any model in `ollama list` (pull it first with `ollama pull <model>`). Models differ in tool-calling quality — Qwen 2.5 7B is materially better than Mistral 7B at emitting structured tool calls; Llama 3.x also works.

### Persist chat history across restarts

`state.chat_history` is a `defaultdict` in memory — resets on restart. To persist, serialize it to JSON in the lifespan shutdown handler and reload in `load_or_init`.

---

## Migration

When `SCHEMA_VERSION` bumps, the embedder dimension changes, or the on-disk index gets corrupted:

```powershell
# 1. Stop uvicorn
# 2. Run the migrator
python -m backend.scripts.migrate_v2
# 3. Restart uvicorn
```

The script:
1. Wipes `data/indexes/*`.
2. Initializes a fresh `CompositeIndex`.
3. Re-ingests every file in `data/uploads/`.
4. Reports per-file chunk counts grouped by type.

No dual-write, no shadow read. Idempotent.

---

## Testing

The pytest suite covers all pure-logic units (parsing, cleaning, chunking, indexing, retrieval, rerank, metrics, charts, tool gating, threshold bypass, confidence tiers).

### Run

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/ -v
```

(Or just `pytest tests/` after activating the venv.) Cold runs take ~90s — importing `app.main` indirectly pulls in faiss + sentence-transformers + the cross-encoder stub.

### Layout

| Test file | What it locks down |
|---|---|
| `test_text_normalize.py` | Smart-quote/em-dash → ASCII, NBSP/zero-width/BOM stripping, dehyphenation, whitespace collapse. |
| `test_chunking.py` | `chunk_text` empty/single/multi-chunk + step-clamp guard. |
| `test_tabular_cleaning.py` | `_clean` (column strip, nulls, drop-empty, currency/percent coercion, 80% threshold, date hint). `parse_tabular` (CSV roundtrip, single + multi-sheet Excel, header sniffing). `synopsis`. |
| `test_pdf_helpers.py` | `_table_to_markdown`, `_clean_table_rows`, `_is_valid_table`, `_detect_boilerplate`, `_strip_boilerplate`. |
| `test_docx_parser.py` | Heading rendering, table rendering, document order preservation, Unicode normalization. |
| `test_canonical.py` | `normalize_name` idempotency + punctuation invariance. `AliasMap` roundtrip. |
| `test_chunkers_tabular.py` | Row chunks carry `canonical_id`; entity chunks list aliases; enumeration completeness. |
| `test_inverted_index.py` | id/cell/enum lookup; case-insensitive cell match; JSON roundtrip. |
| `test_metadata_store.py` | Upsert + get_many; delete_by_file; by_canonical ordering. |
| `test_numeric_scaler.py` | Constant-column safety, NaN handling, save/load equality. |
| `test_text_ann.py` | Empty search, add+search, save/reload roundtrip. |
| `test_timewindow.py` | Single-date returns None; multi-date builds 30/90/365; CAGR matches manual calc within 1e-6. |
| `test_retrieval_pipeline.py` | Integration: point lookup top-1 = right canonical_id; list-distinct top-1 = enumeration chunk; numeric threshold returns rows satisfying predicate; trend returns time_window chunks. |
| `test_retriever.py` | `format_context_v2` formatting + edge cases. |
| `test_cross_encoder.py` | CE rerank with stub model (token-overlap heuristic — no network required). |
| `test_metrics.py` | All ops (mean/sum/top_n/trend), group-by paths, error paths. |
| `test_charts.py` | All chart types, top_n label-column auto-pick. |
| `test_query.py` | All filter ops, string→number coercion, sort ordering, multi-filter chaining. |
| `test_column_match.py` | `resolve_column` token-overlap matching. |
| `test_tool_gating.py` | `_should_enable_tools` + `_select_tools` mode-override logic. |
| `test_threshold_bypass.py` | `_try_numeric_threshold_bypass` deterministic path: AUM/NAV thresholds, multi-file aggregation, source-file column on multi-file results, zero-match messaging, fallthrough conditions. `_confidence_for_llm_path` tier assignment. |
| `test_source_card.py` | `_source_card` mapping from `_score_breakdown` to displayed match badge. |

The cross-encoder tests use a stub model so they run without network. The real CE is exercised by manual end-to-end testing.

### Known benign warnings

Three `DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute` messages come from FAISS's SWIG bindings on import — out of our control. The dateutil "Could not infer format" warning in tabular tests comes from `pd.to_datetime(errors="coerce")` doing best-effort parsing, which is exactly what we want.

### Demo notebook

`backend/notebooks/demo_v2.ipynb` walks each retrieval intent (point_lookup, list_distinct, aggregate, threshold, trend) showing the router output, candidate set, and score breakdown.
