# Fundlenz — Developer Guide

A local RAG chatbot that ingests fund documents (PDFs, Word, CSV, Excel) and answers questions using semantic search, with optional pandas-driven quantitative analysis and Plotly charts. Everything runs locally — no cloud services required.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Project Layout](#project-layout)
4. [Data Flow](#data-flow)
5. [Backend — Module Reference](#backend--module-reference)
6. [Frontend — Module Reference](#frontend--module-reference)
7. [Key Design Decisions](#key-design-decisions)
8. [Pitfalls & Constraints](#pitfalls--constraints)
9. [Adding Features](#adding-features)
10. [Testing](#testing)

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

Opens two cmd windows: backend on port 8000, frontend on port 5173.

Open `http://localhost:5173` in a browser.

---

## Architecture Overview

```
Browser (React)
    │
    ├── POST /api/ingest   ──► Ingestion Pipeline ──► FAISS index (disk)
    │                                               └─► In-memory DataFrame store
    │
    ├── GET  /api/documents ─► FAISS metadata list
    ├── DELETE /api/documents/{fn} ─► FAISS rebuild-on-delete
    │
    └── POST /api/chat      ──► Retriever (FAISS top-5)
            │                       │
            │                  SSE stream back to browser
            │
            └── Ollama (Qwen 2.5 7B by default)
                    │
                    ├── text answer ─────────────────► token events (streamed)
                    ├── compute_metric tool call ────► pandas aggregation ──► Plotly JSON
                    └── query_table tool call ───────► pandas filter/sort ──► markdown table
```

**Stack**

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript + Vite |
| Backend | FastAPI + uvicorn |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` (local, 384-dim) |
| Vector DB | FAISS `IndexFlatIP` (cosine similarity via inner product on L2-normalized vectors) |
| LLM | Qwen 2.5 7B via Ollama (configurable via `OLLAMA_MODEL` env var) |
| PDF parsing | pdfplumber (preserves tables) |
| Tabular | pandas |
| Streaming | Server-Sent Events (SSE) |

---

## Project Layout

```
Fundlenz/
├── start.bat                   # Launch both servers
├── backend/
│   ├── .env                    # Runtime config (gitignored)
│   ├── .env.example            # Template — copy to .env
│   ├── requirements.txt
│   ├── data/
│   │   ├── indexes/            # index.faiss + metadata.pkl (persisted)
│   │   └── uploads/            # uploaded files (gitignored)
│   ├── conftest.py            # Adds backend/ to sys.path for tests
│   ├── tests/                 # pytest suite (see Testing section)
│   └── app/
│       ├── main.py             # FastAPI app, routes
│       ├── config.py           # Settings (pydantic-settings, reads .env)
│       ├── state.py            # In-memory singletons
│       ├── ingest.py           # Ingestion orchestrator (handles multi-sheet)
│       ├── text.py             # Universal Unicode + whitespace normalization
│       ├── parsers/
│       │   ├── pdf.py          # pdfplumber + boilerplate stripping + normalization
│       │   ├── docx.py         # python-docx: heading hierarchy + tables → markdown
│       │   └── tabular.py      # pandas: header sniff + multi-sheet + cleaning + synopsis
│       ├── rag/
│       │   ├── embedder.py     # SentenceTransformer + chunk_text()
│       │   ├── vector_store.py # FAISS wrapper with save/load/delete
│       │   └── retriever.py    # Query embedding + top-k search
│       ├── llm/
│       │   ├── ollama_client.py  # Streaming chat with tool support
│       │   └── tools.py          # Tool schema + system prompt
│       └── analysis/
│           ├── metrics.py      # pandas aggregations (compute_metric tool)
│           ├── charts.py       # Plotly JSON specs
│           └── query.py        # filter/sort/select/limit (query_table tool)
└── frontend/
    ├── vite.config.ts          # Proxies /api/* to 127.0.0.1:8000 (IPv4 only)
    └── src/
        ├── main.tsx            # React entry point
        ├── App.tsx             # Root layout (sidebar + chat)
        ├── styles.css          # Global CSS (CSS variables, grid layout)
        ├── types.ts            # Shared TypeScript types
        ├── api/
        │   └── client.ts       # All fetch/SSE calls to the backend
        └── components/
            ├── ChatWindow.tsx  # Message list + composer
            ├── MessageBubble.tsx # Single message (text + sources + chart)
            ├── DocumentList.tsx  # Sidebar: indexed files + delete
            └── FileUpload.tsx    # Sidebar: file picker + status
```

---

## Data Flow

### 1. Document Ingestion (`POST /ingest`)

```
File upload
    │
    ▼
ingest.py::ingest_file()
    │
    ├─ .pdf  ──► parsers/pdf.py::parse_pdf()
    │                ├─ pdfplumber extracts per-page text + tables
    │                ├─ Repeating boilerplate (headers/footers/disclaimers) detected
    │                │    by counting line frequency across pages — anything appearing
    │                │    on ≥3 pages is stripped before embedding
    │                ├─ Page-number patterns ("Page 3 of 12", "5 / 12") stripped
    │                ├─ text.normalize() applied (Unicode + dehyphenate + whitespace)
    │                └─ Tables → markdown (| col | col |) before chunking
    │
    ├─ .docx ──► parsers/docx.py::parse_docx()
    │                ├─ Walks the body XML in document order so paragraphs and
    │                │    tables interleave correctly
    │                ├─ Heading 1..9 styles → markdown # / ## / ### prefixes
    │                ├─ Tables → markdown pipe-tables (same renderer as PDF)
    │                └─ text.normalize() applied at the end
    │
    └─ .csv/.xlsx ──► parsers/tabular.py::parse_tabular()
                         ├─ Returns a list of sheet records (CSV → 1 record;
                         │    multi-sheet Excel → 1 record per non-empty sheet)
                         ├─ Header-row sniffing: if pandas creates `Unnamed: N`
                         │    columns at row 0, scans rows 1..4 for the real header
                         ├─ _clean(): strip column names, normalize null sentinels,
                         │    drop empty rows/cols, coerce numeric strings (currency,
                         │    percent), coerce date-named columns. Coercion only
                         │    commits if ≥80% of non-null values parse.
                         ├─ synopsis(): one-paragraph descriptive blurb (rows × cols,
                         │    column types, date ranges, top categorical values) —
                         │    embedded as the FIRST chunk for high-recall lookup.
                         └─ _summarize(): rich stats dump (head + tail + middle rows
                              for big files; describe(); value_counts; date ranges).
                              Cleaned DataFrame is kept in state.dataframes_by_file_id.

    │
    ▼
rag/embedder.py::chunk_text()   — 180-word chunks, 40-word overlap
                                  (kept under embedder's 256-token cap;
                                   see Pitfalls & Constraints below)
    │
    ▼
rag/embedder.py::embed()        — SentenceTransformer encodes all chunks
    │                              (lazy-loaded on first call, cached)
    ▼
rag/vector_store.py::add()      — adds vectors to FAISS IndexFlatIP
    │                              stores raw vectors in self._vectors
    │                              stores metadata list in self.metadata
    ▼
rag/vector_store.py::save()     — writes index.faiss + metadata.pkl to disk
    │
    ▼
state.documents[filename] = {file_id, type, chunks}
```

Each chunk's metadata record stored in FAISS:

```python
{
  "file_id": "abc123",          # hex UUID slice
  "filename": "fund.pdf",       # Compound for multi-sheet Excel: "fund.xlsx :: Sheet1"
  "page": 3,                    # None for docx/tabular
  "type": "text"|"table"|"tabular_summary"|"synopsis",
  "text": "...chunk content..."
}
```

**Multi-sheet Excel files** become multiple logical files. A single upload of
`fund.xlsx` with sheets `Holdings` and `Performance` produces:
- Two file_ids
- Two entries in `state.dataframes_by_file_id`
- Two entries in `state.documents` (and the sidebar list) keyed by
  `"fund.xlsx :: Holdings"` and `"fund.xlsx :: Performance"`
- compute_metric calls use the compound filename as its `filename` argument

Single-sheet Excel and CSV files use the bare filename, unchanged.

### 2. Chat Request (`POST /chat`)

```
{session_id, message}
    │
    ▼
rag/retriever.py::retrieve()
    └─ embed query → FAISS search top-5 → list of chunk dicts with scores

    │
    ▼
llm/ollama_client.py::build_messages()
    └─ [system prompt] + last 4 turns history + user message with injected context

    │
    ▼
main.py::_should_enable_tools(message)        — heuristic gate (see below)
    │
    ▼
llm/ollama_client.py::stream_chat(messages, tools=…)  — streaming Ollama call
    │                                                   tools omitted unless gated in
    │
    ├─ Model answers in text  ──► yields {"type":"token", "content":"..."} per chunk
    │
    └─ Model calls compute_metric ──► yields {"type":"tool_call", "name":..., "arguments":...}

    │
    ▼  (back in main.py::event_stream())
    │
    ├─ token events  ──► "event: token\ndata: <text>\n\n"  (streamed live)
    │
    └─ tool_call ──► _run_compute_metric()
                         └─ metrics.py::compute()  — pandas aggregation
                         └─ charts.py::chart_spec() — Plotly JSON
                         └─ "event: chart\ndata: <json>\n\n"
```

SSE event sequence the browser receives:

```
event: sources
data: [{"filename":"...", "page":2, "score":0.87}, ...]

event: token
data: The fund's expense ratio is

event: token
data: 0.75% annually...

event: done
data:
```

For a chart query, a `chart` event replaces or follows the text events.

### 3. Document Deletion (`DELETE /documents/{filename}`)

FAISS does not support in-place deletion. The approach:

1. Filter `self.metadata` — keep entries where `filename != target`.
2. Get the index positions of kept entries.
3. Slice `self._vectors` (raw numpy array stored alongside the FAISS index) to get only kept vectors.
4. Create a new empty `IndexFlatIP` and `add()` the kept vectors into it.
5. Save the rebuilt index to disk.

This avoids re-embedding any documents — the raw vectors are already stored.

---

## Backend — Module Reference

### `app/text.py`

Universal text normalization run by every parser before chunking. Functions:

| Function | Purpose |
|---|---|
| `normalize(text)` | NFKC normalize → replace Unicode look-alikes (smart quotes, en/em dashes, NBSPs, BOMs) → dehyphenate line-wrapped words (`expense-\nratio` → `expenseratio`) → strip trailing whitespace before newlines → collapse 3+ newlines to a single paragraph break |

Why it matters: identical content in two different documents (one with smart
quotes, one with straight quotes) used to embed differently and retrieve
separately. After normalization they collide on the same vector — better recall
and smaller index.

### `app/config.py`

Reads `.env` via pydantic-settings. All configuration lives here — never hardcoded elsewhere.

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model name as seen in `ollama list`. Qwen 2.5 is the recommended default for tool-call discipline; Mistral 7B and Llama 3.x also work. |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (256-token max — see Pitfalls) |
| `TOP_K` | `5` | FAISS results per query |
| `CHUNK_TOKENS` | `180` | Words per chunk — kept under the embedder's 256-token cap |
| `CHUNK_OVERLAP` | `40` | Overlap words between chunks |
| `HISTORY_TURNS` | `4` | Number of conversation turns sent to LLM |

### `app/state.py`

Module-level singletons — effectively the application's in-memory database.

```python
vector_store          # VectorStore singleton (FAISS + metadata)
dataframes_by_file_id # {file_id: pd.DataFrame} for tabular files
filename_to_file_id   # {filename: file_id} for compute_metric lookup
documents             # {filename: {file_id, type, chunks}} for /documents endpoint
chat_history          # {session_id: deque(maxlen=history_turns*2)}
```

`vector_store.load_or_init()` is called once at startup in the FastAPI lifespan (in a thread, as it's blocking file I/O).

### `app/rag/vector_store.py`

Wraps FAISS. Key points:

- Uses `IndexFlatIP` (inner product). Vectors are L2-normalized by the embedder, so inner product equals cosine similarity.
- Embedding dimension hardcoded to `_EMBED_DIM = 384` so FAISS can initialize at startup without loading the embedding model.
- `self._vectors` stores raw numpy vectors alongside the FAISS index, enabling rebuild-on-delete without re-embedding.
- `save()` / `load_or_init()` use `faiss.write_index` + pickle. The pickle stores a `dict` with both `metadata` and `vectors` keys (old format was a plain list — backwards-compat handled in `load_or_init`).

### `app/rag/embedder.py`

- `_model()` is `@lru_cache(maxsize=1)` — SentenceTransformer loads once on first call, cached forever.
- `embed()` normalizes vectors (`normalize_embeddings=True`) for cosine similarity.
- `chunk_text()` splits on whitespace and uses word count as a token approximation. Sufficient for this scale.

### `app/parsers/tabular.py`

The most heavily-engineered parser, because tabular data is where the system
both ingests AND analyzes (via `compute_metric`). Pipeline:

1. **Header-row sniffing** — call `_has_unnamed_columns()` on the row-0 read.
   If pandas auto-named ≥30% of columns `Unnamed: N`, the real header is below
   a title/branding row. Scan rows 1–4 for a clean header.
2. **Multi-sheet handling** — `parse_tabular()` returns a list of dicts. CSV →
   single entry. Multi-sheet Excel → one entry per non-empty sheet, with
   `logical_name = "{filename} :: {sheet}"`.
3. **`_clean()`** — strip column names, normalize null sentinels (`""`, `"-"`,
   `"N/A"`, etc.), drop fully-empty rows/columns, coerce currency/percent
   strings to floats, coerce date-named columns to datetime. Coercion is
   gated by an 80% parse-rate threshold to avoid destroying mostly-text
   columns.
4. **`synopsis()`** — descriptive one-paragraph blurb. Goes in as the first
   chunk per sheet so questions like "what's in fund.csv?" hit a dense,
   high-information embedding. Distinct from `_summarize()` (stats dump).
5. **`_summarize()`** — head + tail + middle rows (for files >30 rows),
   `describe(include="number")`, `value_counts().head(5)` for text columns,
   min/max for datetime columns. This is what gets embedded as
   `tabular_summary` chunks.

### `app/parsers/pdf.py`

Per-page extraction with table-aware processing:

**Table extraction pipeline** (the `_extract_validated_tables()` function):
1. **Two-strategy detection.** First try pdfplumber's `lines` strategy
   (`vertical_strategy="lines"`, `horizontal_strategy="lines"`) — this
   reliably catches tables with visible ruled borders. If it returns nothing,
   fall back to the `text` strategy with `min_words_vertical=3` — catches
   borderless tables aligned by whitespace alone, common in fund factsheets.
   Only the more reliable result is used (lines wins if both find tables).
2. **Validation** (`_is_valid_table`) rejects obvious false positives:
   <2 rows, <2 columns, or >70% empty cells. Stops multi-column page
   bodies and stray figures from being promoted to "tables".
3. **Cleaning** (`_clean_table_rows`):
   - `_clean_cell` strips whitespace, replaces `\r`/`\n` with spaces (folds
     multi-line cells), and collapses repeated whitespace.
   - Empty rows AND empty columns are dropped.
   - Ragged rows are padded to the table's max width with `""`.
4. **Safety wrappers** — `_find_tables_safe` catches exceptions from
   `find_tables`/`extract` (exotic PDFs occasionally raise) and silently skips
   the offending table rather than failing the whole ingestion.

Tables that survive all of the above go through `_table_to_markdown` to become
standalone chunks. They are NOT merged with surrounding text — table chunks
have different retrieval semantics than prose.

**Body text processing**:
- `_detect_boilerplate(pages)` — counts each line's appearance across pages
  (de-duplicating per-page so a footer appearing twice on the same page
  counts once). Lines appearing ≥3 distinct pages are header/footer/disclaimer
  noise; they get stripped before embedding.
- `_PAGE_NUMBER_RE` — line-level regex catching `Page N`, `Page N of M`,
  `N / M`, bare numerics. Stripped per line.
- `text.normalize()` applied last so the de-hyphenation regex sees real
  newlines.

**Known limitations** (won't be fixed unless they bite in practice):
- A table may end up extracted both as a structured chunk AND as flowing
  text in the body (because pdfplumber's `extract_text()` includes table
  cells as text). Acceptable redundancy — the table chunk has structure,
  the text chunk is a fallback.
- Page-spanning tables get split into two chunks with no link.
- Multi-column page layouts may still confuse the text strategy on rare
  pages without ruled lines.

### `app/parsers/docx.py`

Preserves structure that the old "flatten to plain text" parser threw away:

- `_iter_body(doc)` walks `doc.element.body.iterchildren()` (raw XML) so
  paragraphs and tables stay in document order. Looks up the existing wrapped
  Paragraph/Table objects from `doc.paragraphs` / `doc.tables` by element
  identity — constructing them ad hoc fails because the wrapper needs a
  parent with a `.part` attribute.
- `_heading_level()` reads `paragraph.style.name` and returns `1`–`9` for
  `Heading 1`–`Heading 9` styles. Rendered as markdown `#` / `##` / `###`
  prefixes.
- Tables use the same markdown renderer as PDF tables.
- `text.normalize()` applied to the full assembled string.

### `app/llm/ollama_client.py` + `app/llm/tools.py`

- Two tools are registered: `compute_metric` (aggregations → number / chart) and `query_table` (filter/sort/slice → markdown table). Both target tabular files only.
- `stream_chat(messages, tools=None)` is a single streaming call. When `tools=None`, no tools are sent to Ollama at all. When `tools=[...]`, the model may emit a tool call in the final chunk; text tokens stream live before that.
- **Tool selection is gated server-side, not left to the model alone.** Local 7B-class models tend to call any registered tool even for plain-text questions like "summarize the file" (with hallucinated arguments). Qwen 2.5 7B is markedly better at this than Mistral 7B, but the gate stays in place because it costs nothing and protects against regressions when swapping models. The selection logic is `main.py::_select_tools(message, mode)`, which combines:
  1. **User-chosen mode** (sent in the chat request as `mode: "auto" | "chat" | "aggregate" | "query"`):
     - `chat` → no tools, regardless of message
     - `aggregate` → only `compute_metric` exposed (forces it when a tabular file exists)
     - `query` → only `query_table` exposed
     - `auto` (default) → keyword-gated (see below)
  2. **Auto-mode keyword gate** — both tools exposed when the message contains a `_QUANT_KEYWORDS` term (aggregation: `average`, `sum`, `top`, `trend`, `plot`, `highest`, etc.) OR a `_QUERY_KEYWORDS` phrase (row-level: `list all`, `show me`, `find all`, `which fund`, `rows with`, etc.). For all other messages, tools are stripped and the model just streams text.
  3. **Tabular precondition** — even forced modes return `None` if no CSV/Excel file has been ingested. The chat endpoint additionally short-circuits with a friendly error message in that case, before invoking the LLM.
- The frontend exposes mode as four pill buttons above the composer (`Auto` / `Chat` / `Calculate` / `Filter`). Default is `Auto`; the user toggles when they notice the model picking the wrong tool.
- The system prompt explicitly tells the model: aggregations → `compute_metric`, row subsets → `query_table`. The model still occasionally picks the wrong one in `auto` mode; forced modes (`aggregate` / `query`) collapse the choice to a single tool.

### `app/analysis/metrics.py`

Pure pandas. Supported operations:

| Op | Behavior |
|---|---|
| `mean`, `sum`, `count`, `min`, `max` | Scalar or grouped aggregation |
| `top_n` | `df.nlargest(n, column)` |
| `trend` | Group by a date/period column, compute mean, sort |

### `app/analysis/charts.py`

Returns a Plotly figure spec as a plain Python dict (JSON-serializable). Chart type selection:

- `trend` → line chart (scatter with lines+markers)
- `top_n` or grouped ops → bar chart
- Scalar (no group_by) → indicator (big number)

### `app/analysis/query.py`

Pure DataFrame query: filter → sort → select_columns → limit. Backs the
`query_table` LLM tool.

- **Filter ops**: `==`, `!=`, `>`, `>=`, `<`, `<=`, `contains` (case-insensitive
  substring), `in` (membership). Numeric comparison values are coerced from
  strings since LLMs frequently emit `"100"` instead of `100`.
- **Sort**: takes a column name; `sort_desc=True` by default (most users want
  "top X" semantics).
- **Select columns**: any unknown columns are silently dropped from the list;
  empty list falls back to all columns.
- **Limit**: defaults to 50; negative values fall back to default.

Errors (unknown column, unknown op) are raised as `ValueError` and surfaced to
the user as the assistant message.

---

## Frontend — Module Reference

### `src/api/client.ts`

All network calls. The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000` (not `localhost` — see Pitfalls below) so there are no CORS issues in development.

- `uploadFile(file)` — multipart POST to `/api/ingest`
- `getDocuments()` — GET `/api/documents`
- `deleteDocument(filename)` — DELETE `/api/documents/{encoded filename}`
- `streamChat(sessionId, message, handlers)` — POST `/api/chat`, then reads the response body as a stream. Parses SSE manually (not `EventSource`) because `EventSource` does not support POST requests.

**SSE parsing details (subtle):**
- Frame separator is `\r?\n\r?\n` (regex). `sse-starlette` emits HTTP-style CRLF (`\r\n\r\n`); a naive split on `\n\n` will silently never match and no events will fire.
- Per the SSE spec, exactly **one** leading space is stripped from `data:` values — not `trimStart()`, which would collapse leading-space token separators that some local LLMs emit (`" The"`, `" provided"`) into a wall of run-on text.

### `src/types.ts`

```typescript
Source       // {filename, page, score}
ChartPayload // {text, chart_spec, filename, op}
Message      // {role, content, sources?, chart?}
```

### `src/App.tsx`

Root layout. Two-column CSS grid: 220px sidebar + flexible chat area. `refreshTick` state is incremented on upload and passed to `DocumentList` as `refreshTrigger` to force a re-fetch.

### `src/components/ChatWindow.tsx`

Manages the message list and tool-mode selection. On send:

1. Appends a user message and an empty assistant message to state.
2. Calls `streamChat(..., mode)` with handlers that mutate the last message in state. `mode` is one of `"auto" | "chat" | "aggregate" | "query"` and is sent in the request body to the backend.
3. `onToken` appends to `content`. `onSources` sets `sources`. `onChart` sets `chart`.
4. `onDone` / `onError` set `busy = false`.

The `updateLast` helper always targets the last array element, which is the in-flight assistant message. The mode pills are rendered between the chat scroll area and the composer; they're disabled while the assistant is streaming.

### `src/components/MessageBubble.tsx`

Renders one message. If `msg.content` is empty and role is assistant, shows `"…"` as a loading placeholder. Renders `<PlotlyChart>` if `msg.chart` is set, and a collapsible `<details>` for sources.

### `src/components/DocumentList.tsx`

Fetches `GET /api/documents` on mount and whenever `refreshTrigger` changes. Delete button sets `deleting` state (shows `"…"`) then calls `DELETE /api/documents/{filename}` and re-fetches.

---

## Key Design Decisions

**Why FAISS + pickle instead of a vector database?**
No server process, no Docker, no schema migrations. For a local portfolio demo with O(thousands) of chunks, FAISS with a flat index is instant and correct.

**Why store raw vectors alongside the FAISS index?**
FAISS `IndexFlatIP` has no delete operation. Storing raw numpy vectors in the pickle enables rebuild-on-delete without re-reading files or calling the embedding model again.

**Why is tool exposure gated by a keyword heuristic?**
Local 7B-class models have shaky tool-call discipline. With any tool registered, they tend to call it on virtually every question — including "summarize the file" — and hallucinate the required arguments (e.g. `filename="file.csv"`, `column="sales"`). Qwen 2.5 7B is much better at this than Mistral 7B but still not perfect. The tool list is therefore stripped from the Ollama request unless the message looks quantitative *and* a tabular file has been ingested. See `main.py::_should_enable_tools`.

**Why manual SSE parsing instead of the browser's `EventSource` API?**
`EventSource` only supports GET requests. The chat endpoint requires POST (to send session ID and message body). The SSE format is simple enough to parse manually in ~20 lines.

**Why whitespace-based chunking instead of tiktoken?**
Sufficient for this scale and removes a dependency. The `tiktoken` package is in `requirements.txt` if you want to switch — just replace `chunk_text` in `embedder.py`.

**Why does Vite proxy point at `127.0.0.1`, not `localhost`?**
On Windows, `localhost` resolves to both `::1` (IPv6) and `127.0.0.1` (IPv4). uvicorn binds IPv4 only by default, so Node's `net` module tries IPv6 first, fails noisily (`AggregateError [ECONNREFUSED]` from `internalConnectMultiple`), then retries IPv4 and succeeds. Hard-coding the IPv4 target skips the failed-then-retried lookup entirely.

---

## Pitfalls & Constraints

These are the non-obvious traps that the codebase already accounts for. Don't undo them without knowing why.

**Embedder has a 256-token cap.**
`all-MiniLM-L6-v2` silently truncates input past `max_seq_length=256`. With the default chunking of 800 words (≈1000 tokens), only the first ~190 words of every chunk were being embedded — the rest was invisible to retrieval. `CHUNK_TOKENS` is now 180 words (≈234 tokens), safely under the limit. If you swap the embedder, check its `max_seq_length` and update `CHUNK_TOKENS` accordingly.

**FAISS dimension is hardcoded.**
`vector_store.py` defines `_EMBED_DIM = 384` so the FAISS index can initialize at startup without loading the (slow) embedding model. If you change `EMBED_MODEL`, also change `_EMBED_DIM` and delete `data/indexes/` — embeddings are model-specific.

**SSE format is CRLF-separated.**
`sse-starlette` emits `\r\n\r\n` between frames, not `\n\n`. The frontend parser uses `\r?\n\r?\n` regex. Don't switch to `indexOf("\n\n")` — it will silently match nothing and the chat will appear broken with no console errors.

**Leading-space tokens are significant.**
Some local LLMs emit tokens as `" The"`, `" provided"`, etc. The SSE parser strips one leading space (per spec) from `data:` values. `trimStart()` would collapse those into "Theprovided…".

**Local LLMs over-call tools.**
Don't register a tool and assume the system prompt will keep the model from calling it — it won't, even on Qwen 2.5. Gate at the request level: don't send `tools=...` to Ollama unless you've confirmed the question actually warrants it.

**Re-uploading a file appends duplicate chunks.**
`ingest_file` always assigns a new `file_id` and calls `vector_store.add()`, which appends. Use the delete button in the UI before re-uploading the same file, or you'll get N copies of every chunk. For multi-sheet Excel files, each sheet appears as a separate entry — delete each one before re-uploading.

**Multi-sheet Excel splits one upload into multiple logical files.**
Uploading `fund.xlsx` with three sheets produces three sidebar entries (`fund.xlsx :: Sheet1`, etc.), three FAISS file_ids, and three DataFrames in `state.dataframes_by_file_id`. The `compute_metric` tool needs the full compound filename. The system prompt includes the list of tabular logical names so the model picks correctly.

**Header-row sniffing only kicks in when row 0 looks bad.**
`tabular.py` reads row 0 as the header by default. It only scans rows 1–4 for an alternative if ≥30% of columns came back as `Unnamed: N`. Files where row 0 is wrong but pandas didn't generate Unnamed columns (e.g., row 0 has full numeric values that pandas accepts as column names) won't trigger the fallback. Manually re-save the file with a real header row in that case.

**PDF boilerplate detection needs ≥3 pages.**
Single-page or 2-page PDFs short-circuit `_detect_boilerplate()` — there's no way to distinguish content from boilerplate without repetition. For short docs the boilerplate stays in the index. The page-number regex still applies regardless.

**`_DEHYPHEN_RE` over-merges in rare cases.**
Words like `co-\nfounder` become `cofounder` because the regex can't tell whether `-\n` was a hard hyphen or a line-wrap artifact. For fund factsheets this is almost always a wrap and the merge is correct, but be aware if you swap to a domain with frequent compound hyphenation.

**`state.documents` is in-memory only.**
After a backend restart, the FAISS index reloads from disk but `state.documents` (which holds chunk counts and file types per filename) is empty until you re-ingest. The `/documents` endpoint will still list filenames (those come from FAISS metadata) but `chunks` and `type` will be undefined.

---

## Adding Features

### Support a new file type

1. Add a parser in `backend/app/parsers/yourformat.py` returning text string(s).
2. Add a branch in `backend/app/ingest.py::ingest_file()` for the new extension.
3. Add the extension to the `accept` attribute in `frontend/src/components/FileUpload.tsx`.

### Add a new LLM tool

1. Add the tool schema to `TOOLS` in `backend/app/llm/tools.py`.
2. Handle the new tool name in `main.py::event_stream()` (the `if tool_call and tool_call["name"] == ...` block).
3. Update the system prompt if needed so the model knows when to use it.

### Swap the embedding model

Change `EMBED_MODEL` in `.env`. Also update `_EMBED_DIM` in `backend/app/rag/vector_store.py` to match the new model's output dimension. Delete `data/indexes/` and re-ingest all documents (embeddings are model-specific and incompatible across models).

### Swap the LLM

Change `OLLAMA_MODEL` in `.env` to any model available in `ollama list`. Pull it first with `ollama pull <model>`. Models differ in tool-calling quality — Qwen 2.5 7B (recommended default) is materially better than Mistral 7B at emitting structured tool calls instead of describing them in prose; Llama 3.x also works.

### Persist chat history across restarts

Currently `state.chat_history` is a `defaultdict` in memory — it resets on server restart. To persist it, serialize it to a JSON file in the lifespan shutdown handler and reload it in `load_or_init`.

---

## Testing

The pytest suite lives in `backend/tests/` and covers all pure-logic units
(parsing, cleaning, chunking, vector store, metrics, charts, tool gating).
HTTP endpoints and Ollama streaming are intentionally not unit-tested —
the cost/value ratio is poor for an interactive personal-portfolio app.

### Run

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/ -v
```

(Or just `pytest tests/` after activating the venv.) Cold runs take ~15s
because importing `app.main` indirectly pulls in faiss + sentence-transformers.

### Layout

| Test file | What it locks down |
|---|---|
| `test_text_normalize.py` | Smart-quote/em-dash → ASCII, NBSP/zero-width/BOM stripping, dehyphenation, whitespace collapse, paragraph-break preservation |
| `test_chunking.py` | Empty input, single-chunk, multi-chunk overlap correctness, step-clamp guard against infinite loop |
| `test_tabular_cleaning.py` | `_clean` (column strip, nulls, drop-empty, currency/percent coercion, 80% threshold, date hint), `parse_tabular` (CSV roundtrip, single + multi-sheet Excel, header sniffing, summary contents), `synopsis` |
| `test_pdf_helpers.py` | `_table_to_markdown`, `_clean_table_rows` (whitespace, multi-line cells, empty rows/cols, ragged rows, None handling), `_is_valid_table` (size/empty thresholds), `_detect_boilerplate`, `_strip_boilerplate` |
| `test_docx_parser.py` | Heading rendering, table rendering, empty paragraph skipping, document order preservation, Unicode normalization |
| `test_metrics.py` | All ops (mean/sum/top_n/trend), group-by paths, error paths |
| `test_charts.py` | All chart types, top_n label-column auto-pick (regression test for the broken-x-axis fix) |
| `test_query.py` | All filter ops (`==`/`!=`/`>`/`>=`/`<`/`<=`/`contains`/`in`), string-to-number coercion, sort ordering, column selection, multi-filter chaining, error paths |
| `test_tool_gating.py` | `_should_enable_tools` keyword gating, plus `_select_tools` mode-override logic for `auto` / `chat` / `aggregate` / `query`, including the "no tabular files" short-circuit for forced modes |
| `test_retriever.py` | `format_context` formatting + edge cases |
| `test_vector_store.py` | Empty search, add+search, save/reload roundtrip, dedupe filename listing, delete-by-filename |
| `test_tool_gating.py` | `_should_enable_tools` keyword + state-based gating |

### Known benign warnings

Three `DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute`
messages come from FAISS's SWIG bindings on import. They aren't from project
code and are out of our control. The dateutil "Could not infer format" warning
in tabular tests comes from `pd.to_datetime(errors="coerce")` doing best-effort
parsing, which is exactly what we want.
