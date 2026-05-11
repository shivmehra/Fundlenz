# Fundlenz

A local RAG chatbot for fund document analysis. Upload fund factsheets (PDF), holdings/NAV spreadsheets (CSV/XLSX), or reports (DOCX) and ask questions; the chatbot retrieves from a multi-index pipeline (inverted exact-match + text ANN + numeric range), streams an answer from an LLM (local Ollama by default, or Anthropic / OpenAI when you supply an API key), and can run pandas-driven aggregations or filters with Plotly charts inline.

For full architectural details, module-by-module references, pitfalls, migration, and testing, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

## Stack

- **Backend:** FastAPI + SSE streaming
- **Indexing:** `CompositeIndex` — FAISS `IndexFlatIP` for text ANN, JSON inverted index for `id:`/`cell:`/`enum:` exact matches, SQLite for chunk metadata, optional per-file numeric FAISS (gated)
- **Embeddings:** SentenceTransformers `multi-qa-MiniLM-L6-cos-v1` (asymmetric, query/passage trained, 384-dim)
- **Reranker (optional, on by default):** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** Ollama (default `qwen2.5:7b`, configurable via env). Optional cloud routing to Anthropic (`claude-sonnet-4-6`) or OpenAI (`gpt-4o`) — provider + API key are entered in the UI, stored only in browser `localStorage`, and sent per-request. No key → falls back to local Ollama.
- **Parsing:** pdfplumber, python-docx, pandas (+ openpyxl)
- **Frontend:** React + TypeScript + Vite, react-plotly.js for charts

## Prerequisites

- Python 3.11+
- Node 18+
- [Ollama](https://ollama.com/) running locally with a model pulled:
  ```bash
  ollama pull qwen2.5:7b
  ```

## Setup

### Backend

```bash
cd backend
python -m venv ../.venv
# Windows: ..\.venv\Scripts\activate
source ../.venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # adjust if needed
uvicorn app.main:app --reload --port 8000
```

Health check: <http://localhost:8000/health>

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api/*` to the backend on port 8000.

On Windows you can launch both with `start.bat` from the repo root.

## Usage

1. Click **+** in the composer and pick a PDF, CSV, XLSX, or DOCX.
2. Wait for `Indexed N chunks (+M tables from PDF)` in the upload status. PDF tables are auto-extracted to CSVs and ingested through the tabular pipeline so you can query them with `compute_metric`/`query_table`, just like a directly-uploaded spreadsheet.
3. Pick a mode (default `Auto`):
   - **Auto** — keyword-gated tool exposure; falls back to plain text for qualitative questions.
   - **Chat** — never call tools.
   - **Calculate** — force `compute_metric` (averages, sums, top-N, trends → optional chart).
   - **Filter** — force `query_table` (filter / sort / select rows → markdown table).
4. Ask a question. The Sources panel lists the chunks used, with per-source match badges (exact / field / semantic), and a confidence pill above the answer (deterministic / grounded / semantic).

Numeric-threshold queries ("funds with AUM > 5000") bypass the LLM entirely and run a deterministic pandas filter — that's why their confidence reads `deterministic`.

### Choosing the LLM

The badge next to the title in the header always shows the active LLM (e.g. `Local: qwen2.5:7b` or `Anthropic: claude-sonnet-4-6`). To switch:

1. Open the sidebar **LLM provider** section.
2. Pick **Anthropic (Claude)** or **OpenAI (GPT)** from the dropdown.
3. Paste your API key into the password field.
4. Click **Save**. The badge updates and subsequent chats route to the cloud provider.

To go back to local, set the dropdown to **Local (Ollama)** and click Save (the saved key is cleared from `localStorage`).

The API key lives only in your browser's `localStorage` (keys: `fundlenz_llm_provider`, `fundlenz_llm_api_key`). It is sent in the `/chat` request body but is never written to disk by the backend. Cloud failures (bad key, rate limit, network) surface as an error message in the chat — there is no silent fallback to local.

## Layout (high level)

```
backend/
  app/
    main.py            FastAPI app: /ingest, /chat (SSE), /documents, /health, /stats, /settings, /llm/local
    ingest.py          parse → chunk → embed → index orchestrator
    config.py          pydantic-settings (.env-driven)
    state.py           CompositeIndex + dataframes + chat history
    parsers/           pdf, docx, tabular (+ extract_pdf_tables_as_csv)
    chunkers/          text, tabular, entity, numeric, time-window
    id/canonical.py    normalize_name + AliasMap
    index/             text_ann, numeric_ann, inverted, metadata_store, composite
    rag/               schema, embedder, numeric_scaler
    retrieval/         router, exact, semantic, rerank, cross_encoder, orchestrator
    llm/               ollama_client, anthropic_client, openai_client, router, tools
    analysis/          metrics, query, charts, column_match
  data/                indexes/, uploads/  (gitignored)
  scripts/migrate_v2.py
  tests/               208 tests
frontend/
  src/
    App.tsx
    components/        ChatWindow, MessageBubble, FileUpload, DocumentList, StatsCard, PlotlyChart
    api/client.ts      fetch + SSE parser
```

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for the full layout, data flow, and module reference.

## Configuration (`backend/.env`)

| Var | Default | Notes |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Recommended for tool-call discipline. Llama 3.x and Mistral 7B also work. |
| `EMBED_MODEL` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | Asymmetric query/passage embedder, 384-dim, max_seq_length=512. |
| `TOP_K` | `10` | Chunks retrieved per query. |
| `CHUNK_TOKENS` | `180` | Approx-word chunk size (under embedder's 512-token cap). |
| `CHUNK_OVERLAP` | `40` | Words overlap between chunks. |
| `HISTORY_TURNS` | `4` | Last N user+assistant turns folded into `rewrite_query`. |
| `ENABLE_NUMERIC_ANN` | `false` | Build per-file numeric FAISS index. Off because pandas filtering is faster at our scale. |
| `ENABLE_CROSS_ENCODER` | `true` | Run CE second-stage rerank on the semantic tier. |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HF model id. |
| `CE_TOP_N` | `20` | Max candidates passed to CE per query. |

## Notes

- **Persistence.** Restarting the backend reloads the `CompositeIndex` artefacts (`text.faiss`, `inverted.json`, `metadata.sqlite`, `aliases.json`, per-file `scaler_*.json`) from `backend/data/indexes/`, so previously ingested files remain queryable. Tabular DataFrames live in memory only — re-upload spreadsheets after a restart if you need analysis on them.
- **PDF table extraction.** When you upload a PDF, every detected table is also written as a CSV to `backend/data/uploads/{pdf_stem}_page{N}_table{M}.csv` and ingested through the tabular pipeline. The upload status tells you how many landed and how many failed.
- **Re-uploading appends.** `ingest_file` always assigns a new `file_id`. Use the delete button before re-uploading the same file, or run `python -m backend.scripts.migrate_v2` to wipe-and-rebuild from `data/uploads/`.
- **Migration.** When `SCHEMA_VERSION` bumps or the index gets corrupted, stop the backend and run `python -m backend.scripts.migrate_v2`. It wipes `data/indexes/*` and re-ingests everything from `data/uploads/`.
- **Tool calling** requires an Ollama model that supports it. Qwen 2.5, Mistral, and recent Llama 3.x all do. Anthropic and OpenAI both support tool calling out of the box — the same `compute_metric` and `query_table` schemas are converted to each provider's format inside `app/llm/anthropic_client.py` / `app/llm/openai_client.py`.
- **Cloud LLM credentials** never leave your browser → backend request path. The API key is held in `localStorage`, sent in the JSON body of `/chat`, and used for the duration of that request only. Nothing is logged, cached, or written to disk on the backend.
- **Sidebar capacity indicator.** After each upload, the sidebar **Capacity** card refreshes from `GET /stats` and shows total tabular rows (persisted to SQLite, so the count survives a backend restart) plus system RAM available / total / process RSS. The RAM row turns amber at ≥75% system RAM used and red at ≥90% — warn-only, ingestion is never blocked.
- Out of scope for the scaffold: auth, multi-user, cloud deployment.

## Testing

```bash
cd backend
pytest -q
```

208 tests, ~90s cold (faiss + sentence-transformers + cross-encoder stub all import). See the Testing section in [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#testing) for the per-file breakdown.
