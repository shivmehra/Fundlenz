# Fundlenz

A local RAG chatbot for fund document analysis. Upload fund factsheets (PDF), holdings/NAV spreadsheets (CSV/XLSX), or reports (DOCX) and ask questions; the chatbot retrieves relevant chunks from a local FAISS index, streams an answer from a local Qwen 2.5 (or any Ollama-supported model), and can also run pandas-driven aggregations and render Plotly charts.

## Stack

- **Backend:** FastAPI + SSE streaming
- **Vector store:** FAISS (local, persisted to disk)
- **Embeddings:** SentenceTransformers `all-MiniLM-L6-v2`
- **LLM:** Ollama (default model: `qwen2.5:7b`, configurable via env)
- **Parsing:** pdfplumber, python-docx, pandas (+ openpyxl)
- **Analysis routing:** LLM-as-router via Ollama tool calling
- **Frontend:** React + TypeScript + Vite, react-plotly.js for charts

## Prerequisites

- Python 3.10+
- Node 18+
- [Ollama](https://ollama.com/) running locally with at least one model pulled:
  ```bash
  ollama pull qwen2.5:7b
  ```

## Setup

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
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

## Usage

1. Click **Upload** and pick a PDF, CSV, XLSX, or DOCX.
2. Wait for "Indexed N chunks from <filename>" — the FAISS index is persisted to `backend/data/indexes/`.
3. Ask a question:
   - **Qualitative** ("What is the expense ratio?") → RAG path with streamed answer + Sources panel.
   - **Quantitative** ("Plot average monthly NAV by year from holdings.csv") → the model calls `compute_metric`, the backend runs pandas, and a Plotly chart renders inline.

## Layout

```
backend/
  app/
    main.py            FastAPI app, /ingest, /chat (SSE), /health
    ingest.py          parse → chunk → embed → index orchestrator
    config.py          env-driven settings
    state.py           in-memory: vector store, dataframes, chat history
    parsers/           pdf, docx, tabular
    rag/               embedder, vector_store (FAISS save/load), retriever
    llm/               ollama_client, tools (function schemas + system prompt)
    analysis/          metrics (pandas), charts (Plotly spec)
  data/                indexes/, uploads/  (gitignored)
frontend/
  src/
    App.tsx
    components/        ChatWindow, MessageBubble, FileUpload, PlotlyChart
    api/client.ts      fetch + SSE parser
```

## Configuration (`backend/.env`)

| Var | Default | Notes |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `qwen2.5:7b` | recommended for tool-call discipline; swap to `llama3.2:3b` if 7B is too heavy, or `mistral:7b` if you have it cached |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | |
| `TOP_K` | `5` | chunks retrieved per query |
| `CHUNK_TOKENS` | `800` | approx-token chunk size |
| `CHUNK_OVERLAP` | `100` | tokens overlap between chunks |
| `HISTORY_TURNS` | `4` | last N user+assistant turns sent with each query |

## Notes

- Index persistence: restarting the backend reloads `index.faiss` + `metadata.pkl` from `backend/data/indexes/`, so previously ingested files remain queryable. Tabular DataFrames live in memory only — re-upload spreadsheets after a restart if you need analysis on them.
- Tool calling requires an Ollama model that supports it (Qwen 2.5, Mistral, and recent Llama 3.x all do).
- Out of scope for the scaffold: auth, multi-user, cloud deployment, PDF export.
