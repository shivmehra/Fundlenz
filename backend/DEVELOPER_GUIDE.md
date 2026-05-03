# Fundlenz Backend — Developer Guide

> Local RAG chatbot over fund-performance documents. Multi-index retrieval (inverted exact-match + text ANN + numeric range) with deterministic ranking, type-aware chunking, and provenance tracking.

## 1. Architecture at a glance

```
                INGEST                                   RETRIEVAL
+----------------------+   +-------------------+      +-------------------------+
| parsers/             |   | chunkers/         |      | retrieval/router.py     |
|   pdf.py             |   |   text_chunker    |      |   intent classifier     |
|   docx.py            |-->|   tabular_chunker |      +------------+------------+
|   tabular.py         |   |   entity_chunker  |                   |
+----------------------+   |   timewindow      |                   v
                           |   numeric_encoder |      +-------------------------+
                           +---------+---------+      | retrieval/exact.py      |
                                     |                |   inverted lookup       |
                                     v                +------------+------------+
                           +---------+---------+                   |
                           | id/canonical.py   |                   v
                           |   normalize_name  |      +-------------------------+
                           |   AliasMap        |      | retrieval/semantic.py   |
                           +---------+---------+      |   text ANN              |
                                     |                |   numeric ANN (gated)   |
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
                                                      | LLM synthesis           |
                                                      |   provenance + cites    |
                                                      +-------------------------+
```

## 2. Module reference

### `app/`
| Path | Purpose |
|---|---|
| `config.py` | Pydantic `Settings`. All tunables live here (model names, paths, feature flags). |
| `state.py` | Process-level state: the `CompositeIndex` instance, dataframes by file_id, chat history. |
| `main.py` | FastAPI app. `/ingest`, `/chat` (SSE), `/documents`, `/health`. Maps `_score_breakdown` → confidence card. |
| `ingest.py` | Parser → chunker dispatch. Builds `IngestItem` list, embeds text bodies, calls `composite.add_chunks`. |

### `app/parsers/` (unchanged from v1)
| Path | Purpose |
|---|---|
| `pdf.py` | pypdfium2-based; emits `[{page, text, tables_md}]`. |
| `docx.py` | python-docx; emits a single text blob. |
| `tabular.py` | Excel/CSV; header sniffing, type coercion, multi-sheet. Helpers: `_clean`, `_summarize`, `enumerations`, `synopsis`, `row_chunks`. |

### `app/chunkers/`
| Path | Purpose |
|---|---|
| `common.py` | `make_meta()` — fills the 8 mandatory `ChunkMeta` fields. |
| `text_chunker.py` | Sliding-window chunks for prose; preserves `doc_header` for context. |
| `tabular_chunker.py` | Emits `synopsis` + `tabular_summary` + `enumeration` + `row` chunks. Picks the entity-name column heuristically. |
| `entity_chunker.py` | One chunk per `canonical_id` aggregating all attributes (latest date if time-series). |
| `numeric_encoder.py` | Builds `numeric_vector` chunks via the per-file `NumericScaler`. No text body. |
| `timewindow_chunker.py` | `detect_timeseries` + `build_time_windows` (30/90/365d mean/std/min/max + CAGR for 365d). |

### `app/id/`
| Path | Purpose |
|---|---|
| `canonical.py` | `normalize_name()` — idempotent canonicalization. Drops `{fund, scheme, plan, the, an, a}`. Keeps `Regular/Direct/Growth/IDCW`. `AliasMap` — JSON-persisted alias↔canonical map. |

### `app/index/`
| Path | Purpose |
|---|---|
| `inverted.py` | Three keyspaces: `id:` `cell:` `enum:`. JSON persistence. |
| `text_ann.py` | FAISS `IndexFlatIP`, parallel `chunk_ids` array (decouples vector position from metadata). `rebuild_keeping()` for delete. |
| `numeric_ann.py` | Per-file `IndexFlatL2`. Gated by `enable_numeric_ann`. |
| `metadata_store.py` | SQLite-backed. Schema: `chunks(chunk_id PK, file, sheet, chunk_type, canonical_id, row_number, ingestion_time, version, file_id, payload)`. Indexes on `(canonical_id, file, chunk_type, file_id)`. |
| `composite.py` | `CompositeIndex` orchestrator. `IngestItem` dataclass. `add_chunks` / `delete_by_file` / `text_search` / `numeric_search` / `stats`. |

### `app/rag/`
| Path | Purpose |
|---|---|
| `schema.py` | `ChunkMeta` `TypedDict`. `SCHEMA_VERSION = 2`. `ChunkType` `Literal`. |
| `embedder.py` | `multi-qa-MiniLM-L6-cos-v1` — asymmetric (query, passage), 384-dim, L2-normalized output. |
| `numeric_scaler.py` | Per-column z-score. Fit/transform/save/load. Constant cols get std=1; NaN → 0 post-scale. |
| `retriever.py` | Thin shim over `orchestrator.retrieve_v2` for back-compat. |

### `app/retrieval/`
| Path | Purpose |
|---|---|
| `router.py` | Rule-based intent classifier. `Intent` `Literal`, `Plan` dataclass. Captures entity phrases via two regexes (structured "of/for/about" + permissive capitalized phrase). |
| `exact.py` | `resolve_canonical_id` (longest-first + progressive trim) and `extract_cell_predicates`. |
| `semantic.py` | `text_search` wrapper around `composite.text_search`. |
| `rerank.py` | `deterministic_score` — hand-tuned linear formula over signals. |
| `cross_encoder.py` | Stage-2 cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Lazy-loaded. Skips exact-id chunks. |
| `orchestrator.py` | `retrieve_v2` — runs the full pipeline. `format_context_v2` — renders chunks for the LLM. |

### `app/llm/`
| Path | Purpose |
|---|---|
| `ollama_client.py` | Ollama async client. `build_messages`, `rewrite_query` (folds chat history into the search query), `stream_chat` (SSE-friendly). |
| `tools.py` | `compute_metric` and `query_table` tool schemas + `SYSTEM_PROMPT`. |

### `app/analysis/` (unchanged from v1)
| Path | Purpose |
|---|---|
| `metrics.py` | `compute(df, op, column, group_by, n)` — top/bottom/mean/sum/count/median/min/max grouped or flat. |
| `query.py` | `query_table` — filter/sort/select on a DataFrame. Deterministic. |
| `charts.py` | Plotly spec generator. |

## 3. Chunk schema (`rag/schema.py`)

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
    text, page, aliases, column_names, numeric_columns, window, stats, source_chunk_ids, file_id
```

When the version bumps, the server refuses to load a v(N-1) index and prints the migration command.

## 4. Retrieval pipeline walkthrough

`retrieve_v2(query, idx, k)` in `retrieval/orchestrator.py`:

1. **Classify intent.** `classify_intent(query)` produces a `Plan` with `intent`, `raw_entity_phrases`, optional `threshold` and `window`.
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

## 5. Reranking knobs

### Linear coefficients (`retrieval/rerank.py`)
Hand-tuned, deterministic. To change:
1. Edit `_TYPE_PRIOR` to alter intent ↔ chunk-type preferences.
2. Edit `deterministic_score` to alter signal weights.
3. Re-run `pytest backend/tests/test_retrieval_pipeline.py` — the integration tests pin the determinism contract (exact-id always wins).

### Cross-encoder (`retrieval/cross_encoder.py`)
Default: ON. To turn off: set `enable_cross_encoder=False` in `.env`.

| Knob | Default | Effect |
|---|---|---|
| `enable_cross_encoder` | `True` | Master switch. |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HF model id. Swappable for `BAAI/bge-reranker-base` (slower, more accurate). |
| `ce_top_n` | `20` | Max semantic candidates passed to CE per query. |

CE never re-orders exact-id chunks — that preserves the determinism guarantee. CE only acts on the semantic tier (chunks where `exact_id == False`).

First call after server start triggers a one-time model download (~90MB) into `~/.cache/huggingface/`. Subsequent calls hit the cache.

## 6. Config flags reference (`config.py`)

| Setting | Default | Purpose |
|---|---|---|
| `ollama_host` | `http://localhost:11434` | Ollama server URL. |
| `ollama_model` | `qwen2.5:7b` | Local LLM. |
| `embed_model` | `multi-qa-MiniLM-L6-cos-v1` | Sentence-transformer for chunk + query embedding. |
| `data_dir` | `./data` | Root for `uploads/` and `indexes/`. |
| `top_k` | `10` | Retrieved chunks per query. |
| `chunk_tokens` | `180` | Word-count target per text chunk. |
| `chunk_overlap` | `40` | Overlap between adjacent text chunks. |
| `history_turns` | `4` | Chat-history turns folded into rewrite_query. |
| `enable_numeric_ann` | `False` | Build per-file numeric FAISS index at ingest. Off by default — pandas filtering is faster at our scale. |
| `enable_cross_encoder` | `True` | Run CE second-stage rerank on the semantic tier. |
| `cross_encoder_model` | `ms-marco-MiniLM-L-6-v2` | HF model id for CE. |
| `ce_top_n` | `20` | Max candidates to score with CE. |

Override via `.env` file or environment variables (e.g. `ENABLE_CROSS_ENCODER=false`).

## 7. Adding a new chunk type

1. **Schema.** Add the new label to `ChunkType` in `rag/schema.py`.
2. **Chunker.** Create a new file in `app/chunkers/` exporting `build_<type>_chunks(df_or_text, ...)` that returns `list[tuple[ChunkMeta, str]]`. Use `make_meta(...)` from `chunkers/common.py` to fill mandatory fields.
3. **Ingest dispatch.** Wire the chunker into `ingest.py` (`_ingest_tabular_sheet` for tabular files, the top-level `ingest_file` body for prose).
4. **Inverted keys.** If the chunk should be findable by id/cell/enum, add `("id", canon, "")` etc. tuples to its `IngestItem.inverted_keys`.
5. **Type prior.** Add an entry for the new chunk type to `_TYPE_PRIOR` in `rerank.py` for each intent that should surface it.
6. **Format hook.** If the chunk has no text body, extend `format_context_v2` in `orchestrator.py` to synthesize one for LLM context.
7. **Test.** Add a unit test in `tests/test_chunkers_<type>.py` and an integration assertion in `test_retrieval_pipeline.py` that this chunk type can win for a relevant query.
8. **Migrate.** Bump `SCHEMA_VERSION` if existing chunks need re-ingestion. Run `python -m backend.scripts.migrate_v2`.

## 8. Running tests

```powershell
# All tests
.\.venv\Scripts\activate
cd backend
pytest -q

# Just retrieval pipeline
pytest backend/tests/test_retrieval_pipeline.py -q

# Cross-encoder tests only
pytest backend/tests/test_cross_encoder.py -q
```

The cross-encoder tests use a stub model (token-overlap heuristic) so they run without network. The real CE is exercised by manual end-to-end testing.

## 9. Migration

When `SCHEMA_VERSION` bumps, or when the on-disk index gets corrupted:

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

## 10. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Returns chunk counts by type, inverted postings, file list, flags. |
| `GET` | `/documents` | Lists ingested documents with type and chunk count. |
| `DELETE` | `/documents/{filename}` | Removes a doc + all its chunks across the four sub-stores. |
| `POST` | `/ingest` | Multipart file upload → parse → chunk → embed → index. |
| `POST` | `/chat` | SSE stream of `{event: sources \| token \| chart \| done}`. Body: `{session_id, message, mode}` where mode ∈ `auto, chat, aggregate, query`. |

## 11. Known limits / honest tradeoffs

- **Numeric ANN is overkill at our scale.** Default OFF. `query_table` (pandas) is faster and exact under ~50k rows.
- **Snapshot data has no time-window chunks.** Trend queries on snapshot data degrade gracefully — no fabricated trends.
- **Cross-encoder is English-only.** Fine for current data; revisit if multilingual fund data lands.
- **CE adds ~80ms p50.** Disable via `enable_cross_encoder=False` if you need lower latency.
- **Canonical-id keeps `Regular/Direct/Growth/IDCW`.** If users frequently omit those, register synonyms in `data/indexes/aliases.json`.
- **SQLite metadata store** — Windows file locking risk if migration runs while uvicorn is up. Stop the server before migrating.

## 12. Demo notebook

`backend/notebooks/demo_v2.ipynb` walks each retrieval intent (point_lookup, list_distinct, aggregate, threshold, trend) showing the router output, candidate set, and score breakdown.
