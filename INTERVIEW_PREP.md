# Fundlenz — Interview Prep & Demo Script

> One-page reference. Read top-to-bottom for narrative flow. Each section has a **Talking point** (the sound bite to deliver) and **Why** (the rationale to back it up if asked).

---

## 1. Elevator Pitch (30 seconds)

**Talking point:** *"Fundlenz is a RAG system for analyzing mutual fund documents — Excel sheets, PDFs, Word docs. Indexing, retrieval, and embeddings all run locally; generation defaults to a local Ollama LLM but can be routed to Anthropic or OpenAI by entering an API key in the UI. The novelty is the retrieval pipeline: instead of a single FAISS index over text chunks, I built a four-layer composite index that combines deterministic exact-match, text ANN, optional numeric ANN, and a SQLite metadata store, all stitched together by a rule-based intent router. This lets me answer fund-name lookups deterministically while still falling back to semantic search for qualitative questions."*

**Why this design:** A naive single-index RAG kept getting fund NAVs wrong because per-row chunks looked too similar to each other. Adding deterministic identifier resolution and intent-aware chunk-type priors fixed that without giving up qualitative quality.

---

## 2. The Stack (one-liner each)

| Layer | Choice | Why |
|---|---|---|
| **LLM** | `qwen2.5:7b` via Ollama (default); `claude-sonnet-4-6` or `gpt-4o` opt-in via UI | Local default = no API cost, no data egress. Cloud option for quality lift — pluggable through a single `router.py` abstraction |
| **Embedder** | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (384-dim) | Asymmetric — trained on (query, passage) pairs; better for short user questions retrieving longer chunks than symmetric MiniLM |
| **Cross-encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Standard MS-MARCO reranker; ~80ms overhead, big quality lift on qualitative queries |
| **Vector store** | FAISS `IndexFlatIP` over L2-normalized vectors | Brute-force at this scale (<10k chunks) — 100% accuracy, no index-build complexity |
| **Metadata** | SQLite | Single file, transactional, fast filter on `(file)`, `(canonical_id)` |
| **Inverted index** | JSON file | Tiny dataset; sub-millisecond exact-match lookup |
| **Backend** | FastAPI + SSE streaming | Async, native streaming for token-by-token UI |
| **Frontend** | React + Vite + TypeScript | Clean component model; Vite proxy avoids CORS |

---

## 3. The End-to-End Flow

### Ingest flow (`POST /ingest`)
```
file upload
  → SHA-256 hash check (skip if identical to last upload)
  → parse (pdf.py / docx.py / tabular.py)
  → for PDFs: also extract embedded tables as CSVs and re-ingest each
  → chunkers fan out: text / tabular (synopsis, enum, summary, row) / entity / time-window / numeric_vector
  → embed all text chunks (batched)
  → CompositeIndex.add_chunks()
       ├─ MetaStore.upsert (SQLite)
       ├─ TextANN.add (FAISS)
       ├─ InvertedIndex.add_id / add_cell / add_enum
       └─ NumericANN.add (per file_id, gated by enable_numeric_ann)
  → save to disk (text.faiss, inverted.json, metadata.sqlite, scalers)
```

### Query flow (`POST /chat`, SSE stream)
```
user question
  → rewrite_query() folds prior turns into a self-contained search query
  → classify_intent() picks one of 7 intents (rules, no LLM)
  → retrieve_v2():
       ├─ exact identifier resolution (canonical_id via alias map + inverted index)
       ├─ cell-predicate extraction (regex → inverted index)
       ├─ text ANN over-fetch (k=40, FAISS cosine)
       ├─ deterministic linear scoring → top-k
       └─ optional cross-encoder rerank on the semantic tier
  → format_context_v2() renders numbered citations
  → llm/router picks Ollama (no key) OR Anthropic/OpenAI (key in request body)
  → SSE stream:
       1. event:sources    — citation cards
       2. event:confidence — tier + value
       3. event:token      — streamed tokens
       4. event:chart      — if compute_metric ran
       5. event:done
```

**Deterministic fast-path:** Numeric-threshold queries (e.g. "AUM > 5000") bypass the LLM entirely — pandas filter via `query_table`, confidence tier `deterministic`, value `1.0`.

---

## 4. Chunking Strategy — the most differentiated part of the project

### The core idea
Different question shapes need different chunk shapes. A single chunk type — say, "split prose into 500-token windows" — handles qualitative questions well but fails on tabular data, where the right answer is a specific row, a list of distinct categories, or a column's mean. So I emit **nine chunk types** at ingest, and the retrieval router boosts the right type for each intent.

### The nine chunk types

| Chunk type | Produced by | What it contains | Best for |
|---|---|---|---|
| `text` | `text_chunker.py` | 180-word sliding window, 40-word overlap, optional `[Doc] page N:` header prefix | qualitative prose questions |
| `table` | PDF parser | Markdown rendering of an embedded PDF table | qualitative questions about a tabular section |
| `synopsis` | tabular | "X is a tabular dataset with N rows and M columns. Text columns: ..." | "what's in this file?" |
| `tabular_summary` | tabular | head/tail/middle rows + dtypes + describe() + cleaning report | overview / dataset description |
| `enumeration` | tabular | "All distinct values in column X: a, b, c, ..." (cardinality 2–500 only) | "list all categories", "show distinct funds" |
| `row` | tabular | "Row from file: col1=val1; col2=val2; ..." (cap 500/table) | point lookups by entity |
| `entity` | tabular | One per canonical_id with aliases + aggregated attributes | fund-name lookup |
| `numeric_vector` | tabular | Z-scored float32 vector over numeric columns (no text body) | numeric ANN — gated, off by default |
| `time_window` | tabular | "Time window 30d for X: NAV mean=... std=... cagr=..." | trend / time-series queries |

### Sliding window for prose — defaults
- **180 words ≈ 234 tokens** (the embedder's max_seq is 512, so this leaves headroom).
- **40-word overlap** — ~22% — preserves cross-boundary context without doubling index size.
- **Why words, not tokens:** Token-accurate chunking needs the model's tokenizer; word-based is good enough at this size and avoids loading the tokenizer for chunking.

### Tabular fan-out — why FOUR chunks per table
**Talking point:** *"For each tabular file I emit four chunks because each one optimizes a different retrieval shape. The synopsis is high-recall — anyone asking 'what's in this file' hits it. The enumeration chunk is critical for 'list all X' queries because asking ANN to retrieve all distinct values is hopeless — it'll dilute. The summary chunk gives the LLM the dtype and describe() output for free. And the per-row chunks are what point-lookups actually want."*

### Entity chunks — the deduplication of the entity space
**Why:** A fund like "HDFC Top 100 Direct Growth" might appear in multiple files. The entity chunker emits one chunk per `canonical_id` carrying aliases, so a query for "HDFC Top 100" resolves through the alias map even if the user omits "Direct Growth".

### Time-window chunks — only when warranted
- Detected via `detect_timeseries()`: a file is time-series if any entity has multiple distinct dates.
- Emits 30/90/365-day rolling stats (mean, std, min, max, count).
- 365-day window also computes CAGR.
- For snapshot data this is silently skipped — no time chunks. **No fabricated trends from snapshots.**

---

## 5. Embedding Strategy

### Model: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`
- 384 dimensions, max_seq 512 tokens
- L2-normalized at encode time → IndexFlatIP search returns cosine similarity directly
- **Asymmetric**: explicitly trained on (short query, long passage) pairs

### Why asymmetric over symmetric (e.g. `all-MiniLM-L6-v2`)
**Talking point:** *"User questions are short — five to fifteen words. Document chunks are long — 100 to 500 words. Symmetric embedders are trained on similar-length pairs and behave best when query and document look the same. Asymmetric embedders are trained specifically on this mismatch, so they don't penalize the length asymmetry."*

### The honest caveat
**Talking point:** *"On row chunks specifically, the asymmetric advantage degrades because row chunks aren't natural prose — they're `key=value; key=value` strings, which are out-of-distribution for any model trained on web text. That's part of why I added deterministic exact-match — semantic similarity isn't strong enough on row chunks alone."*

### Why not bigger embedders (e.g. bge-large)?
- 1024-dim doubles memory + recall latency
- 4× model size; meaningful CPU latency at ingest
- Quality lift on a small corpus didn't justify the cost
- Drop-in: change `settings.embed_model`, re-ingest

---

## 6. Indexing Strategy — the four-layer CompositeIndex

```
        ┌─────────────────────────────────────────────┐
        │             CompositeIndex                  │
        ├─────────────────┬───────────────────────────┤
        │   TextANN       │   FAISS IndexFlatIP       │
        │                 │   384-dim, L2-normalized  │
        ├─────────────────┼───────────────────────────┤
        │   InvertedIndex │   JSON file               │
        │                 │   id: / cell: / enum:     │
        ├─────────────────┼───────────────────────────┤
        │   MetaStore     │   SQLite                  │
        │                 │   chunk_id PK, indexes on │
        │                 │   (file), (canonical_id)  │
        ├─────────────────┼───────────────────────────┤
        │   NumericANN    │   FAISS IndexFlatL2       │
        │                 │   per file_id, GATED      │
        └─────────────────┴───────────────────────────┘
```

### Why four stores instead of one
**Talking point:** *"Each store is optimized for a different access pattern. The text ANN is for fuzzy semantic match; the inverted index is for deterministic exact-match in O(1); SQLite is for metadata hydration after retrieval; numeric ANN is for range-style queries when I scale past pandas. Combining them in one index would force compromises — for instance, you can't put exact-match logic inside a FAISS index without bolting on filters."*

### Why FAISS `IndexFlatIP` and not HNSW or IVF
- Corpus is small (under 10k chunks for typical Fundlenz dataset)
- Flat is exact (100% recall); HNSW is approximate
- No index-build phase — just append vectors
- Latency at this scale: <5ms; HNSW would be the same; IVF would be slower (training overhead)
- **Drop-in upgrade path:** if corpus grows past 100k chunks, swap `IndexFlatIP` → `IndexHNSWFlat` — same API

### Why per-file numeric ANN
- One scaler per file (different files have different numeric schemas; can't mix)
- Disabled by default (`enable_numeric_ann: false`)
- Pandas filter via `query_table` is faster and exact at current scale
- Code path exists for when corpus crosses the ~50k row threshold

### Inverted index keyspaces
- `id:<canonical_id>` → list of chunk_ids carrying that entity
- `cell:<col>=<value>` → list of chunk_ids where column equals that value
- `enum:<col>` → list of enumeration chunks for that column

---

## 7. Retrieval Strategy

### Intent routing — 7 intents, rule-based, deterministic
| Intent | Trigger keywords / patterns | Boost target |
|---|---|---|
| `point_lookup` | "what is the NAV of X", "tell me about X" | row=1.0, entity=0.9 |
| `list_distinct` | "list all", "show all", "all distinct" | enumeration=1.0 |
| `aggregate` | "average", "sum", "max", "top N" | tabular_summary=0.6 |
| `filter_rows` | "show me", "find all", "funds with" | row=1.0, entity=0.6 |
| `numeric_threshold` | regex `>`, `<`, `≥`, "greater than" | numeric_vector=0.8, row=0.7 |
| `trend` | "trend", "over time", "by year" | time_window=1.0 |
| `qualitative` | fallback | text=1.0, table=0.6 |

**Why rule-based and not an LLM router:** Cost (an extra ~200ms LLM call), latency, determinism. Rules cover the 80%; if a query slips through to `qualitative`, retrieval still works — it just doesn't get a type-prior boost.

### The orchestrator (`retrieve_v2`)
1. **Intent classification** — rules-based, returns `Plan(intent, canonical_id, threshold, window, ...)`
2. **Exact identifier match** — alias-map resolution → inverted index `id:` lookup → marked `exact_id=True`
3. **Cell predicate extraction** — regex `\b(col)\s*(=|is|:)\s*(val)\b` → inverted index `cell:` → marked `exact_cell=True`
4. **Text ANN over-fetch** — `k = max(top_k * 4, 20)` — gives reranking headroom
5. **Linear deterministic scoring** (see formula below)
6. **Optional cross-encoder rerank** of top 20 semantic-tier candidates
7. **Top-k slice** with `_score_breakdown` injected into each chunk

### Why over-fetch 4× before reranking
**Talking point:** *"FAISS scoring alone won't surface a row chunk that has the right canonical_id but lower text similarity than 10 lookalike rows. By over-fetching 40 candidates and then re-scoring with the type-prior + exact-match boost, the right chunk floats to the top."*

---

## 8. Reranking Strategy — the deterministic linear formula

```
score = 1.00 * exact_id_match
      + 0.40 * exact_cell_match
      + 0.50 * text_cosine_similarity
      + 0.30 * (-min(numeric_l2, 9.9) / 10)     # closer is better
      + 0.20 * type_prior(chunk_type, intent)
```

### Why these specific weights
**Talking point:** *"I designed this so that an exact identifier match — score 1.0 — beats any combination of fuzzy signals. A cell match alone is 0.4, which is the same as a text similarity of 0.8 weighted at 0.5. So a chunk has to be a really good semantic match to outrank a deterministic field hit. The type prior is the smallest weight because intent classification is heuristic — I don't want a wrong intent guess to dominate."*

### Cross-encoder reranking — the second stage
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores `(query, chunk)` jointly — much higher quality than bi-encoder cosine
- Cost: ~80ms per query (loads a second model into RAM)
- **Only re-orders the semantic-tier candidates** — exact-id matches keep their position
- Bounded by `ce_top_n=20` — doesn't try to score the long tail
- **User-toggleable** at runtime via the UI (`enable_cross_encoder` setting)

### Why CE is opt-in (defaults on)
**Talking point:** *"The cross-encoder gives the biggest quality lift on qualitative questions where the bi-encoder is weakest. But it adds 80ms and a second model in RAM. So I made it user-toggleable — defaults on, but if the user has tight memory or wants minimum latency, they can flip it off."*

---

## 9. Confidence Tiers — the trust signal we surface to users

| Tier | Value | When |
|---|---|---|
| `deterministic` | 1.0 | A tool ran (compute_metric/query_table) OR all top sources are exact-id matches |
| `grounded` | 0.75–0.99 | Average source score ≥ 0.85 |
| `semantic` | 0.0–0.85 | Otherwise — pure ANN retrieval, no exact match |

**Why this matters for UX:** The user sees a colored badge in the UI showing whether the answer is backed by a deterministic computation or a soft semantic match. This is the kind of trust UI you almost never see in consumer RAG tools.

---

## 10. Deduplication on Re-Upload (the bug I just fixed in design)

**The problem:** Re-uploading the same file doubles all chunk counts because every ingestion creates fresh UUID `chunk_id`s — SQLite's `INSERT OR REPLACE` never matches.

**The fix:**
- **Layer 1:** SHA-256 of upload bytes; if hash matches stored hash → return cached result, skip ingestion entirely
- **Layer 2:** `delete_by_filename(original_filename)` at top of `ingest_file()` — purges old chunks before adding new ones, also handles multi-sheet Excel via prefix match (`file = ? OR file LIKE ? || ' :: %'`)

**Why two layers:** Layer 1 is the fast path for accidental re-uploads; Layer 2 is the safety net that fires regardless of caller — guarantees at most one version per filename.

---

## 11. Capacity Tracking — RAM headroom + persistent row count

A small but interview-worthy operational polish: after every upload, the sidebar shows total indexed rows and how much RAM is left on the machine.

### Two independent backends, one card

**Row count** lives in SQLite. New `file_stats` table in the existing `metadata.sqlite`: `(file_id PK, filename, row_count, ingestion_time)`. Written at ingest by `ingest.py`; cascade-deleted by `composite.delete_by_file`. Survives a backend restart **even though the DataFrames in `state.dataframes_by_file_id` don't** — so the count stays honest across the very common "restart, don't immediately re-upload" flow.

**RAM** is computed live by `psutil.virtual_memory()` plus `psutil.Process().memory_info().rss`. Cheap (~1ms) — safe to hit on every refresh.

Both are exposed by one endpoint: `GET /stats` returns `{ram: {process_rss_bytes, system_available_bytes, system_total_bytes, system_percent_used}, rows: {total_tabular_rows}}`.

### Why RAM and not disk

**Talking point:** *"The actual ceiling on this app is RAM, not disk. The FAISS index and all the ingested DataFrames live in process memory. When RAM fills up the OS starts swapping and the whole pipeline slows down. Disk is essentially infinite by comparison on a modern laptop. So I show available RAM with a colour signal — amber at 75% used, red at 90% — but I never block ingestion. The user gets a warning, then makes their own call."*

### Why persist in SQLite, not in `state.documents`

**Talking point:** *"The DataFrames don't survive a restart — only the index on disk does. If I computed row count from `state.dataframes_by_file_id` it would read zero every time the backend rebooted, even though the indexes on disk still have those rows. That's a confusing UX. Storing the count in SQLite costs me one tiny extra row per file and gives me a stat that's consistent with what's actually queryable."*

### What's not done (honest)

- Disk-usage tracking — not shown. If a user uploads enough PDFs to fill the data partition, they'll see disk-full errors from the OS, not from the app.
- Per-file RAM accounting — Fundlenz's RSS includes the loaded embedder + cross-encoder weights (~150 MB combined), not just the DataFrames. The number is honest about Fundlenz's footprint but doesn't break out "DataFrames vs models."

---

## 12. LLM Provider Abstraction — pluggable backend

**The shape of the problem:** local LLMs are free and private but plateau around 7B-class quality. Cloud LLMs are more capable but cost money and leak data. A real product needs both — local as the default, cloud as the opt-in.

### The abstraction

A single function signature, three implementations:

```python
# app/llm/{ollama,anthropic,openai}_client.py
async def stream_chat(messages, tools=None, *, api_key=..., model=...) -> AsyncIterator[dict]:
    # yields {"type": "token", "content": str}
    # yields {"type": "tool_call", "name": str, "arguments": dict}
```

`router.py` picks the right client based on whether the request carries a valid `(provider, api_key)` pair. All three yield the same event shape, so `/chat`'s event loop doesn't change.

### What's different between providers (the interesting bits)

| Concern | Ollama | OpenAI | Anthropic |
|---|---|---|---|
| Tool schema | OpenAI-compatible | OpenAI-compatible | Different — `{name, description, input_schema}`. Converted on the fly. |
| System prompt | Role=system message | Role=system message | Separate `system` param, not a message. Split out at conversion. |
| Tool-call streaming | Final chunk | Across multiple deltas, buffered per `tc.index` until `finish_reason=tool_calls` | After all text streamed; read from final assembled message |
| Token streaming | `msg.content` per part | `delta.content` per chunk | `content_block_delta` with `text_delta` |

**Talking point:** *"The tool schema is the messiest part. OpenAI and Ollama both accept the `{type: function, function: {...}}` format, so I keep the schemas in that shape. The Anthropic client has a small `_convert_tools` helper that translates to their flat `{name, description, input_schema}` format. Doing the conversion inside the Anthropic client (instead of forcing all clients to use a neutral internal format) keeps the abstraction thin — three clients, one signature, each owns its own adapter logic."*

### Where the API key lives

- **Frontend:** `localStorage` (keys: `fundlenz_llm_provider`, `fundlenz_llm_api_key`)
- **In transit:** request body to `/chat`, only when the user has opted in
- **Backend:** in process memory for the duration of one `stream_chat` call — never logged, never written to SQLite, never serialized

**Why this design:** the user opts into the cost/privacy trade-off explicitly, and the backend has zero persistent secret state. A new dev cloning the repo can run it without any API keys; the key is per-session, per-browser.

### Failure semantics

No silent fallback on cloud failure. If the user picked Anthropic and the key is invalid, the exception propagates through router → `/chat`'s try/except → SSE token event: `"Model error: ..."`. We do not silently switch to local on failure because that would mask config mistakes and produce confusingly different answers.

**One quiet exception:** `rewrite_query` always uses local Ollama, even when chat generation is routed to cloud. Reason: rewriting runs on every turn and burns tokens on a step the user doesn't see — not worth the marginal quality lift.

### The header badge

Small UX detail with a reason: the header shows `Local: qwen2.5:7b` or `Anthropic: claude-sonnet-4-6` — so the user always knows which LLM produced the answer. The `activeProvider` state is decoupled from the form state and updates only on Save — the badge never claims a provider whose key hasn't been persisted.

---

## 13. The User-Facing Settings Toggles

Three runtime-configurable settings, exposed in the sidebar with click-pinned info tooltips (the Capacity card from §11 lives just above them):

1. **Numeric ANN** (`enable_numeric_ann`, default `false`)
   - Tooltip: "Enable when you have 50,000+ rows..."
   - Builds per-file FAISS L2 indices for numeric columns
   - At small scale, pandas is faster and exact

2. **Cross-encoder** (`enable_cross_encoder`, default `true`)
   - Tooltip: "Second-stage rerank that scores (query, chunk) jointly..."
   - Adds ~80ms; loads a second model

3. **LLM provider** (dropdown: Local / Anthropic / OpenAI)
   - API key field for cloud providers; stored in `localStorage` only
   - Defaults to Local — chatbot works out of the box with no key
   - Per-session decision; doesn't touch backend state

**The first two mutate via PATCH `/settings`** at runtime — no restart needed. **The third lives entirely in the browser** and rides along in each `/chat` request body.

---

## 14. Demo Script — Questions to Ask the Chatbot

Pick 3–4 of these depending on time. Each one demonstrates a different capability of the pipeline.

### Tier A — must-show (covers the design pitch)

| # | Question | Demonstrates |
|---|---|---|
| 1 | **"What is the NAV regular of [actual fund name from your data]?"** | `point_lookup` intent → canonical_id resolution → row chunk surfaces top-1 → confidence tier `deterministic` or `grounded` |
| 2 | **"List all benchmarks"** (or any column with low cardinality in your data) | `list_distinct` intent → enumeration chunk surfaces → complete listing, not a sample |
| 3 | **"Funds with AUM greater than 5000"** | `numeric_threshold` intent → bypasses LLM → pandas filter via `query_table` tool → confidence `deterministic` value 1.0 → chart event in stream |
| 4 | **"Top 5 funds by 1-year return"** | `aggregate` intent → `compute_metric` tool with `op=top_n` → chart spec event surfaces in UI |

### Tier B — show range

| # | Question | Demonstrates |
|---|---|---|
| 5 | **"What does this fund factsheet say about risk?"** (PDF prose) | `qualitative` intent → text chunks dominate → cross-encoder rerank visible in score breakdown |
| 6 | **"Compare 1-year returns of [Fund X] and [Fund Y]"** | Multi-entity point lookup → two canonical_ids resolved → both row chunks surface |
| 7 | **"What is the AUM of HDFC Top 100?"** (omitting Direct/Growth qualifiers) | Alias-map resolution; if it fails on first try, demonstrates how `aliases.json` would be edited |
| 8 | **Re-upload the same Excel file** (UI action) | Deduplication: chunk count in `/health` stays flat; on a freshly-loaded indexed file, confirms hash skip |
| 8b | **Switch LLM provider mid-demo** (sidebar: pick Anthropic, paste key, Save, ask the same Tier-A question) | Pluggability — header badge flips from `Local: qwen2.5` to `Anthropic: claude-sonnet-4-6`; answer quality lift visible on qualitative prompts. Same prompt, different generation backend. |
| 8c | **Upload an Excel, watch the Capacity card** (point at the sidebar) | StatsCard refreshes: `Total rows` jumps by the row count of the file; `Fundlenz` RSS ticks up; `RAM free` drops. Then restart the backend, refresh — `Total rows` is still correct (persisted in SQLite). Shows operational visibility, not just functional. |

### Tier C — show the honest limits

| # | Question | Demonstrates |
|---|---|---|
| 9 | **"NAV trend for [Fund X] over last year"** on snapshot data | Graceful degradation — `trend` intent fires, but no time_window chunks exist (snapshot data), so it falls back to row chunk + the LLM acknowledges no time history. **No hallucinated trend.** |
| 10 | **"Which fund manager has the longest tenure?"** when that column doesn't exist | The LLM should say it cannot answer without that column — demonstrates grounding. If you want to hammer the point: ask, then check the source cards — none of them carry that field. |

### What to point at on screen during the demo
- **Source cards** — show that they cite specific files and pages (provenance)
- **Confidence badge** — point at the tier color; explain the difference
- **Streaming tokens** — call out that this is SSE, not polling
- **Chart event** — when a tool runs, the chart specifying appears alongside the text answer
- **Sidebar toggles** — show that disabling cross-encoder changes the answer ordering

---

## 15. Anticipated Interview Questions — short answers

**Q: Why default to local LLM instead of GPT-4 / Claude?**
*"Cost predictability, no data egress, and the chatbot works for any user the moment they clone the repo — no API keys, no credit card. Quality on the 7B model is acceptable for retrieval-grounded answers because most of the heavy lifting is in retrieval, not generation. But I also wired in cloud routing: a user can paste an Anthropic or OpenAI key in the UI and the next message goes through that provider. It's a per-session, browser-local choice — the backend has no persistent secret state. So the default is private and free, and the opt-in is one paste away."*

**Q: How does the cloud LLM integration actually work?**
*"Three clients implement the same `stream_chat` signature — ollama_client, anthropic_client, openai_client — each yields a uniform `{type: 'token' | 'tool_call', ...}` event stream. A small router.py picks the client based on whether the request body carries a valid (provider, api_key) pair. The chat handler doesn't know which backend it's talking to. The trickiest part is tool-schema conversion: OpenAI and Ollama share the same `{type: function, ...}` format, but Anthropic uses a different flat shape with `input_schema`. The Anthropic client owns that conversion internally — I didn't introduce a neutral intermediate format because that would force complexity on the two clients that don't need it."*

**Q: Why FAISS and not Pinecone / Weaviate / Qdrant?**
*"Single-file persistence, no service to run, embedded in process. At this corpus size none of the cloud vector DBs would outperform a flat index. If I scaled to 1M+ chunks I'd evaluate Qdrant for the metadata-filter performance."*

**Q: Why not a graph database for entity relationships?**
*"The relationships I have are flat — entity → attributes, entity → time-series. No multi-hop. SQLite + canonical_id covers it. A graph DB would be premature complexity."*

**Q: How do you evaluate retrieval quality?**
*"For deterministic intents, I assert top-1 must be the right chunk type with the right canonical_id — that's a hard-coded recall@1=1.0 contract. For qualitative, I use MRR@10 against a hand-curated query set. Latency I measure p50/p95 of `retrieve_v2`."*

**Q: What was the hardest bug?**
*"Re-uploading a file silently doubled all indexes because every ingestion creates fresh UUIDs. SQLite's INSERT OR REPLACE never matched. Fixed with delete-before-add plus a SHA-256 hash skip at the upload boundary — two layers because the hash skip avoids unnecessary work, and the delete is the safety net."*

**Q: What would you do differently?**
*"Two things. First, the asymmetric embedder didn't help on row chunks because they're `key=value` strings, not natural prose — I'd train a small adapter or rewrite row chunks as natural sentences. Second, intent routing is rules-based; for harder intents I'd add an LLM-fallback router with a small model — keep rules for the 80% but escalate ambiguous queries."*

**Q: How do you handle hallucination?**
*"Three guardrails. One: deterministic fast-paths for numeric thresholds and aggregations bypass the LLM entirely — pandas computes the answer. Two: source cards with citations are surfaced in the UI; the user sees what the answer is grounded in. Three: confidence tier signals when the answer is from semantic search alone vs. a tool execution. The LLM is only generating prose summaries over retrieved facts, not inventing data."*

**Q: How does the cross-encoder help compared to just bi-encoder?**
*"The bi-encoder embeds query and chunk independently and compares them — it can't see them together. The cross-encoder concatenates them and runs a transformer over the pair, so it can attend to lexical overlap and word-level relationships the bi-encoder misses. The cost is you can't cache the chunk encoding — it has to be recomputed for every (query, chunk) pair. So I only run it on 20 candidates after FAISS has narrowed the field."*

**Q: Why nine chunk types? Isn't that a lot?**
*"Each type has a job. Drop the entity chunk and alias resolution breaks. Drop the enumeration chunk and 'list all categories' falls back to fuzzy ANN over row chunks, which dilutes. Drop time_window and trend queries get answered from snapshot row chunks, which is wrong. They're not redundant — they each cover a question shape that the others handle poorly."*

---

## 16. Architecture Diagram (talk-through)

```
                   FRONTEND (React + Vite)
                          │
                  POST /ingest, /chat
                          │
                          ▼
                   FastAPI + SSE
                          │
        ┌─────────┬───────┴──────┬────────┬─────────┐
        │         │              │        │         │
   ingest_file() /chat handler  /settings /stats  /llm/local
        │         │              │        │         │
        │   rewrite_query()      │   psutil + meta  │
        │   classify_intent()    │   .total_rows()  │
        │   retrieve_v2()        │                  │
        │         │              │                  │
        ▼         ▼              ▼                  ▼
                   CompositeIndex
        ┌──────────┬──────────┬──────────┬──────────┐
        │ TextANN  │ Inverted │  Meta    │ NumericANN│
        │ FAISS    │   JSON   │ SQLite   │  FAISS    │
        │ 384-dim  │  id/cell │ chunks   │ per file  │
        │ FlatIP   │   /enum  │  table   │  GATED    │
        └──────────┴──────────┴──────────┴──────────┘
                          │
                  format_context_v2()
                          │
                   build_messages()
                          │
                          ▼
                   llm/router.stream_chat()
                  ┌────────┼────────┐
                  ▼        ▼        ▼
              Ollama   Anthropic  OpenAI
                          │
         compute_metric() / query_table() (tool calls)
                          │
                  SSE: sources, confidence,
                       token, chart, done
```

### How to talk through it
*"Top-down. Frontend hits FastAPI. Ingest fans out to chunkers, all chunks land in one Composite Index — four stores, each optimized for a different query shape. Query goes through query rewrite, intent classification, then the orchestrator that combines exact + semantic + rerank. Context is formatted with citations, then the LLM router picks between local Ollama and a cloud provider based on whether the request carries an API key. Tokens stream back as SSE events."*

---

## 17. The Honest Trade-offs (mention these unprompted — shows maturity)

1. **Per-row chunks bloat the index** — 200-row Excel becomes ~410 chunks. Fine on a laptop, would matter at 100× scale.
2. **Numeric ANN is overkill at current scale** — pandas is faster and exact. Code path exists; flag is off.
3. **Asymmetric embedder advantage erodes on row chunks** — they're not natural prose. Mitigation: deterministic exact-match.
4. **Rule-based intent classifier has blind spots** — anything outside its keyword list falls through to `qualitative`. LLM fallback would help; not implemented.
5. **No multi-hop reasoning** — can't answer "which fund has the highest return *among debt funds*" if "debt fund" categorization is implicit. The LLM tries but it's not deterministic.
6. **SQLite Windows file-locking risk** during concurrent ingest + read. Mitigation: stop server before bulk migration.
7. **Time-window chunks need actual time-series data** — snapshot files don't get them. By design — no fabricated trends.
8. **Cloud-LLM API key travels in the `/chat` request body** — fine on localhost (no network hop), but if I ever exposed this backend over a network I'd need TLS to keep the key off the wire. The backend never persists it, but it does briefly hold it in process memory during the API call.
9. **Capacity tracks RAM, not disk.** Disk fills up rarely on a modern laptop; RAM is the real ceiling because DataFrames + FAISS sit in process memory. If a user did fill the data partition, they'd see a disk-full OSError, not a graceful "out of capacity" message. Acceptable for a local app; would need both indicators if this ever went multi-tenant.

---

## Cheat Sheet — The Numbers

- Embedder: 384-dim, max_seq 512
- Chunk size: 180 words, overlap 40
- Top-k: 10, over-fetch: 40 (max(k*4, 20))
- CE top-n: 20
- Type prior weight: 0.20
- Score formula: 1.0×id + 0.4×cell + 0.5×text + 0.3×numeric + 0.2×prior
- LLM default: qwen2.5:7b (Ollama, port 11434)
- Cloud defaults: claude-sonnet-4-6 (Anthropic), gpt-4o (OpenAI) — set in `*_client.py`
- LLM clients: 3 (ollama, anthropic, openai), 1 router, same `stream_chat` signature
- Frontend: localhost:5173, backend: 127.0.0.1:8000
- Confidence tiers: deterministic=1.0, grounded≥0.85, semantic<0.85
- Time windows: 30d / 90d / 365d (CAGR on 365d only)
- Enumeration cardinality: 2–500
- Row chunk cap: 500/table
- Capacity card thresholds: amber ≥75% RAM used, red ≥90% (warn-only, never blocks)
- Row count persistence: `file_stats` table in `metadata.sqlite`, written at ingest, cascade-deleted on file delete
- Tests: 208 (203 original + 5 file_stats)

---

**Final tip during the interview:** When you cite a number, it sounds more confident if you back it with the *why*. "180 words" is a fact; "180 words because the embedder caps at 512 tokens and that gives us headroom plus a 22% overlap" is a story. Tell the story.
