import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app import state
from app.analysis.charts import chart_spec
from app.analysis.column_match import resolve_column
from app.analysis.metrics import compute
from app.analysis.query import query_table
from app.config import settings
from app.ingest import ingest_file
from app.llm.ollama_client import build_messages, rewrite_query
from app.llm.router import stream_chat
from app.llm.tools import TOOL_COMPUTE_METRIC, TOOL_QUERY_TABLE, TOOLS
from app.parsers.pdf import extract_pdf_tables_as_csv
from app.retrieval.orchestrator import format_context_v2, retrieve_v2
from app.retrieval.router import classify_intent


# Aggregation-leaning words that should expose compute_metric.
_QUANT_KEYWORDS = {
    "average", "avg", "mean", "median", "sum", "total", "top", "bottom",
    "trend", "plot", "chart", "graph", "calculate", "compute",
    "highest", "lowest", "max", "maximum", "min", "minimum",
    "compare", "rank", "count", "aggregate", "by year", "by month",
    "analyz", "analys", "breakdown", "distribution", "histogram",
}

# Row-query phrases that should expose query_table. Multi-word phrases are
# preferred — bare "list", "show the", "list the" catch too many qualitative
# questions ("show the fund manager bio", "list the assumptions").
_QUERY_KEYWORDS = {
    "list all", "show me", "show all",
    "find all", "find the", "find me",
    "which fund", "which row",
    "where ", "filter", "rows with", "rows where",
    "funds with", "stocks with", "entries with", "items with",
    "give me all", "give me the",
}


def _should_enable_tools(message: str) -> bool:
    """Local 7B-class models over-call tools when any are registered — only
    expose them when the message looks plausibly tabular. A tabular file must
    also be ingested; otherwise the tools have no DataFrame to operate on."""
    if not state.filename_to_file_id:
        return False
    msg_lower = message.lower()
    if any(kw in msg_lower for kw in _QUANT_KEYWORDS):
        return True
    if any(kw in msg_lower for kw in _QUERY_KEYWORDS):
        return True
    return False


def _select_tools(message: str, mode: str) -> list | None:
    """Decide which tools to expose to the LLM. The user can force a specific
    tool from the UI mode selector (`chat` / `aggregate` / `query`); otherwise
    `auto` falls back to the keyword-gated heuristic.

    Returns None to mean "no tools sent to Ollama" (model stays in plain text)."""
    has_tabular = bool(state.filename_to_file_id)

    if mode == "chat":
        return None
    if mode == "aggregate":
        return [TOOL_COMPUTE_METRIC] if has_tabular else None
    if mode == "query":
        return [TOOL_QUERY_TABLE] if has_tabular else None
    # mode == "auto" (default)
    return TOOLS if _should_enable_tools(message) else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Run in a thread — faiss.read_index + sqlite open are blocking I/O.
    await asyncio.to_thread(state.composite.load_or_init)
    yield


app = FastAPI(title="Fundlenz API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SettingsPatch(BaseModel):
    enable_numeric_ann: bool | None = None
    enable_cross_encoder: bool | None = None


def _settings_snapshot() -> dict:
    return {
        "enable_numeric_ann": state.composite.enable_numeric_ann,
        "enable_cross_encoder": settings.enable_cross_encoder,
    }


@app.get("/settings")
def get_settings() -> dict:
    return _settings_snapshot()


@app.patch("/settings")
def patch_settings(body: SettingsPatch) -> dict:
    if body.enable_numeric_ann is not None:
        state.composite.enable_numeric_ann = body.enable_numeric_ann
    if body.enable_cross_encoder is not None:
        settings.enable_cross_encoder = body.enable_cross_encoder
    return _settings_snapshot()


@app.get("/llm/local")
def get_llm_local() -> dict:
    """The local LLM the backend falls back to when no API key is provided.
    The frontend uses this to render the LLM badge when no cloud config is set."""
    return {"provider": "ollama", "model": settings.ollama_model}


@app.get("/health")
def health() -> dict:
    stats = state.composite.stats()
    return {
        "status": "ok",
        "indexed_chunks": stats["text_chunks"],
        "by_type": stats["by_type"],
        "inverted_postings": stats["inverted_postings"],
        "tabular_files": list(state.filename_to_file_id.keys()),
        "files": stats["files"],
        "enable_numeric_ann": stats["enable_numeric_ann"],
    }


@app.get("/documents")
def list_documents() -> list[dict]:
    files = state.composite.meta.list_files()
    return [{"filename": fn, **state.documents.get(fn, {})} for fn in files]


@app.delete("/documents/{filename}")
async def delete_document(filename: str) -> dict:
    deleted = await asyncio.to_thread(state.composite.delete_by_file, filename)
    if not deleted:
        raise HTTPException(404, f"No document named '{filename}' in the index.")
    await asyncio.to_thread(state.composite.save)
    state.documents.pop(filename, None)
    file_id = state.filename_to_file_id.pop(filename, None)
    if file_id:
        state.dataframes_by_file_id.pop(file_id, None)
    return {"deleted": filename, "chunks_removed": deleted}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(400, "missing filename")
    dest = settings.upload_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = await asyncio.to_thread(ingest_file, dest, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if dest.suffix.lower() != ".pdf":
        return result

    # PDF post-processing: extract any tables as CSV and ingest each as a
    # tabular file. We isolate per-CSV failures so one bad table can't suppress
    # the rest, and we surface every attempt in the response so the UI/upload
    # status can show what landed and what didn't.
    extracted: list[dict] = []
    try:
        csv_paths = await asyncio.to_thread(
            extract_pdf_tables_as_csv, dest, settings.upload_dir
        )
    except Exception as e:
        print(f"[ingest] PDF table extraction crashed for {file.filename}: {e}")
        csv_paths = []

    total_table_chunks = 0
    summary_lines: list[str] = []
    for csv_path in csv_paths:
        try:
            csv_result = await asyncio.to_thread(
                ingest_file, csv_path, csv_path.name
            )
        except Exception as e:
            extracted.append(
                {"filename": csv_path.name, "chunks": 0, "error": str(e)}
            )
            print(f"[ingest] CSV ingest failed for {csv_path.name}: {e}")
            continue
        extracted.append(
            {"filename": csv_path.name, "chunks": csv_result["chunks"]}
        )
        total_table_chunks += csv_result["chunks"]
        if csv_result["chunks"] > 0:
            summary_lines.append(
                f"- `{csv_path.name}` — {csv_result['chunks']} chunks"
            )
        else:
            # Tabular pipeline accepted the CSV but produced nothing — usually
            # means parse_tabular dropped every row as empty after cleaning.
            summary_lines.append(
                f"- `{csv_path.name}` — 0 chunks (cleaned to empty; not queryable)"
            )

    if extracted:
        result = dict(result)
        result["chunks"] = result.get("chunks", 0) + total_table_chunks
        result["extracted_tables"] = extracted
        if summary_lines:
            existing = result.get("summary", "")
            joined = "\n".join(summary_lines)
            header = f"\n\nExtracted {len(extracted)} table(s) from PDF:\n"
            result["summary"] = (existing + header + joined).strip()

    return result


class LLMConfig(BaseModel):
    provider: str  # "anthropic" | "openai"
    api_key: str
    model: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "auto"  # "auto" | "chat" | "aggregate" | "query"
    llm: LLMConfig | None = None


@app.post("/chat")
async def chat(req: ChatRequest):
    history = list(state.chat_history[req.session_id])
    # Fold prior turns into the retrieval query so follow-ups like "what is its NAV?"
    # find the entity from earlier turns. Generation still sees the original message.
    search_query = await rewrite_query(req.message, history)
    # Classify intent on the LITERAL message (req.message), not the rewritten one.
    # The rewriter can hijack broad-intent queries — e.g. "List all benchmarks"
    # → "List benchmarks of HDFC Top 100" — flipping list_distinct to
    # point_lookup and surfacing entity chunks instead of the enumeration chunk.
    # Entity phrases from the rewritten query are merged in so coreference
    # resolution ("its NAV") still works for queries that genuinely need it.
    plan = classify_intent(req.message)
    if search_query != req.message:
        rewritten_plan = classify_intent(search_query)
        seen = set(plan.raw_entity_phrases)
        for phrase in rewritten_plan.raw_entity_phrases:
            if phrase not in seen:
                plan.raw_entity_phrases.append(phrase)
                seen.add(phrase)
    chunks = await asyncio.to_thread(
        retrieve_v2, search_query, state.composite, settings.top_k, plan
    )
    context = format_context_v2(chunks)
    sources = [_source_card(c) for c in chunks]
    messages = build_messages(
        req.message,
        context,
        history,
        tabular_files=list(state.filename_to_file_id.keys()),
    )

    async def event_stream() -> AsyncIterator[dict]:
        yield {"event": "sources", "data": json.dumps(sources)}

        # Forced-mode pre-check: a friendlier failure than letting the LLM
        # hallucinate a tool call against a non-existent file.
        if req.mode in ("aggregate", "query") and not state.filename_to_file_id:
            msg = (
                "No tabular file (CSV or Excel) is currently uploaded. "
                "Upload one to use this mode, or switch back to Auto."
            )
            state.chat_history[req.session_id].append({"role": "user", "content": req.message})
            state.chat_history[req.session_id].append({"role": "assistant", "content": msg})
            conf = {"tier": "semantic", "value": 0.0, "reason": "no tabular file uploaded"}
            yield {"event": "confidence", "data": json.dumps(conf)}
            yield {"event": "token", "data": msg}
            yield {"event": "done", "data": ""}
            return

        # Numeric-threshold deterministic bypass. The router already extracts
        # `(op, value)` from the query — pair it with a column resolved from the
        # query phrase and we can run pandas filtering directly. This avoids the
        # 7B-class model's well-documented unreliability on numeric comparisons.
        # Run on req.message (literal user text), not search_query: rewrite_query
        # can rephrase "less than 100" into a form the threshold regex misses.
        if req.mode != "chat":
            bypass = _try_numeric_threshold_bypass(req.message)
            if bypass is not None:
                full_answer, _, column, op, value, n_rows, n_files = bypass
                if n_files > 1:
                    reason = (
                        f"pandas filter on {column} {op} {value} "
                        f"across {n_files} file(s)"
                    )
                else:
                    reason = f"pandas filter on {column} {op} {value}"
                conf = {"tier": "deterministic", "value": 1.0, "reason": reason}
                yield {"event": "confidence", "data": json.dumps(conf)}
                yield {"event": "token", "data": full_answer}
                state.chat_history[req.session_id].append({"role": "user", "content": req.message})
                state.chat_history[req.session_id].append({"role": "assistant", "content": full_answer})
                yield {"event": "done", "data": ""}
                return

        full_answer: str = ""
        tool_call: dict | None = None
        tool_executed = False

        tools = _select_tools(req.message, req.mode)

        llm_cfg = req.llm.model_dump() if req.llm else None
        try:
            async for item in stream_chat(messages, tools=tools, llm_config=llm_cfg):
                if item["type"] == "token":
                    full_answer += item["content"]
                    yield {"event": "token", "data": item["content"]}
                elif item["type"] == "tool_call":
                    tool_call = item
                    break
        except Exception as e:
            err = f"Model error: {e}"
            full_answer = err
            yield {"event": "token", "data": err}

        if tool_call and tool_call["name"] == "compute_metric":
            try:
                payload = await asyncio.to_thread(_run_compute_metric, tool_call["arguments"])
                yield {"event": "chart", "data": json.dumps(payload)}
                full_answer = payload["text"]
                tool_executed = True
                yield {"event": "token", "data": full_answer}
            except Exception as e:
                err = f"Could not compute metric: {e}"
                full_answer = err
                yield {"event": "token", "data": err}
        elif tool_call and tool_call["name"] == "query_table":
            try:
                full_answer = await asyncio.to_thread(_run_query_table, tool_call["arguments"])
                tool_executed = True
                yield {"event": "token", "data": full_answer}
            except Exception as e:
                err = f"Could not query table: {e}"
                full_answer = err
                yield {"event": "token", "data": err}

        conf = _confidence_for_llm_path(chunks, tool_executed=tool_executed)
        yield {"event": "confidence", "data": json.dumps(conf)}

        state.chat_history[req.session_id].append({"role": "user", "content": req.message})
        state.chat_history[req.session_id].append({"role": "assistant", "content": full_answer})
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())


def _try_numeric_threshold_bypass(
    query: str,
) -> tuple[str, str, str, str, float, int, int] | None:
    """Run a deterministic pandas filter when the query is a numeric-threshold
    question (e.g. "Funds with AUM > 50000"). Returns
    (answer_md, primary_filename, primary_column, op, value, n_rows, n_files)
    on success, or None to fall through to the LLM path.

    Aggregates across every loaded tabular file whose schema contains a column
    matching the query phrase — funds in different uploads should all show up
    in the result. `primary_filename` / `primary_column` reference the first
    contributing file (used only for telemetry / the confidence reason).

    Falls through when:
      - intent is not numeric_threshold
      - no tabular file is loaded
      - no column in any loaded file matches the query phrase
    """
    plan = classify_intent(query)
    if plan.intent != "numeric_threshold" or plan.threshold is None:
        return None
    if not state.filename_to_file_id:
        return None

    op, value = plan.threshold

    # Every file with a token-overlapping numeric column joins the search.
    matched: list[tuple[str, str, pd.DataFrame]] = []
    for filename, file_id in state.filename_to_file_id.items():
        df = state.dataframes_by_file_id.get(file_id)
        if df is None:
            continue
        col, score = resolve_column(query, df)
        if col and score > 0:
            matched.append((filename, col, df))
    if not matched:
        return None

    per_file: list[tuple[str, str, pd.DataFrame]] = []
    for filename, column, df in matched:
        result = query_table(
            df,
            filters=[{"column": column, "op": op, "value": value}],
            limit=50,
        )
        per_file.append((filename, column, result))

    contributing = [(fn, col, r) for fn, col, r in per_file if not r.empty]
    total_rows = sum(len(r) for _, _, r in per_file)
    n_files_searched = len(per_file)

    if total_rows == 0:
        files_searched = ", ".join(f"`{fn}`" for fn, _, _ in per_file)
        cols_unique = sorted({col for _, col, _ in per_file})
        cols_label = ", ".join(f"`{c}`" for c in cols_unique)
        msg = (
            f"No rows match {cols_label} {op} {value} across "
            f"{n_files_searched} file(s): {files_searched}. "
            f"(Filter applied directly via pandas — no LLM inference involved.)"
        )
        return msg, per_file[0][0], per_file[0][1], op, float(value), 0, n_files_searched

    if len(contributing) == 1:
        filename, column, result = contributing[0]
        summary = (
            f"{len(result)} row(s) in `{filename}` where "
            f"`{column}` {op} {value} (filter applied directly via pandas)."
        )
        return (
            f"{summary}\n\n{result.to_markdown(index=False)}",
            filename, column, op, float(value), len(result), 1,
        )

    pieces = []
    for filename, _, result in contributing:
        tagged = result.copy()
        tagged.insert(0, "Source File", filename)
        pieces.append(tagged)
    combined = pd.concat(pieces, ignore_index=True)
    cols_unique = sorted({col for _, col, _ in contributing})
    cols_label = ", ".join(f"`{c}`" for c in cols_unique)
    breakdown = ", ".join(f"`{fn}` ({len(r)})" for fn, _, r in contributing)
    summary = (
        f"{total_rows} row(s) where {cols_label} {op} {value} across "
        f"{len(contributing)} file(s): {breakdown}."
    )
    return (
        f"{summary}\n\n{combined.to_markdown(index=False)}",
        contributing[0][0], contributing[0][1], op, float(value),
        total_rows, len(contributing),
    )


def _confidence_for_llm_path(chunks: list, *, tool_executed: bool) -> dict:
    """Determinism tier for the answer. `deterministic` when a tool ran or all
    top sources are exact-id matches; `grounded` when retrieval scored ≥ 0.85
    on average; `semantic` otherwise."""
    if tool_executed:
        return {
            "tier": "deterministic",
            "value": 1.0,
            "reason": "deterministic tool execution",
        }
    if not chunks:
        return {"tier": "semantic", "value": 0.0, "reason": "no relevant sources"}

    top = chunks[:3]
    if all((c.get("_score_breakdown") or {}).get("exact_id") for c in top):
        return {
            "tier": "deterministic",
            "value": 1.0,
            "reason": "all top sources are exact entity matches",
        }

    # Average displayed source score (the same 0–1 number the source cards show).
    scores = [_source_card(c)["score"] for c in top]
    avg = sum(scores) / len(scores)
    if avg >= 0.85:
        return {
            "tier": "grounded",
            "value": round(avg, 3),
            "reason": f"strong source overlap (avg {round(avg * 100)}%)",
        }
    return {
        "tier": "semantic",
        "value": round(avg, 3),
        "reason": f"semantic-only retrieval (avg {round(avg * 100)}%)",
    }


def _source_card(c: dict) -> dict:
    """Map a retrieved chunk to a UI-friendly source card. The displayed
    `score` is a 0-1 confidence (not the internal ranking score):

      exact_id match  -> 1.00  (badge: "exact match")
      exact_cell hit  -> 0.95  (badge: "field match")
      otherwise       -> raw text cosine similarity   (badge: "semantic")

    `_rank_score` and `_score_breakdown` are still attached for debugging."""
    bd = c.get("_score_breakdown") or {}
    if bd.get("exact_id"):
        score = 1.00
        match = "exact"
    elif bd.get("exact_cell"):
        score = 0.95
        match = "field"
    else:
        score = float(bd.get("text_sim", 0.0))
        match = "semantic"
    return {
        "filename": c["file"],
        "page": c.get("page"),
        "type": c["chunk_type"],
        "canonical_id": c.get("canonical_id"),
        "score": round(score, 3),
        "match": match,
        "rank_score": round(float(c.get("_score", 0.0)), 3),
    }


def _run_compute_metric(args: dict) -> dict:
    filename = args["filename"]
    file_id = state.filename_to_file_id.get(filename)
    if file_id is None:
        raise ValueError(
            f"No tabular file named '{filename}' has been ingested. "
            f"Available: {list(state.filename_to_file_id.keys())}"
        )
    df = state.dataframes_by_file_id[file_id]
    op = args["op"]
    column = args["column"]
    group_by = args.get("group_by")
    n = args.get("n")
    result = compute(df, op, column, group_by, n)
    spec = chart_spec(result, op, column, group_by)
    text = f"Computed {op} of `{column}`" + (f" by `{group_by}`" if group_by else "") + f" from {filename}."
    return {"text": text, "chart_spec": spec, "filename": filename, "op": op}


def _run_query_table(args: dict) -> str:
    """Apply query_table and return a markdown-formatted answer string."""
    filename = args["filename"]
    file_id = state.filename_to_file_id.get(filename)
    if file_id is None:
        raise ValueError(
            f"No tabular file named '{filename}' has been ingested. "
            f"Available: {list(state.filename_to_file_id.keys())}"
        )
    df = state.dataframes_by_file_id[file_id]
    result = query_table(
        df,
        filters=args.get("filters") or [],
        sort_by=args.get("sort_by"),
        sort_desc=args.get("sort_desc", True),
        select_columns=args.get("select_columns") or [],
        limit=args.get("limit") or 50,
    )
    if result.empty:
        return f"No rows matched in {filename}."
    table_md = result.to_markdown(index=False)
    summary_line = f"Found {len(result)} row(s) in {filename}."
    return f"{summary_line}\n\n{table_md}"
