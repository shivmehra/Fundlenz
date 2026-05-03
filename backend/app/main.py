import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app import state
from app.analysis.charts import chart_spec
from app.analysis.metrics import compute
from app.analysis.query import query_table
from app.config import settings
from app.ingest import ingest_file
from app.llm.ollama_client import build_messages, rewrite_query, stream_chat
from app.llm.tools import TOOL_COMPUTE_METRIC, TOOL_QUERY_TABLE, TOOLS
from app.retrieval.orchestrator import format_context_v2, retrieve_v2


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
        return await asyncio.to_thread(ingest_file, dest, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))


class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "auto"  # "auto" | "chat" | "aggregate" | "query"


@app.post("/chat")
async def chat(req: ChatRequest):
    history = list(state.chat_history[req.session_id])
    # Fold prior turns into the retrieval query so follow-ups like "what is its NAV?"
    # find the entity from earlier turns. Generation still sees the original message.
    search_query = await rewrite_query(req.message, history)
    chunks = await asyncio.to_thread(
        retrieve_v2, search_query, state.composite, settings.top_k
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
            yield {"event": "token", "data": msg}
            yield {"event": "done", "data": ""}
            return

        full_answer: str = ""
        tool_call: dict | None = None

        tools = _select_tools(req.message, req.mode)

        try:
            async for item in stream_chat(messages, tools=tools):
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
                yield {"event": "token", "data": full_answer}
            except Exception as e:
                err = f"Could not compute metric: {e}"
                full_answer = err
                yield {"event": "token", "data": err}
        elif tool_call and tool_call["name"] == "query_table":
            try:
                full_answer = await asyncio.to_thread(_run_query_table, tool_call["arguments"])
                yield {"event": "token", "data": full_answer}
            except Exception as e:
                err = f"Could not query table: {e}"
                full_answer = err
                yield {"event": "token", "data": err}

        state.chat_history[req.session_id].append({"role": "user", "content": req.message})
        state.chat_history[req.session_id].append({"role": "assistant", "content": full_answer})
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())


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
