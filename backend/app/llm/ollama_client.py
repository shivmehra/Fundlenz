from typing import AsyncIterator

import ollama

from app.config import settings
from app.llm.tools import SYSTEM_PROMPT


_client = ollama.AsyncClient(host=settings.ollama_host)


def build_messages(
    question: str,
    context: str,
    history: list[dict],
    tabular_files: list[str] | None = None,
) -> list[dict]:
    system = SYSTEM_PROMPT
    if tabular_files:
        # Stops the model from hallucinating filenames in compute_metric calls.
        system += (
            "\n\nThe following tabular files are available to compute_metric "
            "(use these exact filenames): " + ", ".join(tabular_files)
        )
    msgs: list[dict] = [{"role": "system", "content": system}]
    msgs.extend(history)
    user_content = (
        f"Context from retrieved documents:\n\n{context}\n\n"
        f"User question: {question}"
    )
    msgs.append({"role": "user", "content": user_content})
    return msgs


_REWRITE_SYSTEM = """You are a query rewriter for a retrieval system over fund documents.

Given a short conversation and the user's latest message, output a single self-contained search query that explicitly names any entity (fund name, file, page, column) the user is referring to from earlier turns. The rewritten query is fed to a vector search index, so it must contain the keywords the search needs.

Rules:
- Output ONLY the rewritten query. No preamble, no explanation, no quotes, no labels.
- If the latest message already names every entity it depends on, output it unchanged.
- Keep the output under 25 words.
- Do not invent entity names that are not present in the conversation.
- If the latest message is a greeting / acknowledgement / has no informational intent, output it unchanged.
"""


async def rewrite_query(message: str, history: list[dict]) -> str:
    """Use the LLM to fold prior conversation context into a self-contained search
    query. The original message is still what the LLM sees at generation time —
    only retrieval uses the rewritten version.

    Falls back to the original message when there is no history or on any error;
    a degraded retrieval is better than a failed chat turn."""
    if not history:
        return message

    convo_lines: list[str] = []
    for turn in history:
        role = turn.get("role", "?")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        # Trim very long assistant answers — only the topical anchor matters.
        convo_lines.append(f"{role}: {content[:400]}")
    convo_text = "\n".join(convo_lines)

    user_content = (
        f"Conversation so far:\n{convo_text}\n\n"
        f"Latest message: {message}\n\n"
        f"Rewritten query:"
    )

    try:
        resp = await _client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            stream=False,
            options={"temperature": 0.0},
        )
        rewritten = (resp.message.content or "").strip()
    except Exception:
        return message

    if not rewritten:
        return message
    # Models occasionally wrap the answer in quotes despite the instruction.
    if len(rewritten) >= 2 and rewritten[0] == rewritten[-1] and rewritten[0] in ('"', "'"):
        rewritten = rewritten[1:-1].strip()
    return rewritten or message


async def stream_chat(messages: list[dict], tools: list | None = None) -> AsyncIterator[dict]:
    """Stream a chat response. If `tools` is provided, the model may emit a tool_call.

    Yields:
      {"type": "token",     "content": str}
      {"type": "tool_call", "name": str, "arguments": dict}

    Local 7B-class models tend to call any registered tool even for questions
    that don't need one. Pass tools=None for non-quantitative questions so the
    model just answers in plain text.
    """
    kwargs: dict = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools

    async for part in await _client.chat(**kwargs):
        msg = part.message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                yield {
                    "type": "tool_call",
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
        elif msg.content:
            yield {"type": "token", "content": msg.content}
