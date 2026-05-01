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
