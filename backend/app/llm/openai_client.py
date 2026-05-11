import json
from typing import AsyncIterator

from openai import AsyncOpenAI


OPENAI_DEFAULT_MODEL = "gpt-4o"


async def stream_chat(
    messages: list[dict],
    tools: list | None = None,
    *,
    api_key: str,
    model: str | None = None,
) -> AsyncIterator[dict]:
    """Stream a chat response from OpenAI. Mirrors the ollama_client.stream_chat
    event contract: yields {"type": "token", "content": str} and
    {"type": "tool_call", "name": str, "arguments": dict}.

    OpenAI's tool schema matches ours already (the ollama tool schema is
    OpenAI-compatible). Tool-call arguments arrive as JSON strings split across
    multiple delta chunks — we buffer per `tc.index` and parse once on
    `finish_reason == "tool_calls"`."""
    client = AsyncOpenAI(api_key=api_key)

    kwargs: dict = {
        "model": model or OPENAI_DEFAULT_MODEL,
        "messages": messages,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools

    tool_buffer: dict[int, dict[str, str]] = {}
    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        if delta.content:
            yield {"type": "token", "content": delta.content}

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                slot = tool_buffer.setdefault(idx, {"name": "", "arguments": ""})
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

        if choice.finish_reason == "tool_calls":
            for idx in sorted(tool_buffer.keys()):
                slot = tool_buffer[idx]
                try:
                    args = json.loads(slot["arguments"]) if slot["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                yield {
                    "type": "tool_call",
                    "name": slot["name"],
                    "arguments": args,
                }
                return
