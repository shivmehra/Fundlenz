from typing import AsyncIterator

from anthropic import AsyncAnthropic


ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"


def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic puts the system prompt in a separate `system` field, not as
    a role=system message. Concatenate any system messages and drop them from
    the message list."""
    system_parts: list[str] = []
    converted: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
        elif role in ("user", "assistant"):
            converted.append({"role": role, "content": content})
    return "\n\n".join(system_parts), converted


def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
    """OpenAI-style tool schema → Anthropic tool schema.

    OpenAI:    {"type": "function", "function": {"name", "description", "parameters"}}
    Anthropic: {"name", "description", "input_schema"}
    """
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        fn = t.get("function") or t
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or fn.get("input_schema") or {},
        })
    return out


async def stream_chat(
    messages: list[dict],
    tools: list | None = None,
    *,
    api_key: str,
    model: str | None = None,
) -> AsyncIterator[dict]:
    """Stream a chat response from Anthropic. Mirrors the ollama_client.stream_chat
    event contract: yields {"type": "token", "content": str} and
    {"type": "tool_call", "name": str, "arguments": dict}.

    Anthropic emits the full content (text + tool_use blocks) as part of one
    message, so tool_use is read from the final message after streaming text."""
    client = AsyncAnthropic(api_key=api_key)
    system, anthropic_messages = _convert_messages(messages)
    anthropic_tools = _convert_tools(tools)

    kwargs: dict = {
        "model": model or ANTHROPIC_DEFAULT_MODEL,
        "max_tokens": 4096,
        "messages": anthropic_messages,
    }
    if system:
        kwargs["system"] = system
    if anthropic_tools:
        kwargs["tools"] = anthropic_tools

    async with client.messages.stream(**kwargs) as stream:
        async for chunk in stream:
            if chunk.type == "content_block_delta":
                delta = chunk.delta
                if getattr(delta, "type", None) == "text_delta":
                    yield {"type": "token", "content": delta.text}

        final = await stream.get_final_message()
        for block in final.content:
            if getattr(block, "type", None) == "tool_use":
                args = block.input
                if not isinstance(args, dict):
                    args = dict(args) if args else {}
                yield {
                    "type": "tool_call",
                    "name": block.name,
                    "arguments": args,
                }
                return
