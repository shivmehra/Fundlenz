from typing import AsyncIterator

from app.llm import anthropic_client, ollama_client, openai_client


def _valid_cloud_config(cfg: dict | None) -> bool:
    if not cfg:
        return False
    provider = cfg.get("provider")
    api_key = cfg.get("api_key")
    return bool(provider) and bool(api_key) and provider in ("anthropic", "openai")


async def stream_chat(
    messages: list[dict],
    tools: list | None = None,
    *,
    llm_config: dict | None = None,
) -> AsyncIterator[dict]:
    """Route to the correct LLM client.

    If `llm_config` carries a valid (provider, api_key) pair, the corresponding
    cloud client is used. Otherwise falls through to the local Ollama client.
    No silent fallback on cloud failure — exceptions propagate so /chat can
    surface them to the user via the SSE error event."""
    if _valid_cloud_config(llm_config):
        provider = llm_config["provider"]
        api_key = llm_config["api_key"]
        model = llm_config.get("model") or None
        if provider == "anthropic":
            async for item in anthropic_client.stream_chat(
                messages, tools, api_key=api_key, model=model
            ):
                yield item
            return
        if provider == "openai":
            async for item in openai_client.stream_chat(
                messages, tools, api_key=api_key, model=model
            ):
                yield item
            return

    async for item in ollama_client.stream_chat(messages, tools):
        yield item
