"""LLM provider router.

Tier 1 (default, safe): Ollama, Together AI — open weights.
Tier 2 (opt-in, ToS warning): OpenAI, Anthropic, Google.
Tier 3 (advanced): any OpenAI-compatible endpoint.

The provider exposes a single `chat()` coroutine that yields text deltas.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from config import get_settings


@dataclass
class ChatMsg:
    role: str
    content: str

    def as_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMError(RuntimeError):
    pass


class BaseProvider:
    name: str = "base"

    async def stream(
        self, messages: list[ChatMsg], model: str, **kwargs: Any
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    async def complete(
        self, messages: list[ChatMsg], model: str, **kwargs: Any
    ) -> str:
        chunks: list[str] = []
        async for delta in self.stream(messages, model, **kwargs):
            chunks.append(delta)
        return "".join(chunks)


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def stream(self, messages, model, **kwargs):
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [m.as_openai() for m in messages],
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1024),
            },
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=600.0)) as client:
            async with client.stream("POST", url, json=payload) as r:
                if r.status_code != 200:
                    raise LLMError(f"Ollama HTTP {r.status_code}: {await r.aread()!r}")
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = evt.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if evt.get("done"):
                        break


class OpenAICompatibleProvider(BaseProvider):
    """Works for OpenAI, Together AI, and any OAI-compatible endpoint."""

    def __init__(self, name: str, base_url: str, api_key: str):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def stream(self, messages, model, **kwargs):
        if not self.api_key:
            raise LLMError(f"No API key configured for {self.name}.")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [m.as_openai() for m in messages],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=600.0)) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as r:
                if r.status_code != 200:
                    raise LLMError(f"{self.name} HTTP {r.status_code}: {await r.aread()!r}")
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        evt = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = evt.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content") or ""
                    if delta:
                        yield delta


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def stream(self, messages, model, **kwargs):
        if not self.api_key:
            raise LLMError("No Anthropic API key configured.")
        url = "https://api.anthropic.com/v1/messages"
        sys_msgs = [m.content for m in messages if m.role == "system"]
        chat_msgs = [m.as_openai() for m in messages if m.role != "system"]
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "messages": chat_msgs,
            "stream": True,
        }
        if sys_msgs:
            payload["system"] = "\n\n".join(sys_msgs)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=600.0)) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as r:
                if r.status_code != 200:
                    raise LLMError(f"Anthropic HTTP {r.status_code}: {await r.aread()!r}")
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        evt = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("type") == "content_block_delta":
                        delta = evt.get("delta", {}).get("text") or ""
                        if delta:
                            yield delta


def get_provider(name: str) -> BaseProvider:
    s = get_settings()
    name = (name or "ollama").lower()
    if name == "ollama":
        return OllamaProvider(s.ollama_base_url)
    if name == "together":
        return OpenAICompatibleProvider(
            "together", "https://api.together.xyz/v1", s.together_api_key
        )
    if name == "openai":
        return OpenAICompatibleProvider(
            "openai", "https://api.openai.com/v1", s.openai_api_key
        )
    if name == "anthropic":
        return AnthropicProvider(s.anthropic_api_key)
    if name == "google":
        # Google's Gemini OpenAI-compat endpoint.
        return OpenAICompatibleProvider(
            "google",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            s.google_api_key,
        )
    if name == "custom":
        if not s.custom_llm_base_url:
            raise LLMError("CUSTOM_LLM_BASE_URL is not set.")
        return OpenAICompatibleProvider(
            "custom", s.custom_llm_base_url, s.custom_llm_api_key
        )
    raise LLMError(f"Unknown provider: {name}")
