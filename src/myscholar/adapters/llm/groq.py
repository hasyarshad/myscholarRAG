"""Groq backend — hosted inference in production.

Groq serves open-weight models (Llama, Qwen, Gemma) on custom hardware, with a
free tier that needs no card. It is the production counterpart to the local
Ollama adapter: same interface, same prompts, same pipeline, different machine.

The API is OpenAI-compatible, so this adapter also works against any other
OpenAI-shaped endpoint by changing ``base_url`` — useful if the free tier's
terms change.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from myscholar.ports.llm import ChatModel, Message

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


class GroqError(RuntimeError):
    """The hosted endpoint was unreachable, rate-limited, or returned garbage."""


class GroqRateLimitError(GroqError):
    """The free tier's quota is exhausted.

    Distinct from other failures because the caller should degrade rather than
    retry: on a public portfolio, a graceful "temporarily unavailable" beats a
    retry storm against a quota that will not reset for an hour.
    """


class GroqChatModel(ChatModel):
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout_seconds

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": self._temperature if temperature is None else temperature,
            "stream": stream,
            "seed": 42,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    f"{self._base_url}/chat/completions", json=payload, headers=self._headers
                )
            except httpx.HTTPError as exc:
                msg = f"Groq request failed: {exc}"
                raise GroqError(msg) from exc

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("retry-after", "unknown")
            msg = f"Groq rate limit reached; retry after {retry_after}s"
            raise GroqRateLimitError(msg)

        if response.is_error:
            msg = f"Groq returned {response.status_code}: {response.text[:200]}"
            raise GroqError(msg)

        result: dict[str, Any] = response.json()
        return result

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        body = await self._post(
            self._payload(messages, temperature=temperature, max_tokens=max_tokens, stream=False)
        )
        return str(body["choices"][0]["message"]["content"])

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(
            messages, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream(
                "POST", f"{self._base_url}/chat/completions", json=payload, headers=self._headers
            ) as response,
        ):
            if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                msg = "Groq rate limit reached"
                raise GroqRateLimitError(msg)
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                if token := delta.get("content"):
                    yield str(token)

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        payload = self._payload(messages, temperature=temperature, max_tokens=None, stream=False)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }

        body = await self._post(payload)
        content = str(body["choices"][0]["message"]["content"])
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            msg = f"Groq returned output not matching {schema.__name__}: {content[:200]}"
            raise GroqError(msg) from exc
