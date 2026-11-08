"""Hosted inference over the OpenAI chat-completions wire format.

Production runs Gemini through Google AI Studio's OpenAI-compatible endpoint, the
default ``base_url``. A provider must honour schema-constrained output (Ollama Cloud
ignores the schema, and ``structured`` callers fall back silently on prose) and must
accept one generation request of about 10,000 tokens within its per-minute cap.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from myscholar.ports.llm import ChatModel, Message

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

#: Transient capacity errors, worth retrying. A 429 raises immediately.
_TRANSIENT_STATUS = frozenset({500, 502, 503, 504})
_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.0


class HostedModelError(RuntimeError):
    """The hosted endpoint was unreachable, rate-limited, or returned an unusable response."""


class HostedModelRateLimitError(HostedModelError):
    """The provider's quota is exhausted.

    Not retried: a quota does not reset within seconds, so callers degrade instead.
    """


class OpenAICompatibleChatModel(ChatModel):
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        reasoning_effort: str | None = None,
        seed: int | None = 42,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._reasoning_effort = reasoning_effort
        self._seed = seed

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
        }
        # A fixed seed keeps evaluation runs comparable. Optional because some
        # compatible endpoints reject it: Google's fails the whole request with
        # `Unknown name "seed": Cannot find field`.
        if self._seed is not None:
            payload["seed"] = self._seed
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # Only sent when configured: a model that does not accept the parameter
        # rejects the whole request.
        if self._reasoning_effort is not None:
            payload["reasoning_effort"] = self._reasoning_effort
        return payload

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One request, retried only on transient 5xx errors.

        A shared free tier returns 503 under bursts, and one question makes
        several calls within seconds. Callers of ``structured`` degrade silently
        on an exception, so an unretried 503 disables a pipeline step. Measured on
        Google's free tier: six spaced calls succeeded 6/6, six back to back 1/6.
        Rate limits (429) are not retried.
        """
        last_error = ""
        for attempt in range(_ATTEMPTS):
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                try:
                    response = await client.post(
                        f"{self._base_url}/chat/completions", json=payload, headers=self._headers
                    )
                except httpx.HTTPError as exc:
                    msg = f"request to the hosted model failed: {exc}"
                    raise HostedModelError(msg) from exc

            if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                retry_after = response.headers.get("retry-after", "unknown")
                msg = f"hosted model rate limit reached; retry after {retry_after}s"
                raise HostedModelRateLimitError(msg)

            if response.status_code in _TRANSIENT_STATUS and attempt < _ATTEMPTS - 1:
                last_error = f"{response.status_code}: {response.text[:120]}"
                logger.info(
                    "hosted model returned %s, retrying (attempt %d of %d)",
                    response.status_code,
                    attempt + 2,
                    _ATTEMPTS,
                )
                await asyncio.sleep(_BACKOFF_SECONDS * (attempt + 1))
                continue

            if response.is_error:
                msg = f"hosted model returned {response.status_code}: {response.text[:200]}"
                raise HostedModelError(msg)

            result: dict[str, Any] = response.json()
            return result

        msg = f"hosted model unavailable after {_ATTEMPTS} attempts ({last_error})"
        raise HostedModelError(msg)

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
                msg = "hosted model rate limit reached"
                raise HostedModelRateLimitError(msg)
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
            msg = f"hosted model returned output not matching {schema.__name__}: {content[:200]}"
            raise HostedModelError(msg) from exc
