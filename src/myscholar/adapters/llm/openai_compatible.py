"""Hosted inference over the OpenAI chat-completions shape.

The production counterpart to the local Ollama adapter: same interface, same
prompts, same pipeline, different machine. Named for the wire format rather than
a vendor, because several providers speak it and the choice between them comes
down to a ``base_url`` and a model name.

The default points at Google AI Studio, which is what this project deploys
against — its free tier is the only one measured to run the whole pipeline. Two
requirements narrow the field more than they appear to:

* **Schema-constrained output.** Classification, premise checks and the
  groundedness gate are all ``structured`` calls, and a provider that returns
  prose instead of JSON does not fail loudly — every caller treats a validation
  error as an outage and degrades quietly, so those stages simply stop working.
  Ollama Cloud ignores the schema parameter entirely, which rules it out.
* **Room for one large request.** A generation prompt carrying eight context
  blocks runs to roughly 10,000 tokens. A free tier that caps a *minute* at
  8,000 can never produce a single answer, however little traffic it sees.

Providers also disagree about unknown parameters: Google rejects the whole
request with ``Unknown name "seed"`` where others ignore the field. So ``seed``
and ``reasoning_effort`` are both optional and unset by default.
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

#: Statuses worth trying again: transient capacity, not a verdict about the
#: request. 429 is excluded on purpose and raises immediately.
_TRANSIENT_STATUS = frozenset({500, 502, 503, 504})
_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.0


class HostedModelError(RuntimeError):
    """The hosted endpoint was unreachable, rate-limited, or returned garbage."""


class HostedModelRateLimitError(HostedModelError):
    """The free tier's quota is exhausted.

    Distinct from other failures because the caller should degrade rather than
    retry: on a public portfolio, a graceful "temporarily unavailable" beats a
    retry storm against a quota that will not reset for an hour.
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
        # A fixed seed makes runs comparable, which the evaluations depend on.
        # Optional because it is an OpenAI extension and not every compatible
        # endpoint tolerates it — Google's rejects the whole request with
        # `Unknown name "seed": Cannot find field` rather than ignoring it.
        if self._seed is not None:
            payload["seed"] = self._seed
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # Only sent when configured. The gpt-oss models keep their reasoning in
        # a separate `reasoning` field rather than in `content`, so this changes
        # cost and latency rather than what the answer looks like — but a model
        # that does not accept the parameter rejects the whole request, so an
        # unset default is the safe one.
        if self._reasoning_effort is not None:
            payload["reasoning_effort"] = self._reasoning_effort
        return payload

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One request, retried only for the failures that clear by themselves.

        A shared free tier answers `503 The model is currently experiencing high
        demand` under bursts, and one question makes several calls in a few
        seconds. Left alone that is worse than an outage, because nothing above
        this layer reports it: every caller of `structured` treats an exception
        as unavailability and degrades quietly, so a transient 503 turns into
        classification falling back to one class for the question, or the
        groundedness gate passing every statement through. Measured on Google's
        free tier: six spaced calls succeeded 6/6, the same six back to back
        succeeded 1/6.

        Rate limiting is deliberately *not* retried — see
        `HostedModelRateLimitError`. A quota does not clear in a second, so
        retrying only spends the little that is left.
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
