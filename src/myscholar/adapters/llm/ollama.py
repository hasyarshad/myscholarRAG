"""Ollama backend, used for local development and evaluation.

Structured output uses Ollama's ``format`` parameter, which constrains decoding to a
JSON schema.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from myscholar.ports.llm import ChatModel, Message

T = TypeVar("T", bound=BaseModel)


class OllamaError(RuntimeError):
    """Ollama was unreachable or returned something unusable."""


#: Qwen3 suppresses its own reasoning when this appears in the prompt. Measured
#: on qwen3:4b: 51 completion tokens with it, 191 without, for the same one-word
#: answer. Ollama's ``think: false`` parameter behaves differently (see ``_payload``).
NO_THINK_DIRECTIVE = "/no_think"


class OllamaChatModel(ChatModel):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        keep_alive: int | str = -1,
        suppress_reasoning: bool = True,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        # Ollama Cloud uses the same /api/chat as a local server; only the host
        # and a bearer token differ.
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._cloud = bool(api_key)
        self._temperature = temperature
        self._timeout = timeout_seconds
        # -1 keeps the weights loaded. By default Ollama evicts a model after five
        # idle minutes and the next request waits seconds for a reload. Must be a
        # number or a duration string like "10m"; the string "-1" is rejected.
        self._keep_alive = keep_alive
        # Reasoning traces add latency and can pull a grounded answer away from
        # the retrieved context.
        self._suppress_reasoning = suppress_reasoning

    @property
    def model_id(self) -> str:
        return self._model

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self._temperature if temperature is None else temperature,
            # Fixed for reproducible evaluation runs. Ollama is still not
            # bit-deterministic across restarts.
            "seed": 42,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        wire = [{"role": m.role.value, "content": m.content} for m in messages]
        if self._suppress_reasoning and wire:
            wire[-1]["content"] += f" {NO_THINK_DIRECTIVE}"

        # `think: false` is not sent. With qwen3 on Ollama 0.32.6 it stops Ollama
        # moving the reasoning trace into its own `thinking` field, but the model
        # still writes one, so it lands in `message.content`.
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": wire,
            "stream": stream,
            "options": options,
        }
        if not self._cloud:
            # Residency only applies to a server holding the weights itself.
            payload["keep_alive"] = self._keep_alive
        return payload

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = self._payload(
            messages, temperature=temperature, max_tokens=max_tokens, stream=False
        )
        async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
            try:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                msg = f"Ollama request failed against {self._base_url}: {exc}"
                raise OllamaError(msg) from exc

        content = response.json().get("message", {}).get("content", "")
        return str(content)

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
            httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client,
            client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if token := chunk.get("message", {}).get("content"):
                    yield str(token)
                if chunk.get("done"):
                    break

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        payload = self._payload(messages, temperature=temperature, max_tokens=None, stream=False)
        # Constrained decoding: the sampler only emits tokens valid against the schema.
        payload["format"] = schema.model_json_schema()

        async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
            try:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                msg = f"Ollama request failed against {self._base_url}: {exc}"
                raise OllamaError(msg) from exc

        content = response.json().get("message", {}).get("content", "")
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            msg = f"Ollama returned output not matching {schema.__name__}: {content[:200]}"
            raise OllamaError(msg) from exc
