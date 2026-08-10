"""Ollama backend — local inference during development.

Talks to the Ollama HTTP API directly. Structured output uses Ollama's
``format`` parameter, which constrains decoding to a JSON schema rather than
asking the model nicely and parsing whatever comes back. That distinction
matters: every classification and grading decision in the pipeline depends on
getting well-formed output, and a parse failure mid-loop is a wasted round.
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


class OllamaChatModel(ChatModel):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        keep_alive: str = "-1",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout_seconds
        # "-1" keeps weights resident indefinitely. Without it Ollama evicts the
        # model after five idle minutes, and the next question pays a
        # multi-second reload — which looks like a slow pipeline, not a cold model.
        self._keep_alive = keep_alive

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
            # A fixed seed makes runs reproducible, which the evaluation harness
            # depends on. Ollama is still not bit-deterministic across restarts.
            "seed": 42,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        return {
            "model": self._model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": stream,
            "keep_alive": self._keep_alive,
            "options": options,
            # Qwen3 and similar models interleave reasoning traces by default.
            # For grounded extraction that is latency without benefit, and it is
            # a hallucination risk — the model reasons its way past the context.
            "think": False,
        }

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
        async with httpx.AsyncClient(timeout=self._timeout) as client:
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
            httpx.AsyncClient(timeout=self._timeout) as client,
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
        # Constrained decoding: the sampler can only emit tokens that keep the
        # output valid against this schema, so malformed JSON is impossible
        # rather than merely unlikely.
        payload["format"] = schema.model_json_schema()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
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
