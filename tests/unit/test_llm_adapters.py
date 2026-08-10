"""Ollama and Groq adapters, exercised against a stubbed HTTP transport.

No network and no running model server: httpx's MockTransport intercepts
requests, so these tests assert the request bodies and response handling that
would otherwise only be observable in production.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from myscholar.adapters.llm.groq import GroqChatModel, GroqError, GroqRateLimitError
from myscholar.adapters.llm.ollama import (
    NO_THINK_DIRECTIVE,
    OllamaChatModel,
    OllamaError,
)
from myscholar.ports.llm import Message, Role

QUESTION = [Message(role=Role.USER, content="what is FCS-MPCC?")]


class Verdict(BaseModel):
    relevant: bool
    reason: str


# Captured once, at import, so repeated patching within a single test always
# rebuilds from the genuine constructor rather than wrapping a previous wrapper.
_REAL_CLIENT_INIT = httpx.AsyncClient.__init__


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler: httpx.MockTransport) -> None:
    """Route every AsyncClient created afterwards through ``handler``."""

    def wrapped(self: httpx.AsyncClient, **kwargs: object) -> None:
        kwargs["transport"] = handler
        _REAL_CLIENT_INIT(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.AsyncClient, "__init__", wrapped)


class TestOllama:
    async def test_complete_returns_message_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "a finite control set method"}})

        _patch_transport(monkeypatch, httpx.MockTransport(handler))
        model = OllamaChatModel("qwen3:14b")

        assert await model.complete(QUESTION) == "a finite control set method"
        assert captured["model"] == "qwen3:14b"

    async def test_requests_are_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Temperature zero and a fixed seed, so evaluation runs are comparable."""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "x"}})

        _patch_transport(monkeypatch, httpx.MockTransport(handler))
        await OllamaChatModel("qwen3:14b").complete(QUESTION)

        options = captured["options"]
        assert isinstance(options, dict)
        assert options["temperature"] == 0.0
        assert options["seed"] == 42

    async def test_reasoning_is_suppressed_by_prompt_directive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reasoning traces cost latency and invite drift away from the context.

        Suppression happens through a prompt directive appended to the final
        message, not through Ollama's ``think`` parameter — see the next test.
        """
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "x"}})

        _patch_transport(monkeypatch, httpx.MockTransport(handler))
        await OllamaChatModel("qwen3:14b").complete(QUESTION)

        messages = captured["messages"]
        assert isinstance(messages, list)
        assert messages[-1]["content"].endswith(NO_THINK_DIRECTIVE)

    async def test_think_parameter_is_never_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression guard for a real defect found against a live server.

        Sending ``think: false`` stops Ollama separating the reasoning trace
        into its own field, but the model still produces one — so it lands in
        ``message.content`` and corrupts the answer. Omitting the parameter
        keeps ``content`` clean.
        """
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "x"}})

        _patch_transport(monkeypatch, httpx.MockTransport(handler))
        await OllamaChatModel("qwen3:14b").complete(QUESTION)

        assert "think" not in captured

    async def test_reasoning_suppression_can_be_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So the effect on answer quality can be measured in Phase 5."""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "x"}})

        _patch_transport(monkeypatch, httpx.MockTransport(handler))
        await OllamaChatModel("qwen3:14b", suppress_reasoning=False).complete(QUESTION)

        messages = captured["messages"]
        assert isinstance(messages, list)
        assert messages[-1]["content"] == "what is FCS-MPCC?"

    async def test_weights_are_kept_resident(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Otherwise every question after an idle gap pays a model reload."""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "x"}})

        _patch_transport(monkeypatch, httpx.MockTransport(handler))
        await OllamaChatModel("qwen3:14b").complete(QUESTION)

        # A number, not the string "-1" — Ollama rejects that with
        # 'missing unit in duration'. Found against a live server.
        assert captured["keep_alive"] == -1

    async def test_structured_sends_the_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Decoding is constrained to the schema rather than parsed hopefully."""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"message": {"content": '{"relevant": true, "reason": "mentions MPC"}'}},
            )

        _patch_transport(monkeypatch, httpx.MockTransport(handler))
        verdict = await OllamaChatModel("qwen3:4b").structured(QUESTION, Verdict)

        assert verdict.relevant is True
        assert captured["format"] == Verdict.model_json_schema()

    async def test_structured_rejects_output_that_does_not_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": {"content": '{"wrong": "shape"}'}})

        _patch_transport(monkeypatch, httpx.MockTransport(handler))

        with pytest.raises(OllamaError, match="not matching Verdict"):
            await OllamaChatModel("qwen3:4b").structured(QUESTION, Verdict)

    async def test_unreachable_server_raises_a_named_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The API turns this into a 503, so the failure must be identifiable."""

        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        _patch_transport(monkeypatch, httpx.MockTransport(handler))

        with pytest.raises(OllamaError, match="failed"):
            await OllamaChatModel("qwen3:14b").complete(QUESTION)

    async def test_stream_yields_tokens_until_done(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [
            json.dumps({"message": {"content": "finite "}, "done": False}),
            json.dumps({"message": {"content": "control set"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content="\n".join(lines).encode())

        _patch_transport(monkeypatch, httpx.MockTransport(handler))
        tokens = [t async for t in OllamaChatModel("qwen3:14b").stream(QUESTION)]

        assert "".join(tokens) == "finite control set"


class TestGroq:
    def _model(self) -> GroqChatModel:
        return GroqChatModel("llama-3.3-70b-versatile", api_key="gsk-test")

    async def test_complete_reads_openai_shaped_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "a control strategy"}}]}
            )

        _patch_transport(monkeypatch, httpx.MockTransport(handler))

        assert await self._model().complete(QUESTION) == "a control strategy"

    async def test_api_key_is_sent_as_a_bearer_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization", "")
            return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

        _patch_transport(monkeypatch, httpx.MockTransport(handler))
        await self._model().complete(QUESTION)

        assert captured["auth"] == "Bearer gsk-test"

    async def test_rate_limiting_raises_a_distinct_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Callers must degrade rather than retry into an exhausted free-tier quota."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"retry-after": "60"}, json={})

        _patch_transport(monkeypatch, httpx.MockTransport(handler))

        with pytest.raises(GroqRateLimitError, match="60"):
            await self._model().complete(QUESTION)

    async def test_server_error_is_reported_with_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="upstream exploded")

        _patch_transport(monkeypatch, httpx.MockTransport(handler))

        with pytest.raises(GroqError, match="500"):
            await self._model().complete(QUESTION)

    async def test_stream_parses_server_sent_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = (
            'data: {"choices":[{"delta":{"content":"finite "}}]}\n'
            'data: {"choices":[{"delta":{"content":"control"}}]}\n'
            "data: [DONE]\n"
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body.encode())

        _patch_transport(monkeypatch, httpx.MockTransport(handler))
        tokens = [t async for t in self._model().stream(QUESTION)]

        assert "".join(tokens) == "finite control"


class TestAdapterEquivalence:
    """Both backends must be interchangeable behind the port."""

    async def test_both_return_plain_text_for_the_same_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def ollama_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": {"content": "same answer"}})

        _patch_transport(monkeypatch, httpx.MockTransport(ollama_handler))
        local = await OllamaChatModel("qwen3:14b").complete(QUESTION)

        def groq_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "same answer"}}]})

        _patch_transport(monkeypatch, httpx.MockTransport(groq_handler))
        hosted = await GroqChatModel("llama-3.3-70b-versatile", api_key="k").complete(QUESTION)

        assert local == hosted == "same answer"
