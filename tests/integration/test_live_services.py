"""Checks against the real local services.

Marked ``integration`` and excluded from ``make test``. Run with:

    make up && uv run pytest -m integration

These exist because the unit suite deliberately mocks the transport layer, so
it can verify request shapes but not that a real server accepts them. Something
has to close that gap, and it is better done by a test than by a surprise in
Phase 1.
"""

from __future__ import annotations

import os

import httpx
import pytest
from pydantic import BaseModel

from myscholar.adapters.llm.ollama import OllamaChatModel
from myscholar.ports.llm import Message, Role

pytestmark = pytest.mark.integration

OLLAMA_URL = os.environ.get("MYSCHOLAR_LLM__BASE_URL", "http://127.0.0.1:11434")
UTILITY_MODEL = os.environ.get("MYSCHOLAR_LLM__UTILITY_MODEL", "qwen3:4b-instruct")


def _ollama_running() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0).is_success
    except httpx.HTTPError:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_running(), reason=f"no Ollama server at {OLLAMA_URL} (run: make ollama-up)"
)


class Relevance(BaseModel):
    """Shaped like the grader's real output, so this exercises the actual path."""

    relevant: bool
    reason: str


@requires_ollama
async def test_completion_against_a_live_model() -> None:
    model = OllamaChatModel(UTILITY_MODEL, base_url=OLLAMA_URL)

    reply = await model.complete(
        [
            Message(role=Role.SYSTEM, content="Reply with exactly one word."),
            Message(role=Role.USER, content="What is the capital of France?"),
        ],
        max_tokens=16,
    )

    assert "paris" in reply.lower()


@requires_ollama
async def test_constrained_decoding_produces_valid_json() -> None:
    """The property every classifier and grader in the pipeline depends on.

    If schema-constrained decoding does not hold, each of those call sites needs
    its own parsing and retry logic. Worth proving once, here.
    """
    model = OllamaChatModel(UTILITY_MODEL, base_url=OLLAMA_URL)

    verdict = await model.structured(
        [
            Message(
                role=Role.USER,
                content=(
                    "Passage: 'The FCS-MPCC weighting factors are tuned with NSGA-II.'\n"
                    "Question: 'How are weighting factors selected?'\n"
                    "Is the passage relevant?"
                ),
            )
        ],
        Relevance,
    )

    assert isinstance(verdict.relevant, bool)
    assert verdict.reason


@requires_ollama
async def test_streaming_yields_incrementally() -> None:
    model = OllamaChatModel(UTILITY_MODEL, base_url=OLLAMA_URL)

    tokens = [
        t
        async for t in model.stream(
            [Message(role=Role.USER, content="Count from one to five.")], max_tokens=48
        )
    ]

    assert len(tokens) > 1, "streaming should emit more than one chunk"
    assert "".join(tokens).strip()


@requires_ollama
async def test_utility_model_answers_within_a_small_token_budget() -> None:
    """The utility model must not spend its budget on reasoning.

    A regression guard for a real failure: ``qwen3:4b`` (without ``-instruct``)
    is reasoning-only, so with ``max_tokens=16`` it emitted 16 tokens of
    reasoning and returned empty content. Every grading and classification call
    in the pipeline is exactly this shape, so a reasoning model here would make
    the whole adaptive loop fail silently.
    """
    model = OllamaChatModel(UTILITY_MODEL, base_url=OLLAMA_URL)

    reply = await model.complete(
        [
            Message(role=Role.SYSTEM, content="Reply with exactly one word."),
            Message(role=Role.USER, content="What is the capital of France?"),
        ],
        max_tokens=16,
    )

    assert reply.strip(), "utility model returned nothing within a 16-token budget"
    assert "paris" in reply.lower()
