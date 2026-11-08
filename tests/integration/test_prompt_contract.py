"""Canary questions for the generation prompt.

On the ablation, changing only the prompt (from a bare ``"{context} {query}"`` template to
the seven-rule instruction) cut wrong claims from 2.84 to 0.38 per question; every later
component combined is worth 0.35. The unit suite does not run the prompt.

No assertion checks wording. Each is an invariant: every factual sentence carries a
resolvable marker, markers are never sentence subjects, and a question the corpus answers
is answered.

Marked ``integration`` because it runs the prompt. Excluded from ``make test``; run with
``make test-all`` or

    uv run pytest -m integration -k prompt
"""

from __future__ import annotations

import asyncio
import getpass
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest

from myscholar.adapters.persistence.repository import PostgresStore, create_engine
from myscholar.assembly import build_engine
from myscholar.container import build_container
from myscholar.domain.answers import Answer, Refusal
from myscholar.generation import citations
from myscholar.reasoning.engine import Engine
from tests.conftest import make_settings

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get(
    "MYSCHOLAR_TEST_DATABASE_URL",
    f"postgresql+asyncpg://{getpass.getuser()}@127.0.0.1:5433/myscholar",
)
EMBEDDING_DEVICE = os.environ.get("MYSCHOLAR_TEST_EMBEDDING_DEVICE", "cuda")


# The model backend the canaries run against. The default is the local Ollama pair the published
# numbers were measured on. To run against the deployed model instead:
#
#     MYSCHOLAR_TEST_LLM_PROVIDER=openai_compatible \
#     MYSCHOLAR_TEST_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
#     MYSCHOLAR_TEST_LLM_MODEL=gemini-3.5-flash-lite \
#     MYSCHOLAR_TEST_LLM_KEY=... \
#     uv run pytest -m integration -k prompt
#
# Read at import time: the autouse ``_isolate_environment`` fixture hides MYSCHOLAR_*
# variables, so a lookup inside a fixture would always return the default.
LLM_PROVIDER = os.environ.get("MYSCHOLAR_TEST_LLM_PROVIDER", "ollama")
LLM_BASE_URL = os.environ.get("MYSCHOLAR_TEST_LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("MYSCHOLAR_TEST_LLM_MODEL", "")
LLM_UTILITY_MODEL = os.environ.get("MYSCHOLAR_TEST_LLM_UTILITY_MODEL", "")
LLM_KEY = os.environ.get("MYSCHOLAR_TEST_LLM_KEY", "")


@pytest.fixture(autouse=True)
async def _pace_hosted_calls() -> AsyncIterator[None]:
    """Space out tests when the backend is a hosted free tier.

    Google's free tier answers ``503 high demand`` to bursts: six spaced calls succeeded
    6/6 where the same six back to back managed 1/6.
    """
    yield
    if LLM_PROVIDER != "ollama":
        await asyncio.sleep(_HOSTED_PACING_SECONDS)


# Longer than the adapter's 3s retry wait, which must fit a 45s per-question deadline.
# At 6s, two of fifteen canaries still failed on 503.
_HOSTED_PACING_SECONDS = 12.0


def _llm_settings() -> dict[str, object]:
    settings: dict[str, object] = {"provider": LLM_PROVIDER}
    if LLM_PROVIDER != "ollama":
        if not LLM_KEY:
            pytest.skip(f"provider {LLM_PROVIDER} needs MYSCHOLAR_TEST_LLM_KEY")
        settings["api_key"] = LLM_KEY
        # The hosted endpoint rejects unknown parameters, so no seed is sent.
        settings["seed"] = None
        if LLM_BASE_URL:
            settings["base_url"] = LLM_BASE_URL
    if LLM_MODEL:
        settings["generator_model"] = LLM_MODEL
        settings["utility_model"] = LLM_UTILITY_MODEL or LLM_MODEL
    return settings


# Questions the prompt was changed to fix.
ANSWERABLE = [
    "He found that sensorless control performs worse than sensored control. What caused that?",
    "Tell me about his transformer project.",
    "What has he done on reinforcement learning?",
    "talk about haseeb's strengths in machine learning",
]


@pytest.fixture
async def engine() -> Engine:
    # Built from the constants above instead of ``.env``, so it skips cleanly when the
    # local services are not running.
    settings = make_settings(
        database={"url": DATABASE_URL},
        llm=_llm_settings(),
        embedding={"provider": "torch", "device": EMBEDDING_DEVICE},
    )
    store = PostgresStore(create_engine(settings.database.url))
    if not await store.health():
        pytest.skip(f"no database at {DATABASE_URL} (run: make db-up)")
    if (await store.counts())["chunks"] == 0:
        pytest.skip("index is empty (run: myscholar ingest run)")
    container = build_container(settings)
    return build_engine(settings, container, store)


async def _answer(engine: Engine, question: str) -> tuple[Answer | Refusal, tuple[Any, ...]]:
    response, _trace, blocks = await engine.answer_with_context(question)
    return response, blocks


class TestTheCorpusIsAnsweredWhenItHasTheAnswer:
    """Measured over-refusal is 0%, down from 11.8% before query routing."""

    @pytest.mark.parametrize("question", ANSWERABLE)
    async def test_a_question_the_papers_answer_is_answered(
        self, engine: Engine, question: str
    ) -> None:
        response, blocks = await _answer(engine, question)

        assert blocks, "retrieval returned nothing for a question the corpus covers"
        assert not isinstance(response, Refusal), (
            f"refused a question the corpus answers: {response.text[:120]}"
        )
        assert len(response.text.split()) > 12, (
            f"answer collapsed to almost nothing: {response.text!r}"
        )


class TestEveryFactualSentenceCarriesAResolvableMarker:
    """A failure here means the citation verifier let an uncited claim through."""

    @pytest.mark.parametrize("question", ANSWERABLE)
    async def test_no_uncited_factual_sentence_survives(
        self, engine: Engine, question: str
    ) -> None:
        response, _ = await _answer(engine, question)
        if isinstance(response, Refusal):
            pytest.skip("covered by the answerability test")

        sentences = citations.split_sentences(response.text)
        uncited = [
            s
            for i, s in enumerate(sentences)
            if citations.is_factual(s, leading=i == 0) and not citations.parse_markers(s)
        ]

        assert not uncited, f"factual sentences with no citation: {uncited}"

    @pytest.mark.parametrize("question", ANSWERABLE)
    async def test_every_marker_resolves_to_a_retrieved_block(
        self, engine: Engine, question: str
    ) -> None:
        response, blocks = await _answer(engine, question)
        if isinstance(response, Refusal):
            pytest.skip("covered by the answerability test")

        labels = {b.label.upper() for b in blocks}
        dangling = [m for m in citations.parse_markers(response.text) if m not in labels]

        assert not dangling, f"invented citation labels: {dangling}"


class TestMarkersAreReferencesNotSubjects:
    """Prompt rule 2: a marker such as [C2] cites a claim and is never a sentence's subject."""

    @pytest.mark.parametrize("question", ANSWERABLE)
    async def test_no_sentence_begins_with_a_citation_marker(
        self, engine: Engine, question: str
    ) -> None:
        response, _ = await _answer(engine, question)
        if isinstance(response, Refusal):
            pytest.skip("covered by the answerability test")

        offenders = [
            s
            for s in citations.split_sentences(response.text)
            if s.lstrip().startswith("[") or s.lstrip().lower().startswith("in [c")
        ]

        assert not offenders, f"markers used as sentence subjects: {offenders}"


class TestTheQuestionAskedIsTheQuestionAnswered:
    """Prompt rule 5: a yes/no question opens with yes or no, and a *what* question does not."""

    async def test_a_yes_no_question_opens_with_yes(self, engine: Engine) -> None:
        response, _ = await _answer(engine, "has he done any work related to covid-19?")

        assert not isinstance(response, Refusal), response.text[:120]
        assert response.text.strip().lower().startswith(("yes", "no")), (
            f"a yes/no question opened mid-thought: {response.text[:140]!r}"
        )

    async def test_a_what_question_does_not(self, engine: Engine) -> None:
        response, _ = await _answer(
            engine, "What accuracy did the acoustic fault detection framework reach?"
        )

        assert not isinstance(response, Refusal), response.text[:120]
        assert not response.text.strip().lower().startswith(("yes", "no")), (
            f"a 'what' question was answered yes/no: {response.text[:140]!r}"
        )


class TestAnswersDrawOnThePapersNotOnlyTheProjectPages:
    """Project write-ups are dense in machine-learning vocabulary and can crowd out the papers."""

    async def test_a_research_question_cites_at_least_one_paper(self, engine: Engine) -> None:
        question = "talk about haseeb's strengths in machine learning"
        response, blocks = await _answer(engine, question)

        assert not isinstance(response, Refusal), response.text[:120]
        cited = set(citations.parse_markers(response.text))
        kinds = {b.document_kind for b in blocks if b.label.upper() in cited}

        assert "paper" in kinds, (
            f"answered a research question without citing a paper; cited kinds: {kinds or 'none'}"
        )
