"""The generation prompt is 87% of this system, and nothing was testing it.

Measured on the nine-rung ablation, the step that changes only the prompt —
from a bare `"{context} {query}"` template to the seven-rule instruction —
takes wrong claims from 2.84 per question to 0.38. Every later component
combined is worth 0.35, and two of them are negative. The prompt is not one
lever among several; it is the system.

Six hundred and sixty-six unit tests cover the graph, the budgets, the citation
parser, the salvage logic and the rate limiter. **None of them covered the
prompt.** A one-word edit to those seven rules could return the system to 57%
unsupported claims with the whole suite still green, and four such edits were
made in two days.

So these are canaries. Each question below is one the prompt was changed to fix,
and each assertion is an invariant rather than a wording:

    sensorless      answered on every attempt, after a stall made it refuse
                    half the time
    transformer     over-refused before the corpus and prompt work
    reinforcement   the same
    strengths       answered from project pages while ignoring the papers
    covid-19        a yes/no question that opened mid-thought, because the
                    checker read "COVID-19" as an uncited measurement
    accuracy        a *what* question that started answering "Yes." once the
                    yes/no rule was added — the over-correction

Nothing here asserts a phrase. Wording is the model's to choose and changes
between runs of a different model; what must not change is that every factual
sentence carries a resolvable marker, that markers are references rather than
subjects, and that a question the corpus answers is answered.

Marked ``integration`` because catching a prompt regression requires running the
prompt. Excluded from ``make test``; run with ``make test-all`` or

    uv run pytest -m integration -k prompt
"""

from __future__ import annotations

import asyncio
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
    "postgresql+asyncpg://haseeb@127.0.0.1:5433/myscholar",
)
EMBEDDING_DEVICE = os.environ.get("MYSCHOLAR_TEST_EMBEDDING_DEVICE", "cuda")


# The model backend the canaries run against. Defaults to the local pair, which
# is what the published numbers were measured on and needs no key. Point it at
# the deployed configuration to check the prompt against the model that will
# actually serve it:
#
#     MYSCHOLAR_TEST_LLM_PROVIDER=openai_compatible \
#     MYSCHOLAR_TEST_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
#     MYSCHOLAR_TEST_LLM_MODEL=gemini-3.5-flash-lite \
#     MYSCHOLAR_TEST_LLM_KEY=... \
#     uv run pytest -m integration -k prompt
#
# Worth having as a switch rather than a second test file: the prompt is 87% of
# this system's accuracy, and "it holds on the model we measured" says nothing
# about the model in production.
# Read at import time, not inside the fixture. The autouse `_isolate_environment`
# fixture hides every MYSCHOLAR_* variable from `os.environ` so a developer's
# exported settings cannot change whether the suite passes — and these share that
# prefix, so a fixture-time lookup silently returns the default and the run
# quietly tests the wrong backend. That is why DATABASE_URL above is a constant.
LLM_PROVIDER = os.environ.get("MYSCHOLAR_TEST_LLM_PROVIDER", "ollama")
LLM_BASE_URL = os.environ.get("MYSCHOLAR_TEST_LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("MYSCHOLAR_TEST_LLM_MODEL", "")
LLM_UTILITY_MODEL = os.environ.get("MYSCHOLAR_TEST_LLM_UTILITY_MODEL", "")
LLM_KEY = os.environ.get("MYSCHOLAR_TEST_LLM_KEY", "")


@pytest.fixture(autouse=True)
async def _pace_hosted_calls() -> AsyncIterator[None]:
    """Space out tests when the backend is a shared free tier.

    Nothing here is about correctness — it is about not measuring the provider's
    capacity instead of the prompt. Google's free tier answers `503 high demand`
    to bursts: six spaced calls succeeded 6/6 where the same six back to back
    managed 1/6. Run without pacing, three of these canaries fail on 503 while
    the very same questions answer correctly in isolation, which is a false
    negative on the component carrying 87% of the system's accuracy.

    The local backend needs none of this, so it pays nothing.
    """
    yield
    if LLM_PROVIDER != "ollama":
        await asyncio.sleep(_HOSTED_PACING_SECONDS)


# Twelve seconds, not the adapter's three. The adapter retries inside a 45s
# per-question deadline shared with every other call, so it cannot afford to wait
# longer; a test suite can. At 6s this file still lost two of fifteen canaries to
# 503, and both questions answered correctly when run on their own.
_HOSTED_PACING_SECONDS = 12.0


def _llm_settings() -> dict[str, object]:
    settings: dict[str, object] = {"provider": LLM_PROVIDER}
    if LLM_PROVIDER != "ollama":
        if not LLM_KEY:
            pytest.skip(f"provider {LLM_PROVIDER} needs MYSCHOLAR_TEST_LLM_KEY")
        settings["api_key"] = LLM_KEY
        # This endpoint rejects unknown parameters rather than ignoring them.
        settings["seed"] = None
        if LLM_BASE_URL:
            settings["base_url"] = LLM_BASE_URL
    if LLM_MODEL:
        settings["generator_model"] = LLM_MODEL
        settings["utility_model"] = LLM_UTILITY_MODEL or LLM_MODEL
    return settings


# Questions the prompt was changed to fix. Each is a real regression.
ANSWERABLE = [
    "He found that sensorless control performs worse than sensored control. What caused that?",
    "Tell me about his transformer project.",
    "What has he done on reinforcement learning?",
    "talk about haseeb's strengths in machine learning",
]


@pytest.fixture
async def engine() -> Engine:
    # Built explicitly rather than from `.env`, which the suite deliberately
    # cannot read: a local configuration must never decide whether tests pass.
    # The values here are the documented development defaults, so this skips
    # cleanly on a machine that has not run `make up` rather than testing
    # whatever that machine happens to be configured for.
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
    """Over-refusal was 11.8% before routing and is 0% now. It must stay there."""

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
    """The deterministic guarantee. If this fails, the verifier has a hole."""

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
    """Rule 2. Writing "[C2] describes MOPSO" turns an answer into a reading list.

    It is also what produced "In another work, ..." as an opening sentence: the
    model narrating excerpts rather than answering, and the sentence it referred
    back to then removed by verification.
    """

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
    """Rule 5, and the over-correction that followed it.

    Adding "a yes/no question should open with yes or no" immediately made a
    *what* question answer "Yes. The acoustic fault detection framework achieved
    98.53%...". Both directions are pinned, because fixing one broke the other
    once already.
    """

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
    """The regression that appeared when two project write-ups were ingested.

    Asked about strengths in machine learning, the answer described the
    translator and the RAG project and never reached the papers — the newest
    documents are dense in machine-learning vocabulary and crowded them out.
    """

    async def test_a_research_question_cites_at_least_one_paper(self, engine: Engine) -> None:
        question = "talk about haseeb's strengths in machine learning"
        response, blocks = await _answer(engine, question)

        assert not isinstance(response, Refusal), response.text[:120]
        cited = set(citations.parse_markers(response.text))
        kinds = {b.document_kind for b in blocks if b.label.upper() in cited}

        assert "paper" in kinds, (
            f"answered a research question without citing a paper; cited kinds: {kinds or 'none'}"
        )
