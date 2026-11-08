"""Routing of questions about someone's work on a subject.

Such a question spans several papers, so it must reach a class that runs document
selection; ``single_paper_factual`` answers from one paper. The probes in
``tests/eval/work_on_probes.yaml`` cross four phrasings with five topics, each spanning
at least two papers. Cases go through ``assembly.build_router``, wired as the server is.

Marked ``integration`` because it runs the classifier. Run with

    uv run pytest -m integration -k routing
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from myscholar.adapters.persistence.repository import PostgresStore, create_engine
from myscholar.assembly import build_router
from myscholar.container import build_container
from myscholar.query.classifier import QueryClassifier
from myscholar.query.router import QueryRouter
from tests.conftest import make_settings

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get(
    "MYSCHOLAR_TEST_DATABASE_URL",
    f"postgresql+asyncpg://{getpass.getuser()}@127.0.0.1:5433/myscholar",
)
EMBEDDING_DEVICE = os.environ.get("MYSCHOLAR_TEST_EMBEDDING_DEVICE", "cuda")


# The model backend for these checks. The default is the local Ollama pair the published
# numbers were measured on. To run against the deployed model instead:
#
#     MYSCHOLAR_TEST_LLM_PROVIDER=openai_compatible \
#     MYSCHOLAR_TEST_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
#     MYSCHOLAR_TEST_LLM_MODEL=gemini-3.5-flash-lite \
#     MYSCHOLAR_TEST_LLM_KEY=... \
#     uv run pytest -m integration -k routing
#
# Read at import time: the autouse ``_isolate_environment`` fixture hides MYSCHOLAR_*
# variables, so a lookup inside a fixture would always return the default.
LLM_PROVIDER = os.environ.get("MYSCHOLAR_TEST_LLM_PROVIDER", "ollama")
LLM_BASE_URL = os.environ.get("MYSCHOLAR_TEST_LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("MYSCHOLAR_TEST_LLM_MODEL", "")
LLM_UTILITY_MODEL = os.environ.get("MYSCHOLAR_TEST_LLM_UTILITY_MODEL", "")
LLM_KEY = os.environ.get("MYSCHOLAR_TEST_LLM_KEY", "")


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


PROBES = Path("tests/eval/work_on_probes.yaml")

# A measured ceiling: three of twenty probes reach a class that skips document selection.
MAX_PROBES_WITHOUT_SELECTION = 3


def _cases(kind: str) -> list[dict[str, Any]]:
    probes = yaml.safe_load(PROBES.read_text("utf-8"))
    return [p for p in probes if p.get("kind") == kind]


@pytest.fixture
async def router() -> QueryRouter:
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
    documents = await store.list_documents()
    return build_router(settings, container, store, titles=[d.title for d in documents])


async def _classify_and_plan(router: QueryRouter, question: str) -> tuple[str, int]:
    """Classify and plan through the production router, without the retrieval ``route()`` runs."""
    classification = await router._classifier.classify(question)  # noqa: SLF001
    plan = router._planner.plan(classification)  # noqa: SLF001
    return classification.query_class.value, plan.select_documents


class TestTheClassifierIsBuiltWithTheConfiguredPerson:
    """The classification prompt interpolates the configured person's name."""

    def test_the_router_names_the_configured_person(self) -> None:
        settings = make_settings()
        classifier = QueryClassifier(None, person_name=settings.person.name)

        assert classifier._person == settings.person.name  # noqa: SLF001
        assert classifier._person != "the researcher", (  # noqa: SLF001
            "the classifier default leaked into a configured build; the prompt "
            "interpolates this string and classifications change with it"
        )

    def test_the_classifier_default_is_never_what_ships(self) -> None:
        # The placeholder name would make every prompt read like an unfinished template.
        assert make_settings().person.name != "the researcher"


class TestQuestionsAboutWorkOnASubjectSearchMoreThanOnePaper:
    """Asserted on the plan rather than the class label.

    ``methodological`` also selects documents, so only classes that skip selection lose papers.
    """

    async def test_no_more_probes_lose_selection_than_already_do(self, router: QueryRouter) -> None:
        lost = []
        for case in _cases("probe"):
            question = str(case["question"])
            query_class, select = await _classify_and_plan(router, question)
            if select == 0:
                lost.append(f"{question!r} -> {query_class}")

        assert len(lost) <= MAX_PROBES_WITHOUT_SELECTION, (
            f"{len(lost)} of 20 probes skip document selection, up from "
            f"{MAX_PROBES_WITHOUT_SELECTION}. A question about someone's work on "
            f"a subject that does not select documents answers from one paper:\n  "
            + "\n  ".join(lost)
        )

    async def test_an_explicit_multiplicity_cue_always_routes_to_synthesis(
        self, router: QueryRouter
    ) -> None:
        # "how has his work on X developed over time?" states multiplicity outright,
        # like the gold synthesis cases in routing_set.yaml.
        developmental = [c for c in _cases("probe") if c.get("phrasing") == "D"]
        assert len(developmental) == 5, "probe set changed shape"

        wrong = []
        for case in developmental:
            query_class, _ = await _classify_and_plan(router, str(case["question"]))
            if query_class != "cross_paper_synthesis":
                wrong.append(f"{case['question']!r} -> {query_class}")

        assert not wrong, "a developmental question stopped being synthesis:\n  " + "\n  ".join(
            wrong
        )


class TestWideningSynthesisDoesNotBreakSinglePaperQuestions:
    """Controls copied from ``routing_set.yaml`` keep their gold labels.

    Six ask one fact about one named paper; two count papers in the collection.
    """

    async def test_every_control_holds_its_gold_label(self, router: QueryRouter) -> None:
        moved = []
        for case in _cases("control"):
            question = str(case["question"])
            query_class, _ = await _classify_and_plan(router, question)
            if query_class != case["expect"]:
                moved.append(f"{question!r}: {case['expect']} -> {query_class}")

        assert not moved, (
            "a gold-labelled question moved class; a change to the classifier "
            "prompt has overreached:\n  " + "\n  ".join(moved)
        )
