"""Pins how a question about someone's work on a subject is routed.

A question like "his work on chaos theory" spans several papers, so it must be
classified into a class that runs document selection. Classified
`single_paper_factual` instead, `select_documents` is 0 and no per-document cap
applies, and the answer is drawn from whichever paper retrieval liked best — the
second paper on the subject is never fetched. That is a silent failure: the
answer is fluent, correctly cited, and incomplete.

Three of the twenty probes in `tests/eval/work_on_probes.yaml` do this today, and
the assertions below are a ceiling on that rather than a target. Classification
is one model call whose output every later stage branches on, so the cost of a
change here is paid across the whole pipeline; this file makes that cost visible.

The probe set crosses four phrasings with five topics, every topic verified to
span at least two papers. Phrasing D ("how has his work on X developed over
time?") states multiplicity outright and is asserted exactly, at 5 of 5; the
other three phrasings leave it implicit and are asserted in aggregate.

Cases are classified through `assembly.build_router`, so this exercises the same
wiring the server does. A test that constructs its own classifier can pass while
production behaves differently — the prompt interpolates the configured person
name, among other things, and a hand-built classifier silently gets the default.

Marked ``integration`` because catching a routing regression requires running
the classifier. Run with

Marked ``integration`` because catching a routing regression requires running
the classifier. Run with

    uv run pytest -m integration -k routing
"""

from __future__ import annotations

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
    "postgresql+asyncpg://haseeb@127.0.0.1:5433/myscholar",
)
EMBEDDING_DEVICE = os.environ.get("MYSCHOLAR_TEST_EMBEDDING_DEVICE", "cuda")

PROBES = Path("tests/eval/work_on_probes.yaml")

#: A measured ceiling, not a goal. Three of twenty probes land in a class that
#: does not select documents. Raising this number needs a measurement that says
#: why; lowering it is the point of the file.
MAX_PROBES_WITHOUT_SELECTION = 3


def _cases(kind: str) -> list[dict[str, Any]]:
    probes = yaml.safe_load(PROBES.read_text("utf-8"))
    return [p for p in probes if p.get("kind") == kind]


@pytest.fixture
async def router() -> QueryRouter:
    # Built from documented development defaults rather than `.env`, so this
    # skips cleanly on a machine that has not run `make up` instead of testing
    # whatever that machine happens to be configured for.
    settings = make_settings(
        database={"url": DATABASE_URL},
        llm={"provider": "ollama"},
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
    """Classify and plan through the production objects.

    Reaches past the router's privates deliberately: going through
    `build_router` means a change to how production wires the classifier cannot
    pass here while breaking the real thing. Calling `route()` instead would
    retrieve for all twenty-eight cases, which none of these assertions needs.
    """
    classification = await router._classifier.classify(question)  # noqa: SLF001
    plan = router._planner.plan(classification)  # noqa: SLF001
    return classification.query_class.value, plan.select_documents


class TestTheClassifierIsBuiltWithTheConfiguredPerson:
    """The prompt names the person, so classification depends on that name.

    Cheap to assert and worth asserting: anything that builds a classifier
    without passing `person_name` gets the placeholder default, and then
    measures — or serves — a prompt that differs from the configured one.
    """

    def test_the_router_names_the_configured_person(self) -> None:
        settings = make_settings()
        classifier = QueryClassifier(None, person_name=settings.person.name)

        assert classifier._person == settings.person.name  # noqa: SLF001
        assert classifier._person != "the researcher", (  # noqa: SLF001
            "the classifier default leaked into a configured build; the prompt "
            "interpolates this string and classifications change with it"
        )

    def test_the_classifier_default_is_never_what_ships(self) -> None:
        # With person.name left at the placeholder every prompt reads like an
        # unfinished template. Asserted so the coupling is documented.
        assert make_settings().person.name != "the researcher"


class TestQuestionsAboutWorkOnASubjectSearchMoreThanOnePaper:
    """Asserted on the plan rather than on the class label.

    The label is not what breaks the answer. `methodological` still selects
    eight documents and caps them at three, so it produces an answer spanning
    papers even though it is the wrong name for the question. Only the classes
    that skip selection lose the second paper, so that is what is measured.
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
        # "how has his work on X developed over time?" states multiplicity
        # outright, and the gold synthesis cases in routing_set.yaml share that
        # cue. If this breaks, the synthesis class has stopped being reachable
        # by its clearest signal and the aggregate above will follow.
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
    """The cost side of the trade, on the project's own gold labels.

    Every control is copied verbatim from `routing_set.yaml`. Six name one paper
    and ask one fact about it; two count papers in the collection, the class
    where a misroute fabricates a number rather than merely narrowing an answer.
    Any change that widens synthesis is paid for here.
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
