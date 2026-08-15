"""Compare three pipelines end to end.

    uv run python scripts/eval_generation.py

The comparison the whole project exists to make, and the one almost no
portfolio RAG can produce:

  naive      dense-only retrieval, one model call,
             no citations, no verification, no way to decline
  retrieved  the same one-shot generation, fed hybrid + reranked + expanded
             context. Isolates what retrieval alone is worth.
  adaptive   the full system — routing, grading, citation verification,
             grounding, corrective retry, refusal

Same corpus, same embedder, same generator across all three. The only variable
is pipeline design, which is what makes any difference attributable.

Three families of number, and none of them means much alone:

  unsupported claims  factual sentences the retrieved context does not support
  correct refusal     declines the questions the corpus genuinely cannot answer
  over-refusal        declines questions it could have answered

Refusing everything scores perfectly on the second, which is why the third is
measured on a control group and reported beside it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logging.disable(logging.INFO)

from myscholar.adapters.persistence.repository import (  # noqa: E402
    PostgresStore,
    create_engine,
)
from myscholar.config import load_settings  # noqa: E402
from myscholar.container import build_container  # noqa: E402
from myscholar.domain.answers import Refusal  # noqa: E402
from myscholar.domain.documents import ContextBlock  # noqa: E402
from myscholar.evaluation.baseline import NaivePipeline, declines  # noqa: E402
from myscholar.evaluation.faithfulness import FaithfulnessJudge  # noqa: E402
from myscholar.generation import citations  # noqa: E402
from myscholar.generation.answerer import Answerer  # noqa: E402
from myscholar.query.classifier import QueryClassifier  # noqa: E402
from myscholar.query.planner import QueryPlanner  # noqa: E402
from myscholar.query.router import QueryRouter  # noqa: E402
from myscholar.query.transforms import QueryTransformer  # noqa: E402
from myscholar.reasoning.engine import Engine  # noqa: E402
from myscholar.reasoning.graph import Budget  # noqa: E402
from myscholar.retrieval.retriever import HybridRetriever, RetrievalMode  # noqa: E402

TEST_SET = Path("tests/eval/adversarial_set.yaml")
RESULTS = Path("results")

#: Kinds where declining is the correct answer.
MUST_REFUSE = {"unanswerable", "false_premise", "injection"}


@dataclass
class Outcome:
    question: str
    kind: str
    expected: str
    declined: bool
    text: str
    seconds: float
    claims_judged: int = 0
    claims_unsupported: int = 0
    unsupported_examples: tuple[str, ...] = ()
    dangling: int = 0
    markers: int = 0
    path: str = ""
    """Nodes visited, for the adaptive configuration. The distribution shows
    how often the loop corrects itself, which is the behaviour it exists for
    and is invisible in any of the rates."""

    @property
    def should_refuse(self) -> bool:
        return self.kind in MUST_REFUSE

    @property
    def correct(self) -> bool:
        return self.declined == self.should_refuse


@dataclass
class ConfigResult:
    name: str
    outcomes: list[Outcome] = field(default_factory=list)

    def _subset(self, refusing: bool) -> list[Outcome]:
        return [o for o in self.outcomes if o.should_refuse is refusing]

    @property
    def correct_refusal(self) -> float | None:
        cases = self._subset(refusing=True)
        return None if not cases else sum(o.declined for o in cases) / len(cases)

    @property
    def over_refusal(self) -> float | None:
        cases = self._subset(refusing=False)
        return None if not cases else sum(o.declined for o in cases) / len(cases)

    @property
    def unsupported_rate(self) -> float | None:
        """Across every answer that was actually given.

        Refusals contribute nothing: they contain no claims, so counting them
        would let a system improve this number by declining more.
        """
        judged = sum(o.claims_judged for o in self.outcomes)
        return None if not judged else sum(o.claims_unsupported for o in self.outcomes) / judged

    @property
    def citation_validity(self) -> float | None:
        markers = sum(o.markers for o in self.outcomes)
        return None if not markers else 1 - sum(o.dangling for o in self.outcomes) / markers

    @property
    def mean_seconds(self) -> float:
        return sum(o.seconds for o in self.outcomes) / max(len(self.outcomes), 1)


async def _build(settings: Any) -> tuple[Any, HybridRetriever, PostgresStore, Engine]:
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))
    documents = await store.list_documents()

    retriever = HybridRetriever(
        store=store,
        embedder=container.embedder,
        reranker=container.reranker,
        top_k=settings.retrieval.top_k,
        rerank_k=settings.retrieval.rerank_k,
        window=settings.retrieval.window,
        dense_weight=settings.retrieval.dense_weight,
        lexical_weight=settings.retrieval.lexical_weight,
        rrf_k=settings.retrieval.rrf_k,
        max_span_chunks=settings.retrieval.max_span_chunks,
    )
    engine = Engine(
        router=QueryRouter(
            classifier=QueryClassifier(
                container.utility,
                titles=[d.title for d in documents],
                person_name=settings.person.name,
            ),
            planner=QueryPlanner(
                top_k=settings.retrieval.top_k,
                rerank_k=settings.retrieval.rerank_k,
                person_name=settings.person.name,
            ),
            transformer=QueryTransformer(container.utility),
            retriever=retriever,
            store=store,
        ),
        answerer=Answerer(
            generator=container.generator,
            utility=container.utility,
            person_name=settings.person.name,
            short_name=settings.person.short_name,
        ),
        budget=Budget(
            max_steps=settings.reasoning.max_steps,
            max_retrieval_rounds=settings.reasoning.max_retrieval_rounds,
            max_regenerations=settings.reasoning.max_regenerations,
            deadline_seconds=settings.reasoning.deadline_seconds,
        ),
        short_name=settings.person.short_name,
    )
    return container, retriever, store, engine


async def _score(
    judge: FaithfulnessJudge,
    outcome: Outcome,
    text: str,
    blocks: tuple[ContextBlock, ...],
) -> None:
    if outcome.declined or not text:
        return
    result = await judge.score(text, blocks)
    outcome.claims_judged = result.judged
    outcome.claims_unsupported = len(result.unsupported)
    outcome.unsupported_examples = result.unsupported[:2]


async def run_naive(
    cases: list[dict[str, str]],
    *,
    pipeline: NaivePipeline,
    judge: FaithfulnessJudge,
    name: str,
) -> ConfigResult:
    result = ConfigResult(name=name)
    for index, case in enumerate(cases, 1):
        started = time.monotonic()
        answer = await pipeline.answer(case["question"])
        outcome = Outcome(
            question=case["question"],
            kind=case["kind"],
            expected=case["expect"],
            declined=declines(answer.text),
            text=answer.text,
            seconds=time.monotonic() - started,
        )
        await _score(judge, outcome, answer.text, answer.blocks)
        result.outcomes.append(outcome)
        print(f"  [{name}] {index}/{len(cases)} {case['question'][:56]}")
    return result


async def run_adaptive(
    cases: list[dict[str, str]],
    *,
    engine: Engine,
    judge: FaithfulnessJudge,
) -> ConfigResult:
    result = ConfigResult(name="adaptive")
    for index, case in enumerate(cases, 1):
        started = time.monotonic()
        response, trace, blocks = await engine.answer_with_context(case["question"])
        # A structural refusal and a paragraph that says "the excerpts do not
        # mention this" are the same outcome for a reader, so both count.
        declined = isinstance(response, Refusal) or declines(response.text)

        markers = dangling = 0
        if not isinstance(response, Refusal):
            emitted = citations.parse_markers(response.text)
            labels = {c.label for c in response.citations}
            markers = len(emitted)
            dangling = sum(1 for m in emitted if m not in labels)

        outcome = Outcome(
            question=case["question"],
            kind=case["kind"],
            expected=case["expect"],
            declined=declined,
            text=response.text,
            seconds=time.monotonic() - started,
            markers=markers,
            dangling=dangling,
            path=" -> ".join(trace.path),
        )
        await _score(judge, outcome, response.text, blocks)
        result.outcomes.append(outcome)
        print(f"  [adaptive] {index}/{len(cases)} {case['question'][:56]}")
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def report(results: list[ConfigResult], cases: list[dict[str, str]]) -> str:
    kinds = sorted({c["kind"] for c in cases})
    lines = [
        "# Pipeline comparison",
        "",
        f"{len(cases)} questions, three configurations, one corpus.",
        "",
        "Same index, same embedder, same generator throughout. The only "
        "variable is pipeline design, so every difference below is "
        "attributable to it rather than to a bigger model or a cleaner corpus.",
        "",
        "| Configuration | unsupported claims | correct refusal | over-refusal "
        "| citation validity | mean latency |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {_pct(r.unsupported_rate)} | {_pct(r.correct_refusal)} "
            f"| {_pct(r.over_refusal)} | {_pct(r.citation_validity)} | {r.mean_seconds:.1f}s |"
        )

    lines += [
        "",
        "**unsupported claims** — factual sentences the retrieved passages do "
        "not support, judged one at a time. Refusals contain no claims and so "
        "contribute nothing, which stops a system improving the number by "
        "declining more.",
        "",
        "**correct refusal** paired with **over-refusal** — either alone is a "
        "bias rather than a measurement. A system that refuses everything "
        "scores 100% on the first.",
        "",
        "## Correct behaviour by question kind",
        "",
        "| Configuration | " + " | ".join(f"`{k}`" for k in kinds) + " |",
        "|---" * (len(kinds) + 1) + "|",
    ]
    for r in results:
        cells = []
        for kind in kinds:
            subset = [o for o in r.outcomes if o.kind == kind]
            cells.append(
                "—" if not subset else f"{sum(o.correct for o in subset) / len(subset) * 100:.0f}%"
            )
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")

    lines += ["", "## Where each configuration went wrong", ""]
    for r in results:
        wrong = [o for o in r.outcomes if not o.correct]
        lines.append(f"### {r.name} — {len(wrong)} of {len(r.outcomes)}")
        lines.append("")
        if not wrong:
            lines += ["Nothing.", ""]
            continue
        for outcome in wrong[:8]:
            verb = "answered, should have declined" if not outcome.declined else "declined"
            lines.append(f"- **{outcome.kind}** — {verb}: {outcome.question}")
            lines.append(f"  > {outcome.text[:190].strip()}")
        lines.append("")

    paths = Counter(o.path for r in results for o in r.outcomes if o.path)
    if paths:
        lines += [
            "## Paths through the adaptive loop",
            "",
            "Each retry is the system rejecting its own draft. That is the "
            "behaviour the loop exists for and it does not appear in any rate.",
            "",
            "| count | path |",
            "|---|---|",
        ]
        lines += [f"| {n} | `{path}` |" for path, n in paths.most_common(8)]
        lines.append("")

    unsupported = [(r.name, o) for r in results for o in r.outcomes if o.unsupported_examples]
    if unsupported:
        lines += ["## Sample unsupported claims", ""]
        for name, outcome in unsupported[:10]:
            lines.append(f"- **{name}** — {outcome.question}")
            for claim in outcome.unsupported_examples[:1]:
                lines.append(f"  > {claim[:170]}")
        lines.append("")

    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N questions.")
    parser.add_argument(
        "--configs",
        default="naive,retrieved,adaptive",
        help="Comma-separated subset to run.",
    )
    args = parser.parse_args()

    cases = [c for c in yaml.safe_load(TEST_SET.read_text()) if isinstance(c, dict)]
    if args.limit:
        cases = cases[: args.limit]
    wanted = {name.strip() for name in args.configs.split(",")}

    settings = load_settings()
    container, retriever, _store, engine = await _build(settings)
    judge = FaithfulnessJudge(container.utility)

    print(f"{len(cases)} questions x {len(wanted)} configuration(s)\n")
    results: list[ConfigResult] = []

    if "naive" in wanted:
        results.append(
            await run_naive(
                cases,
                pipeline=NaivePipeline(
                    retriever=retriever, model=container.generator, mode=RetrievalMode.DENSE
                ),
                judge=judge,
                name="naive",
            )
        )
    if "retrieved" in wanted:
        results.append(
            await run_naive(
                cases,
                pipeline=NaivePipeline(
                    retriever=retriever, model=container.generator, mode=RetrievalMode.FULL
                ),
                judge=judge,
                name="retrieved",
            )
        )
    if "adaptive" in wanted:
        results.append(await run_adaptive(cases, engine=engine, judge=judge))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "generation.md").write_text(report(results, cases), "utf-8")

    by_config: dict[str, Any] = defaultdict(dict)
    for r in results:
        by_config[r.name] = {
            "unsupported_claims": r.unsupported_rate,
            "correct_refusal": r.correct_refusal,
            "over_refusal": r.over_refusal,
            "citation_validity": r.citation_validity,
            "mean_seconds": r.mean_seconds,
            "outcomes": [
                {
                    "question": o.question,
                    "kind": o.kind,
                    "declined": o.declined,
                    "correct": o.correct,
                    "claims_judged": o.claims_judged,
                    "claims_unsupported": o.claims_unsupported,
                    "text": o.text,
                }
                for o in r.outcomes
            ],
        }
    (RESULTS / "generation.json").write_text(json.dumps(by_config, indent=2), "utf-8")

    print("\n" + "\n".join(report(results, cases).splitlines()[:14]))
    print(f"\nwritten to {RESULTS / 'generation.md'}")


if __name__ == "__main__":
    asyncio.run(main())
