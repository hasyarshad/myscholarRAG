"""Compare pipelines end to end on the adversarial question set.

    uv run python scripts/eval_generation.py

  naive      dense-only retrieval, one model call, no citations, no verification
  retrieved  the same one-shot generation over hybrid, expanded context
  adaptive   the full system: routing, citation verification, grounding,
             corrective retrieval and refusal

The ablation rungs in myscholar.evaluation.ablation can be selected with --configs.
All configurations share the corpus, embedder and generator.

  unsupported claims  factual sentences the retrieved context does not support
  correct refusal     declines the questions the corpus cannot answer
  over-refusal        declines questions it could have answered
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
from myscholar.assembly import (  # noqa: E402
    build_answerer,
    build_engine,
    build_retriever,
    build_router,
)
from myscholar.config import load_settings  # noqa: E402
from myscholar.container import build_container  # noqa: E402
from myscholar.domain.answers import Refusal  # noqa: E402
from myscholar.domain.documents import ContextBlock  # noqa: E402
from myscholar.evaluation.ablation import (  # noqa: E402
    LADDER,
    AblationPipeline,
)
from myscholar.evaluation.baseline import (  # noqa: E402
    NaivePipeline,
    contradicts,
    declines,
)
from myscholar.evaluation.faithfulness import FaithfulnessJudge  # noqa: E402
from myscholar.generation import citations  # noqa: E402
from myscholar.reasoning.engine import Engine  # noqa: E402
from myscholar.retrieval.retriever import HybridRetriever, RetrievalMode  # noqa: E402

TEST_SET = Path("tests/eval/adversarial_set.yaml")
RESULTS = Path("results")

#: Longest a single question may take before it is abandoned. Well above the
#: loop's own 90s deadline, but stops one wedged request stalling the whole run.
QUESTION_TIMEOUT = 300.0

#: Kinds where declining is the correct answer.
MUST_REFUSE = {"unanswerable", "false_premise", "injection"}


@dataclass
class Outcome:
    question: str
    kind: str
    expected: str
    declined: bool
    contradicted: bool
    text: str
    seconds: float
    claims_judged: int = 0
    claims_unsupported: int = 0
    unsupported_examples: tuple[str, ...] = ()
    dangling: int = 0
    markers: int = 0

    #: Distinct source documents in the blocks the generator was given; the
    #: denominator for coverage.
    papers_available: int = 0

    #: Distinct source documents the answer actually cites.
    papers_cited: int = 0

    path: str = ""
    """Nodes visited, for the adaptive configuration and the ablation rungs."""

    @property
    def should_refuse(self) -> bool:
        return self.kind in MUST_REFUSE

    @property
    def correct(self) -> bool:
        """Declining or contradicting the premise is correct on a must-refuse question.

        On an answerable question only declining is wrong. Contradiction is not
        consulted there, so an answer that happens to say "instead" is not a refusal.
        """
        if self.should_refuse:
            return self.declined or self.contradicted
        return not self.declined


@dataclass
class ConfigResult:
    name: str
    outcomes: list[Outcome] = field(default_factory=list)

    def _subset(self, refusing: bool) -> list[Outcome]:
        return [o for o in self.outcomes if o.should_refuse is refusing]

    @property
    def correct_refusal(self) -> float | None:
        """Share of must-refuse questions declined or contradicted, as in the per-kind table."""
        cases = self._subset(refusing=True)
        return None if not cases else sum(o.correct for o in cases) / len(cases)

    @property
    def over_refusal(self) -> float | None:
        cases = self._subset(refusing=False)
        return None if not cases else sum(o.declined for o in cases) / len(cases)

    @property
    def unsupported_rate(self) -> float | None:
        """Across answers actually given. Refusals contain no claims and add nothing."""
        judged = sum(o.claims_judged for o in self.outcomes)
        return None if not judged else sum(o.claims_unsupported for o in self.outcomes) / judged

    @property
    def citation_validity(self) -> float | None:
        markers = sum(o.markers for o in self.outcomes)
        return None if not markers else 1 - sum(o.dangling for o in self.outcomes) / markers

    @property
    def mean_seconds(self) -> float:
        return sum(o.seconds for o in self.outcomes) / max(len(self.outcomes), 1)


async def _build(settings: Any) -> tuple[Any, HybridRetriever, PostgresStore, Engine, Any]:
    """Assembled through myscholar.assembly, exactly as the API assembles it."""
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))
    documents = await store.list_documents()

    retriever = build_retriever(settings, container, store)
    router = build_router(
        settings, container, store, retriever=retriever, titles=[d.title for d in documents]
    )
    engine = build_engine(settings, container, store, router=router)
    return container, retriever, store, engine, router


async def _score_all(
    judge: FaithfulnessJudge,
    pending: list[tuple[Outcome, str, tuple[ContextBlock, ...]]],
) -> None:
    """Judge every answer once generation is finished.

    Not interleaved: alternating the generator and the judge model makes Ollama
    swap models on every question.
    """
    print(f"  judging {len(pending)} answer(s) ...", flush=True)
    for outcome, text, blocks in pending:
        if outcome.declined or not text:
            continue
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
    pending: list[tuple[Outcome, str, tuple[ContextBlock, ...]]] = []
    for index, case in enumerate(cases, 1):
        started = time.monotonic()
        try:
            async with asyncio.timeout(QUESTION_TIMEOUT):
                answer = await pipeline.answer(case["question"])
        except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
            print(f"  [{name}] {index}/{len(cases)} FAILED: {type(exc).__name__}", flush=True)
            continue
        outcome = Outcome(
            question=case["question"],
            kind=case["kind"],
            expected=case["expect"],
            declined=declines(answer.text),
            contradicted=contradicts(answer.text),
            text=answer.text,
            seconds=time.monotonic() - started,
        )
        pending.append((outcome, answer.text, answer.blocks))
        result.outcomes.append(outcome)
        print(
            f"  [{name}] {index}/{len(cases)} {outcome.seconds:5.1f}s  {case['question'][:52]}",
            flush=True,
        )
    await _score_all(judge, pending)
    return result


async def run_adaptive(
    cases: list[dict[str, str]],
    *,
    engine: Any,
    judge: FaithfulnessJudge,
    name: str = "adaptive",
) -> ConfigResult:
    """Run any pipeline exposing ``answer_with_context``: the engine or an ablation rung."""
    result = ConfigResult(name=name)
    pending: list[tuple[Outcome, str, tuple[ContextBlock, ...]]] = []
    for index, case in enumerate(cases, 1):
        started = time.monotonic()
        try:
            async with asyncio.timeout(QUESTION_TIMEOUT):
                response, trace, blocks = await engine.answer_with_context(case["question"])
        except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
            print(f"  [{name}] {index}/{len(cases)} FAILED: {type(exc).__name__}", flush=True)
            continue
        # A Refusal and a paragraph saying "the excerpts do not mention this" both count.
        declined = isinstance(response, Refusal) or declines(response.text)

        markers = dangling = papers_cited = 0
        if not isinstance(response, Refusal):
            emitted = citations.parse_markers(response.text)
            labels = {c.label for c in response.citations}
            markers = len(emitted)
            dangling = sum(1 for m in emitted if m not in labels)
            papers_cited = len({c.document_id for c in response.citations})
        papers_available = len({b.document_id for b in blocks})

        outcome = Outcome(
            question=case["question"],
            kind=case["kind"],
            expected=case["expect"],
            declined=declined,
            contradicted=contradicts(response.text),
            text=response.text,
            seconds=time.monotonic() - started,
            markers=markers,
            dangling=dangling,
            papers_available=papers_available,
            papers_cited=papers_cited,
            path=" -> ".join(trace.path if hasattr(trace, "path") else trace),
        )
        pending.append((outcome, response.text, blocks))
        result.outcomes.append(outcome)
        print(
            f"  [{name}] {index}/{len(cases)} {outcome.seconds:5.1f}s  {case['question'][:52]}",
            flush=True,
        )
    await _score_all(judge, pending)
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def report(results: list[ConfigResult], cases: list[dict[str, str]]) -> str:
    kinds = sorted({c["kind"] for c in cases})
    lines = [
        "# Pipeline comparison",
        "",
        f"{len(cases)} questions, {len(results)} configuration"
        f"{'' if len(results) == 1 else 's'}, one corpus.",
        "",
        "Every configuration uses the same index, embedder and generator, so the "
        "differences below come from pipeline design.",
        "",
        "| Configuration | unsupported claims | claims/question | correct refusal "
        "| over-refusal | citation validity | mean latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        # Claims per question, because an unsupported-claim rate rewards saying less.
        judged = sum(o.claims_judged for o in r.outcomes)
        per_question = judged / max(len(r.outcomes), 1)
        lines.append(
            f"| {r.name} | {_pct(r.unsupported_rate)} | {per_question:.1f} "
            f"| {_pct(r.correct_refusal)} | {_pct(r.over_refusal)} "
            f"| {_pct(r.citation_validity)} | {r.mean_seconds:.1f}s |"
        )

    lines += [
        "",
        "**unsupported claims**: factual sentences the retrieved passages do not "
        "support, judged one at a time. Refusals contain no claims, so declining "
        "more cannot improve the rate.",
        "",
        "**correct refusal** and **over-refusal** must be read together: a system "
        "that refuses everything scores 100% on correct refusal.",
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

    # Coverage is descriptive, with no target: a narrow question deserves a narrow
    # answer. It catches "cited one paper when six were relevant", which every
    # other number here scores as correct.
    lines += [
        "",
        "## Coverage",
        "",
        "Distinct source documents cited, against the number the generator held.",
        "Reported per question kind, because the right breadth depends on the",
        "question: one about a single measurement should cite one paper, and one",
        "about the span of someone's work should cite several.",
        "",
        "| Configuration | kind | n | mean available | mean cited | coverage |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        answered = [o for o in result.outcomes if not o.declined and o.papers_available]
        by_kind: dict[str, list[Outcome]] = {}
        for o in answered:
            by_kind.setdefault(o.kind, []).append(o)
        for kind, group in sorted(by_kind.items()):
            available = sum(o.papers_available for o in group) / len(group)
            cited = sum(o.papers_cited for o in group) / len(group)
            lines.append(
                f"| {result.name} | {kind} | {len(group)} | {available:.1f} | "
                f"{cited:.1f} | {cited / available:.0%} |"
            )

    lines += ["", "Answers citing one source while holding four or more:", ""]
    for result in results:
        narrow = [
            o
            for o in result.outcomes
            if not o.declined and o.papers_cited <= 1 and o.papers_available >= 4
        ]
        if not narrow:
            lines.append(f"- {result.name}: none")
        for o in narrow:
            lines.append(
                f"- ({o.kind}) {o.question} — cited {o.papers_cited} of {o.papers_available}"
            )

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
            "How many questions took each path through the pipeline.",
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
        "--suffix", default="", help="Appended to the output filenames, to keep runs apart."
    )
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
    container, retriever, _store, engine, router = await _build(settings)
    judge = FaithfulnessJudge(container.utility, person=settings.person.name)

    print(f"{len(cases)} questions x {len(wanted)} configuration(s)\n", flush=True)
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
                # The naive generator given the strongest retrieval available.
                pipeline=NaivePipeline(
                    retriever=retriever, model=container.generator, mode=RetrievalMode.HYBRID
                ),
                judge=judge,
                name="retrieved",
            )
        )
    # Each rung adds one mechanism to the one below; all use the same runner as `adaptive`.
    answerer = build_answerer(settings, container)
    for rung in LADDER:
        if rung.name in wanted:
            results.append(
                await run_adaptive(
                    cases,
                    engine=AblationPipeline(
                        router=router,
                        answerer=answerer,
                        ablation=rung,
                        retriever=retriever,
                        context_blocks=settings.retrieval.context_blocks,
                    ),
                    judge=judge,
                    name=rung.name,
                )
            )

    if "adaptive" in wanted:
        results.append(await run_adaptive(cases, engine=engine, judge=judge))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"generation{args.suffix}.md").write_text(report(results, cases), "utf-8")

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
                    # Distinguishes a reasoned refusal from one caused by the deadline.
                    "seconds": o.seconds,
                    "papers_available": o.papers_available,
                    "papers_cited": o.papers_cited,
                    "text": o.text,
                }
                for o in r.outcomes
            ],
        }
    (RESULTS / f"generation{args.suffix}.json").write_text(json.dumps(by_config, indent=2), "utf-8")

    print("\n" + "\n".join(report(results, cases).splitlines()[:14]))
    print(f"\nwritten to {RESULTS / f'generation{args.suffix}.md'}")


if __name__ == "__main__":
    asyncio.run(main())
