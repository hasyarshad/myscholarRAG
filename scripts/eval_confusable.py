"""Attribution precision: did the answer cite the right paper, and not a similar one.

    uv run python scripts/eval_confusable.py
    uv run python scripts/eval_confusable.py --configs cited,graded-strict

Several papers in the corpus share a subject (six apply predictive control to
motor drives, for example). Citing the wrong one yields well-supported sentences,
so the unsupported-claim rate cannot see it.

    retrieved   the expected paper was among the blocks the generator held
    cited       the expected paper appears in the answer's citations
    clean       the expected paper is cited and no decoy is

`clean` is the headline; `cited - clean` is contamination. Fifteen questions
detect only a large effect, so the rule in tests/eval/confusable_set.yaml keeps
the relevance grader only on a move of 20 points or more in `clean`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
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
from myscholar.evaluation.ablation import LADDER, AblationPipeline  # noqa: E402

TEST_SET = Path("tests/eval/confusable_set.yaml")
RESULTS = Path("results")
QUESTION_TIMEOUT = 180.0


@dataclass
class Outcome:
    question: str
    expect: str
    decoys: tuple[str, ...]
    retrieved: bool
    cited: bool
    contaminated: bool
    cited_slugs: tuple[str, ...]
    seconds: float

    @property
    def clean(self) -> bool:
        return self.cited and not self.contaminated


@dataclass
class ConfigResult:
    name: str
    outcomes: list[Outcome] = field(default_factory=list)

    def _rate(self, attr: str) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if getattr(o, attr)) / len(self.outcomes)

    @property
    def retrieved(self) -> float:
        return self._rate("retrieved")

    @property
    def cited(self) -> float:
        return self._rate("cited")

    @property
    def clean(self) -> float:
        return self._rate("clean")

    @property
    def mean_seconds(self) -> float:
        return sum(o.seconds for o in self.outcomes) / max(len(self.outcomes), 1)


async def run(
    cases: list[dict[str, Any]], *, pipeline: Any, name: str, slugs: dict[str, str]
) -> ConfigResult:
    result = ConfigResult(name=name)
    for index, case in enumerate(cases, 1):
        started = time.monotonic()
        try:
            async with asyncio.timeout(QUESTION_TIMEOUT):
                response, _trace, blocks = await pipeline.answer_with_context(case["question"])
        except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
            print(f"  [{name}] {index}/{len(cases)} FAILED: {type(exc).__name__}", flush=True)
            continue

        expect = case["expect"]
        decoys = tuple(case.get("decoys", ()))
        held = {slugs.get(str(b.document_id), "") for b in blocks}
        cited_slugs = tuple(
            slugs.get(str(c.document_id), "") for c in getattr(response, "citations", ())
        )
        outcome = Outcome(
            question=case["question"],
            expect=expect,
            decoys=decoys,
            retrieved=expect in held,
            cited=expect in cited_slugs,
            contaminated=any(d in cited_slugs for d in decoys),
            cited_slugs=cited_slugs,
            seconds=time.monotonic() - started,
        )
        result.outcomes.append(outcome)
        mark = "OK " if outcome.clean else ("DECOY" if outcome.contaminated else "MISS ")
        print(f"  [{name}] {index}/{len(cases)} {mark} {case['question'][:54]}", flush=True)
    return result


def report(results: list[ConfigResult], n: int) -> str:
    lines = [
        "# Attribution precision on confusable questions",
        "",
        f"{n} questions where several papers cover the subject and one is right.",
        "",
        "`retrieved`: the expected paper reached the generator. `cited`: it appears",
        "in the answer. `clean`: it is cited and no decoy is. `clean` is the headline;",
        "the gap between `cited` and `clean` is contamination.",
        "",
        "| Configuration | retrieved | cited | **clean** | mean latency |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.retrieved:.0%} | {r.cited:.0%} | **{r.clean:.0%}** "
            f"| {r.mean_seconds:.1f}s |"
        )
    lines += [
        "",
        "Fifteen questions can detect only a large effect. The rule fixed before",
        "running: keep the grader only on a move of 20 points or more in `clean`.",
        "",
        "## Questions no configuration got clean",
        "",
    ]
    everywhere = [
        o.question
        for o in results[0].outcomes
        if all(not any(x.clean for x in r.outcomes if x.question == o.question) for r in results)
    ]
    lines += [f"- {q}" for q in everywhere] or ["- none"]
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="cited,grounded,graded,graded-strict,adaptive")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = [c for c in yaml.safe_load(TEST_SET.read_text("utf-8")) if isinstance(c, dict)]
    if args.limit:
        cases = cases[: args.limit]
    wanted = {name.strip() for name in args.configs.split(",")}

    settings = load_settings()
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))
    documents = await store.list_documents()
    slugs = {str(d.id): d.slug for d in documents}

    retriever = build_retriever(settings, container, store)
    router = build_router(
        settings, container, store, retriever=retriever, titles=[d.title for d in documents]
    )
    answerer = build_answerer(settings, container)

    print(f"{len(cases)} confusable questions x {len(wanted)} configuration(s)\n", flush=True)
    results: list[ConfigResult] = []
    for rung in LADDER:
        if rung.name in wanted:
            pipeline = AblationPipeline(
                router=router,
                answerer=answerer,
                ablation=rung,
                retriever=retriever,
                context_blocks=settings.retrieval.context_blocks,
            )
            results.append(await run(cases, pipeline=pipeline, name=rung.name, slugs=slugs))
    if "adaptive" in wanted:
        engine = build_engine(settings, container, store, router=router)
        results.append(await run(cases, pipeline=engine, name="adaptive", slugs=slugs))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "confusable.md").write_text(report(results, len(cases)), "utf-8")
    print("\n" + report(results, len(cases)))
    print(f"written to {RESULTS / 'confusable.md'}")


if __name__ == "__main__":
    asyncio.run(main())
