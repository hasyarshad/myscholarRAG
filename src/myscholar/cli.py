"""Command line interface."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from myscholar.config import load_settings
from myscholar.container import build_container
from myscholar.domain.answers import Answer

app = typer.Typer(
    name="myscholar",
    help="Adaptive RAG over a personal corpus of research papers.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    trace: Annotated[bool, typer.Option("--trace", help="Show the path through the loop.")] = False,
) -> None:
    """Answer a question, with citations, or refuse."""
    from myscholar.domain.answers import Refusal
    from myscholar.generation.answerer import Answerer
    from myscholar.reasoning.engine import Engine
    from myscholar.reasoning.graph import Budget

    settings = load_settings()
    container = build_container(settings)

    engine = Engine(
        router=_build_router(settings),
        answerer=Answerer(generator=container.generator, utility=container.utility),
        budget=Budget(
            max_steps=settings.reasoning.max_steps,
            max_retrieval_rounds=settings.reasoning.max_retrieval_rounds,
            max_regenerations=settings.reasoning.max_regenerations,
            deadline_seconds=settings.reasoning.deadline_seconds,
        ),
    )

    response, loop_trace = asyncio.run(engine.answer_with_trace(question))

    if isinstance(response, Refusal):
        console.print(Panel(response.text, title="no answer", border_style="yellow"))
        console.print(f"[dim]reason: {response.reason.value}[/dim]")
    else:
        console.print(Panel(response.text, title="answer", border_style="green"))

    if isinstance(response, Answer) and response.citations:
        table = Table("Ref", "Source", "Section", "Pages", box=None)
        for citation in response.citations:
            table.add_row(
                citation.label,
                citation.document_title[:52],
                citation.section_heading[:28],
                f"{citation.page_start}-{citation.page_end}",
            )
        console.print(table)

    if trace:
        stages = Table("Stage", "Detail", box=None)
        for event in response.trace:
            stages.add_row(event.stage.value, event.detail)
        console.print(stages)
        console.print(f"[dim]path: {' -> '.join(loop_trace.path)}[/dim]")
        if loop_trace.stopped_because:
            console.print(f"[yellow]stopped early: {loop_trace.stopped_because}[/yellow]")


ingest_app = typer.Typer(help="Build the index from source documents.", no_args_is_help=True)
papers_app = typer.Typer(help="Inspect and curate the indexed corpus.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(papers_app, name="papers")


@app.command()
def config() -> None:
    """Show the resolved configuration and which components it selects."""
    settings = load_settings()

    table = Table("Setting", "Value", box=None)
    table.add_row("llm.provider", settings.llm.provider.value)
    table.add_row("llm.generator_model", settings.llm.generator_model)
    table.add_row("llm.utility_model", settings.llm.utility_model)
    table.add_row("embedding.provider", settings.embedding.provider.value)
    table.add_row("embedding.model_id", settings.embedding.model_id)
    table.add_row("embedding.dim", str(settings.embedding.dim))
    table.add_row("reranker.provider", settings.reranker.provider.value)
    table.add_row("database.url", settings.database.url)
    console.print(table)

    container = build_container(settings)
    console.print(
        f"\n[dim]resolved:[/dim] generator=[bold]{container.generator.model_id}[/bold] "
        f"embedder=[bold]{container.embedder.model_id}[/bold] "
        f"({container.embedder.dim}d) "
        f"reranker=[bold]{container.reranker.model_id}[/bold]"
    )


if __name__ == "__main__":  # pragma: no cover
    app()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _build_pipeline(settings: Any) -> tuple[Any, Any]:
    """Assemble the ingestion pipeline and the store it writes to."""
    from myscholar.adapters.extraction.docling_extractor import DoclingExtractor
    from myscholar.adapters.persistence.repository import PostgresStore, create_engine
    from myscholar.ingest.metadata import MetadataExtractor
    from myscholar.ingest.pipeline import IngestionPipeline
    from myscholar.ingest.sectionizer import Sectionizer

    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url, echo=settings.database.echo))

    pipeline = IngestionPipeline(
        extractor=DoclingExtractor(),
        embedder=container.embedder,
        store=store,
        sectionizer=Sectionizer(container.utility),
        metadata_extractor=MetadataExtractor(container.utility),
    )
    return pipeline, store


@ingest_app.command("run")
def ingest_run(
    path: Annotated[Path | None, typer.Option(help="Directory or single PDF.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-ingest unchanged files.")] = False,
    limit: Annotated[int, typer.Option(help="Process at most this many files.")] = 0,
) -> None:
    """Extract, chunk, embed and index source documents."""
    settings = load_settings()
    source = path or settings.papers_dir

    pdfs = sorted(source.glob("*.pdf")) if source.is_dir() else [source]
    if limit:
        pdfs = pdfs[:limit]
    if not pdfs:
        console.print(f"[yellow]no PDFs found in {source}[/yellow]")
        raise typer.Exit(1)

    console.print(f"ingesting [bold]{len(pdfs)}[/bold] document(s) from {source}")

    async def _run() -> Any:
        from myscholar.ingest.metadata import read_review_file

        pipeline, store = _build_pipeline(settings)
        await store.create_schema()

        review = settings.data_dir / "metadata.yaml"
        known = read_review_file(review) if review.exists() else {}
        if known:
            console.print(f"using reviewed metadata for {len(known)} document(s)")

        report = await pipeline.run(pdfs, known_metadata=known, force=force)
        await store.create_vector_indexes()
        return report, store

    report, _store = asyncio.run(_run())

    console.print(
        f"\n[green]indexed[/green] {report.documents} document(s), "
        f"{report.sections} section(s), {report.chunks} chunk(s)"
    )
    if report.section_stats:
        stats = report.section_stats
        total = sum(stats.values()) or 1
        console.print(
            f"[dim]headings resolved: {stats.get('pattern', 0)} by pattern, "
            f"{stats.get('model', 0)} by model, {stats.get('fallback', 0)} unclassified "
            f"({stats.get('pattern', 0) * 100 // total}% needed no model)[/dim]"
        )
    for skipped in report.skipped:
        console.print(f"[dim]skipped: {skipped}[/dim]")
    for warning in report.warnings[:10]:
        console.print(f"[yellow]warn:[/yellow] {warning}")
    for name, error in report.failed:
        console.print(f"[red]failed:[/red] {name}: {error}")

    if report.proposals:
        from myscholar.ingest.metadata import write_review_file

        target = settings.data_dir / "metadata.yaml"
        write_review_file(report.proposals, target)
        flagged = sum(1 for p in report.proposals if p.confidence_notes)
        console.print(
            f"\n[bold]metadata written to {target}[/bold]\n"
            f"  {len(report.proposals)} entries, {flagged} flagged for attention.\n"
            f"  Review it, set verified: true, then run: myscholar papers import-metadata\n"
            f"  [dim]Bibliographic questions are answered only from verified rows.[/dim]"
        )


# ---------------------------------------------------------------------------
# Corpus inspection
# ---------------------------------------------------------------------------


@papers_app.command("list")
def papers_list() -> None:
    """List indexed documents."""
    settings = load_settings()

    async def _list() -> Any:
        from myscholar.adapters.persistence.repository import PostgresStore, create_engine

        store = PostgresStore(create_engine(settings.database.url))
        return await store.list_documents(), await store.counts()

    rows, counts = asyncio.run(_list())
    if not rows:
        console.print("[yellow]nothing indexed yet — run: myscholar ingest run[/yellow]")
        return

    table = Table("Year", "Title", "Venue", "Sec", "Chunks", "OK", box=None)
    for row in rows:
        table.add_row(
            str(row.year or "—"),
            row.title[:62] + ("…" if len(row.title) > 62 else ""),
            (row.venue or "—")[:24],
            str(row.sections),
            str(row.chunks),
            "[green]y[/green]" if row.verified else "[yellow]n[/yellow]",
        )
    console.print(table)
    console.print(
        f"\n[dim]{counts['documents']} documents, {counts['sections']} sections, "
        f"{counts['chunks']} chunks, {counts['embeddings']} embeddings[/dim]"
    )
    if unverified := sum(1 for r in rows if not r.verified):
        console.print(
            f"[yellow]{unverified} document(s) have unverified metadata.[/yellow] "
            f"Review {settings.data_dir / 'metadata.yaml'} and run papers import-metadata."
        )


@papers_app.command("sections")
def papers_sections() -> None:
    """Show how sections were classified across the corpus."""
    settings = load_settings()

    async def _stats() -> dict[str, int]:
        from myscholar.adapters.persistence.repository import PostgresStore, create_engine

        store = PostgresStore(create_engine(settings.database.url))
        return await store.section_kind_counts()

    counts = asyncio.run(_stats())
    if not counts:
        console.print("[yellow]nothing indexed yet[/yellow]")
        return

    table = Table("Section kind", "Count", box=None)
    for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        table.add_row(kind, str(count))
    console.print(table)

    if other := counts.get("other", 0):
        total = sum(counts.values())
        console.print(
            f"\n[yellow]{other} of {total} sections ({other * 100 // total}%) are "
            f"unclassified.[/yellow] A high share means the sectionizer needs work — "
            f"section filtering depends on these."
        )


@papers_app.command("import-metadata")
def papers_import_metadata(
    path: Annotated[Path | None, typer.Option(help="Reviewed YAML file.")] = None,
) -> None:
    """Apply reviewed bibliographic metadata to the index.

    Updates the document rows, then rebuilds and re-embeds every affected
    chunk's contextual header — because the header names the paper, so
    correcting a title changes what those chunks embed as.

    This does not re-extract the PDFs, which is what makes it take seconds
    rather than the half hour a full `ingest run --force` costs.
    """
    from myscholar.ingest.metadata import MetadataFileError, read_review_file

    settings = load_settings()
    target = path or settings.data_dir / "metadata.yaml"
    if not target.exists():
        console.print(f"[red]no such file: {target}[/red]")
        raise typer.Exit(1)

    try:
        entries = read_review_file(target)
    except MetadataFileError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    verified = {name: meta for name, meta in entries.items() if meta.verified}
    if not verified:
        console.print(
            f"[yellow]{len(entries)} entries, none marked verified.[/yellow]\n"
            f"Set [bold]verified: true[/bold] on the entries you have checked."
        )
        return

    async def _apply() -> tuple[int, int]:
        from myscholar.adapters.persistence.repository import PostgresStore, create_engine
        from myscholar.ingest.chunker import build_context_header, strip_mathematics

        container = build_container(settings)
        store = PostgresStore(create_engine(settings.database.url))

        from myscholar.ingest.metadata import slugify

        updated_documents: dict[Any, Any] = {}
        taken = await store.taken_slugs()
        current = await store.slugs_by_filename()
        for filename, meta in verified.items():
            # The slug is derived from the title, so correcting a title leaves
            # a stale identifier behind. Slugs are the public handle for a
            # document, so they have to follow.
            #
            # A document's own current slug is excluded from the collision set:
            # treating it as taken makes every regeneration look like a clash
            # and appends a counter to every slug in the corpus.
            own = current.get(filename)
            others = taken - {own} if own else taken
            new_slug = slugify(meta.title, Path(filename).stem, taken=others)
            if own:
                taken.discard(own)
            taken.add(new_slug)
            document_id = await store.update_metadata(
                filename,
                {
                    "slug": new_slug,
                    "title": meta.title,
                    "authors": list(meta.authors),
                    "year": meta.year,
                    "venue": meta.venue,
                    "doi": meta.doi,
                    "metadata_verified": True,
                },
            )
            if document_id is not None:
                updated_documents[document_id] = meta

        rows = await store.chunks_for_reheadering()
        affected = [r for r in rows if r[1] in updated_documents]
        if not affected:
            return len(updated_documents), 0

        headers = [
            build_context_header(
                title=updated_documents[document_id].title,
                year=updated_documents[document_id].year,
                venue=updated_documents[document_id].venue,
                section_heading=heading,
                section_kind=kind,
            )
            for _chunk_id, document_id, _text, heading, kind in affected
        ]
        embed_texts = [
            f"{header}\n\n{strip_mathematics(text)}"
            for header, (_cid, _did, text, _h, _k) in zip(headers, affected, strict=True)
        ]
        vectors = await container.embedder.embed_documents(embed_texts)

        await store.replace_chunk_headers(
            [
                (row[0], header, vector)
                for row, header, vector in zip(affected, headers, vectors, strict=True)
            ],
            model_id=container.embedder.model_id,
        )
        return len(updated_documents), len(affected)

    documents, chunks = asyncio.run(_apply())

    if not documents:
        console.print(
            "[yellow]no indexed documents matched the reviewed entries.[/yellow]\n"
            "Filenames in the review file must match the ingested source paths."
        )
        return

    console.print(
        f"[green]updated[/green] {documents} document(s), "
        f"re-embedded {chunks} chunk(s) with corrected headers"
    )
    if skipped := len(entries) - len(verified):
        console.print(f"[dim]{skipped} entry(ies) not marked verified, left unchanged[/dim]")


@papers_app.command("show")
def papers_show(
    slug: Annotated[str, typer.Argument(help="Document slug, or any part of the title.")],
    section: Annotated[str | None, typer.Option(help="Only this section kind.")] = None,
    limit: Annotated[int, typer.Option(help="How many chunks to show.")] = 6,
    full: Annotated[bool, typer.Option("--full", help="Show whole chunks, not excerpts.")] = False,
) -> None:
    """Inspect the chunks stored for one document.

    The Phase 1 review step: check that section labels are right and that
    contextual headers name the paper and section, since a chunk whose header
    is wrong is retrievable only by accident.
    """
    settings = load_settings()

    async def _fetch() -> Any:
        from sqlalchemy import select

        from myscholar.adapters.persistence.orm import Chunk, Document, Section
        from myscholar.adapters.persistence.repository import PostgresStore, create_engine

        store = PostgresStore(create_engine(settings.database.url))
        async with store.session() as session:
            document = await session.scalar(
                select(Document).where(Document.slug.ilike(f"%{slug}%")).limit(1)
            )
            if document is None:
                return None, []

            statement = (
                select(Chunk, Section)
                .join(Section, Section.id == Chunk.section_id)
                .where(Chunk.document_id == document.id)
                .order_by(Section.ordinal, Chunk.ordinal_in_section)
            )
            if section:
                statement = statement.where(Section.kind == section)
            rows = await session.execute(statement.limit(limit))
            return document, rows.all()

    document, rows = asyncio.run(_fetch())
    if document is None:
        console.print(f"[red]no document matching {slug!r}[/red]")
        raise typer.Exit(1)
    if not rows:
        console.print(f"[yellow]no chunks found (section filter: {section})[/yellow]")
        return

    console.print(
        Panel(
            f"[bold]{document.title}[/bold]\n"
            f"{', '.join(document.authors) or 'authors unknown'}\n"
            f"{document.venue or 'venue unknown'} · {document.year or 'year unknown'}"
            + ("" if document.metadata_verified else "  [yellow](metadata unverified)[/yellow]"),
            border_style="blue",
        )
    )

    for chunk, section_row in rows:
        body = chunk.text if full else chunk.text[:400] + ("…" if len(chunk.text) > 400 else "")
        console.print(
            f"\n[dim]{section_row.kind} · {section_row.heading or '—'} · "
            f"chunk {chunk.ordinal_in_section} · p.{chunk.page} · "
            f"{chunk.token_count} tokens · {chunk.kind}[/dim]"
        )
        console.print(f"[cyan]header:[/cyan] [dim]{chunk.context_header}[/dim]")
        console.print(body)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _build_retriever(settings: Any, *, with_reranker: bool = True) -> Any:
    from myscholar.adapters.persistence.repository import PostgresStore, create_engine
    from myscholar.retrieval.retriever import HybridRetriever

    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))

    return HybridRetriever(
        store=store,
        embedder=container.embedder,
        reranker=container.reranker if with_reranker else None,
        top_k=settings.retrieval.top_k,
        rerank_k=settings.retrieval.rerank_k,
        window=settings.retrieval.window,
        dense_weight=settings.retrieval.dense_weight,
        lexical_weight=settings.retrieval.lexical_weight,
    )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to search for.")],
    mode: Annotated[
        str, typer.Option(help="dense | hybrid | full (adds the cross-encoder).")
    ] = "full",
    limit: Annotated[int, typer.Option(help="How many results to show.")] = 5,
    show_text: Annotated[bool, typer.Option("--text", help="Show passage text.")] = False,
) -> None:
    """Search the corpus. Generation is not involved.

    The three modes are the Phase 2 ablation: dense reproduces the Phase 1
    baseline, hybrid adds the lexical branch, full adds the cross-encoder.
    """
    from myscholar.retrieval.retriever import RetrievalMode

    settings = load_settings()
    try:
        retrieval_mode = RetrievalMode(mode)
    except ValueError:
        console.print(f"[red]unknown mode {mode!r}; use dense, hybrid or full[/red]")
        raise typer.Exit(1) from None

    async def _run() -> Any:
        retriever = _build_retriever(settings, with_reranker=retrieval_mode is RetrievalMode.FULL)
        return await retriever.retrieve(query, mode=retrieval_mode, limit=limit)

    result = asyncio.run(_run())

    if not result.blocks:
        console.print("[yellow]nothing found[/yellow]")
        return

    timings = " · ".join(f"{stage} {ms}ms" for stage, ms in result.timings_ms.items())
    console.print(
        f"[dim]mode={result.mode.value} · {len(result.candidates)} candidates · {timings}"
        + ("  [yellow](reranker unavailable)[/yellow]" if result.degraded else "")
        + "[/dim]\n"
    )

    for block in result.blocks:
        console.print(
            f"[bold cyan]{block.label}[/bold cyan] "
            f"[bold]{block.document_title[:70]}[/bold]\n"
            f"[dim]{block.section_heading[:60]} · p.{block.page_start}-{block.page_end} · "
            f"score {block.score:.4f}[/dim]"
        )
        if show_text:
            console.print(f"{block.text[:500]}{'…' if len(block.text) > 500 else ''}\n")
        else:
            console.print(f"[dim]{block.text[:160].replace(chr(10), ' ')}…[/dim]\n")


def _build_router(settings: Any) -> Any:
    from myscholar.adapters.persistence.repository import PostgresStore, create_engine
    from myscholar.query.classifier import QueryClassifier
    from myscholar.query.planner import QueryPlanner
    from myscholar.query.router import QueryRouter
    from myscholar.query.transforms import QueryTransformer
    from myscholar.retrieval.retriever import HybridRetriever

    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))

    return QueryRouter(
        classifier=QueryClassifier(container.utility),
        planner=QueryPlanner(top_k=settings.retrieval.top_k, rerank_k=settings.retrieval.rerank_k),
        transformer=QueryTransformer(container.utility),
        retriever=HybridRetriever(
            store=store,
            embedder=container.embedder,
            reranker=container.reranker,
            top_k=settings.retrieval.top_k,
            rerank_k=settings.retrieval.rerank_k,
            window=settings.retrieval.window,
            dense_weight=settings.retrieval.dense_weight,
            lexical_weight=settings.retrieval.lexical_weight,
        ),
        store=store,
    )


@app.command()
def route(
    question: Annotated[str, typer.Argument(help="The question to route.")],
    show_text: Annotated[bool, typer.Option("--text", help="Show passage text.")] = False,
) -> None:
    """Show how a question is classified and what the router does with it.

    Retrieval only — no answer is generated. This is the Phase 3 gate: the
    classification, the plan it produced, and the queries actually searched for.
    """
    settings = load_settings()
    result = asyncio.run(_build_router(settings).route(question))

    classification, plan = result.classification, result.plan
    timings = " · ".join(f"{stage} {ms}ms" for stage, ms in result.timings_ms.items())

    console.print(
        Panel(
            f"[bold]{classification.query_class.value}[/bold] "
            f"[dim](decided by {classification.decided_by})[/dim]\n"
            f"[dim]{classification.reasoning or '—'}[/dim]",
            title="classification",
            border_style="cyan",
        )
    )

    routing = []
    if plan.use_sql:
        routing.append("[bold]SQL over verified metadata[/bold] — no vector search")
    elif not plan.retrieve:
        routing.append("[bold]canned reply[/bold] — no retrieval, no generation")
    else:
        routing.append(f"transforms: {', '.join(t.value for t in plan.transforms) or 'none'}")
        routing.append(f"top_k={plan.top_k} rerank_k={plan.rerank_k}")
        if plan.filter_document_titles:
            routing.append(f"filtered to: {plan.filter_document_titles[0][:60]}")
        if plan.boost_sections:
            routing.append(f"sections: {', '.join(sorted(s.value for s in plan.boost_sections))}")
        if plan.max_chunks_per_document:
            routing.append(f"max {plan.max_chunks_per_document} blocks per paper")
    console.print(Panel("\n".join(routing), title="plan", border_style="green"))

    if result.bibliographic is not None:
        console.print(Panel(result.bibliographic.text, title="answer (SQL)", border_style="blue"))
        if result.bibliographic.unverified:
            console.print(
                f"[yellow]{result.bibliographic.unverified} document(s) excluded "
                f"as unverified[/yellow]"
            )
    elif plan.canned_response:
        console.print(Panel(plan.canned_response, title="answer", border_style="yellow"))

    if result.queries:
        table = Table("Origin", "Query", "Weight", box=None)
        for query in result.queries:
            table.add_row(query.origin.value, query.text[:80], f"{query.weight:.1f}")
        console.print(table)

    for block in result.blocks:
        console.print(
            f"\n[bold cyan]{block.label}[/bold cyan] [bold]{block.document_title[:64]}[/bold]"
            f"\n[dim]{block.section_heading[:56]} · p.{block.page_start}-{block.page_end}[/dim]"
        )
        console.print(
            f"[dim]{block.text[:200].replace(chr(10), ' ')}…[/dim]"
            if not show_text
            else block.text[:600]
        )

    console.print(f"\n[dim]{timings}[/dim]")
