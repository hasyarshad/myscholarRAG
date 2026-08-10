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
from myscholar.reasoning import Engine

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
    trace: Annotated[bool, typer.Option("--trace", help="Show pipeline stages.")] = False,
) -> None:
    """Ask a question about the corpus."""
    container = build_container()
    engine = Engine(generator=container.generator, utility=container.utility)

    response = asyncio.run(engine.answer(question))

    if response.refused:
        console.print(Panel(response.text, title="No answer", border_style="yellow"))
    else:
        console.print(Panel(response.text, title="Answer", border_style="green"))

    if isinstance(response, Answer) and response.citations:
        table = Table("Ref", "Source", "Section", box=None)
        for citation in response.citations:
            table.add_row(citation.label, citation.document_title, citation.section_heading)
        console.print(table)

    if trace:
        stages = Table("Stage", "Detail", "ms", box=None)
        for event in response.trace:
            stages.add_row(event.stage.value, event.detail, str(event.elapsed_ms))
        console.print(stages)


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
    """Apply reviewed bibliographic metadata to the index."""
    from myscholar.ingest.metadata import read_review_file

    settings = load_settings()
    target = path or settings.data_dir / "metadata.yaml"
    if not target.exists():
        console.print(f"[red]no such file: {target}[/red]")
        raise typer.Exit(1)

    entries = read_review_file(target)
    verified = {k: v for k, v in entries.items() if v.verified}
    console.print(
        f"{len(entries)} entries, {len(verified)} marked verified.\n"
        f"[dim]Re-run `myscholar ingest run --force` to apply them.[/dim]"
    )
