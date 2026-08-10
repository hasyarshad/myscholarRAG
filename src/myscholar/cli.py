"""Command line interface."""

from __future__ import annotations

import asyncio
from typing import Annotated

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
