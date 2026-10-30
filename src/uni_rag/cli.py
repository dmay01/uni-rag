"""Local RAG CLI. Nothing executes on import."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from uni_rag.config import PROJECT_ROOT, Config, load_config
from uni_rag.generate import ask as generate_answer
from uni_rag.generate import source_lines
from uni_rag.ingest.pipeline import ingest as run_ingest
from uni_rag.ollama import Ollama

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
console = Console()
ConfigOption = Annotated[Path, typer.Option("--config", help="Path to a local TOML config")]


def configuration(path: Path) -> Config:
    try:
        return load_config(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        console.print(f"Configuration error: {exc}", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def ingest(
    config: ConfigOption = PROJECT_ROOT / "config.toml",
    source: str | None = None,
    only: str | None = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    dry_run: bool = False,
) -> None:
    """Discover and index documents; --dry-run never opens document contents."""
    try:
        console.print_json(
            json.dumps(run_ingest(configuration(config), source, only, limit, dry_run))
        )
    except Exception as exc:
        console.print(f"Ingest failed (completed batches are retained): {exc}", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def ask(
    question: str,
    config: ConfigOption = PROJECT_ROOT / "config.toml",
    source: str | None = None,
    model: str | None = None,
    k: Annotated[int | None, typer.Option(min=1)] = None,
    think: Annotated[bool, typer.Option("--think/--no-think")] = False,
    show_context: bool = False,
    file_type: str | None = None,
    path_prefix: str | None = None,
) -> None:
    """Ask a question using local document context and Ollama."""
    cfg = configuration(config)
    try:
        with Ollama(cfg) as client:
            answer = generate_answer(
                question,
                cfg,
                client,
                model,
                k,
                think,
                source,
                file_type,
                path_prefix,
                on_token=lambda text: console.print(text, end="", markup=False),
            )
        console.print()
        for line in source_lines(answer):
            console.print(line, markup=False)
        if show_context:
            from uni_rag.generate import context_text

            console.print(context_text(answer.hits), markup=False)
    except Exception as exc:
        console.print(f"Query failed: {exc}", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def chat(
    config: ConfigOption = PROJECT_ROOT / "config.toml",
    source: str | None = None,
    model: str | None = None,
    k: Annotated[int | None, typer.Option(min=1)] = None,
    think: Annotated[bool, typer.Option("--think/--no-think")] = False,
) -> None:
    """Interactive, independently grounded questions. Type /exit or press Ctrl-D to leave."""
    console.print("Local chat. Ask self-contained questions; /exit to leave.")
    while True:
        try:
            question = console.input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if question in {"/exit", "/quit"}:
            return
        if question:
            ask(question, config, source, model, k, think)
