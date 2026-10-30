"""Local RAG CLI. Nothing executes on import."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from uni_rag.config import PROJECT_ROOT, Config, load_config
from uni_rag.generate import ask as generate_answer
from uni_rag.generate import source_lines
from uni_rag.ingest.pipeline import ingest as run_ingest
from uni_rag.models import Answer
from uni_rag.ollama import Ollama

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
console = Console()
ConfigOption = Annotated[Path, typer.Option("--config", help="Path to a local TOML config")]


class EnergyBackend(StrEnum):
    auto = "auto"
    macmon = "macmon"
    powermetrics = "powermetrics"
    none = "none"


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
    energy: Annotated[EnergyBackend | None, typer.Option()] = None,
) -> None:
    """Ask a question using local document context and Ollama."""
    cfg = configuration(config)
    run_query(question, cfg, source, model, k, think, show_context, file_type, path_prefix, energy)


def run_query(
    question: str,
    cfg: Config,
    source: str | None,
    model: str | None,
    k: int | None,
    think: bool,
    show_context: bool = False,
    file_type: str | None = None,
    path_prefix: str | None = None,
    energy: EnergyBackend | None = None,
    history: list[dict] | None = None,
) -> Answer:
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
                energy=energy,
                mode="chat" if history is not None else "ask",
                history=history,
            )
        console.print()
        for line in source_lines(answer):
            console.print(line, markup=False)
        if show_context:
            from uni_rag.generate import context_text

            console.print(context_text(answer.hits), markup=False)
        return answer
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
    energy: Annotated[EnergyBackend | None, typer.Option()] = None,
) -> None:
    """Interactive grounded chat. Type /exit or press Ctrl-D to leave."""
    cfg = configuration(config)
    history: list[dict] = []
    console.print("Local chat. /exit to leave; the last three turns stay in memory.")
    while True:
        try:
            question = console.input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if question in {"/exit", "/quit"}:
            return
        if question:
            answer = run_query(
                question, cfg, source, model, k, think, energy=energy, history=history
            )
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer.text},
                ]
            )
            history = history[-6:]


@app.command()
def stats(config: ConfigOption = PROJECT_ROOT / "config.toml") -> None:
    """Inspect the local manifest/index/metrics without touching source folders."""
    from uni_rag.stats import stats as index_stats

    console.print_json(json.dumps(index_stats(configuration(config))))


@app.command(name="eval")
def eval_command(
    config: ConfigOption = PROJECT_ROOT / "config.toml",
    questions: Path = PROJECT_ROOT / "eval/questions.yaml",
    models: str | None = None,
    think: Annotated[bool, typer.Option("--think/--no-think")] = False,
    k: Annotated[int | None, typer.Option(min=1)] = None,
    energy: Annotated[EnergyBackend | None, typer.Option()] = None,
    judge_model: str | None = None,
) -> None:
    """Benchmark local models with warm-up, unloading and optional local judging."""
    from uni_rag.evaluate import evaluate

    cfg = configuration(config)
    selected = [name.strip() for name in models.split(",")] if models is not None else None
    try:
        result = evaluate(cfg, questions, selected, think, k, energy, judge_model)
        console.print_json(json.dumps(result))
    except Exception as exc:
        console.print(f"Evaluation failed; completed results are retained: {exc}", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def report(config: ConfigOption = PROJECT_ROOT / "config.toml", run_id: str | None = None) -> None:
    """Write a Markdown summary for the latest completed eval (or --run-id)."""
    from uni_rag.report import report as write_report

    try:
        path, text = write_report(configuration(config), run_id)
        console.print(text, markup=False)
        console.print(f"Written to {path}", markup=False)
    except Exception as exc:
        console.print(f"Report failed: {exc}", markup=False)
        raise typer.Exit(1) from exc
