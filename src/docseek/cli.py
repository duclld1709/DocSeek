"""DocSeek command line interface (Typer)."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from docseek import __version__, doctor
from docseek.core.config import Settings, load_corpus_config
from docseek.core.logging import configure_logging
from docseek.data.fetch import FetchError, fetch_fastapi_docs

app = typer.Typer(
    name="docseek",
    help="Measurable agentic retrieval over technical documentation.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Fetch and generate corpora.", no_args_is_help=True)
app.add_typer(data_app, name="data")

console = Console()

_STATUS_STYLE = {
    doctor.Status.OK: "[green]ok[/green]",
    doctor.Status.WARN: "[yellow]warn[/yellow]",
    doctor.Status.FAIL: "[red]FAIL[/red]",
}


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"docseek {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    log_level: Annotated[str, typer.Option(help="DEBUG, INFO, WARNING or ERROR.")] = "WARNING",
    _version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    configure_logging(log_level.upper())


@app.command("doctor")
def doctor_cmd(
    strict: Annotated[bool, typer.Option(help="Also fail on warnings.")] = False,
) -> None:
    """Check configs, Qdrant, Neo4j, API keys (set / not set only) and Tesseract."""
    results = doctor.run_checks(Settings())
    table = Table(title="docseek doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for r in results:
        table.add_row(r.name, _STATUS_STYLE[r.status], r.detail)
    console.print(table)
    if not doctor.is_healthy(results, strict=strict):
        console.print("[red]Some required checks failed.[/red] Is `docker compose up -d` running?")
        raise typer.Exit(code=1)


@data_app.command("fetch")
def data_fetch(
    force: Annotated[bool, typer.Option(help="Re-clone even if already pinned.")] = False,
) -> None:
    """Sparse-clone FastAPI docs at the pinned commit and write data/raw/MANIFEST.json."""
    corpus = load_corpus_config(Settings().configs_dir)
    try:
        result = fetch_fastapi_docs(corpus.fastapi, corpus.manifest_path, force=force)
    except FetchError as exc:
        console.print(f"[red]fetch failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    m = result.manifest
    action = "cloned" if result.cloned else "already at pinned commit"
    console.print(f"{corpus.fastapi.dest}: {action} ({m.commit[:12]}, {m.commit_date})")
    console.print(f"{m.file_count} files: {m.counts_by_extension} -> {corpus.manifest_path}")
