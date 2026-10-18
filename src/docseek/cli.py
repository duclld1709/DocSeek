"""DocSeek command line interface (Typer)."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from docseek import __version__, doctor
from docseek.core.config import Settings
from docseek.core.logging import configure_logging

app = typer.Typer(
    name="docseek",
    help="Measurable agentic retrieval over technical documentation.",
    no_args_is_help=True,
    add_completion=False,
)

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
