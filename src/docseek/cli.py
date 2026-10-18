"""DocSeek command line interface (Typer)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from docseek import __version__, doctor
from docseek.core.config import Settings, load_corpus_config, load_models_config, load_principals
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


@app.command("ingest")
def ingest_cmd(
    force: Annotated[bool, typer.Option(help="Rebuild even if inputs are unchanged.")] = False,
    offline_tokens: Annotated[
        bool, typer.Option(help="Count tokens with the regex counter (no model download).")
    ] = False,
) -> None:
    """Parse corpora into data/processed/{sections,chunks}.jsonl and stats.json."""
    import json
    from datetime import UTC, datetime

    from docseek.core.config import load_ingest_config
    from docseek.ingest import pipeline
    from docseek.ingest.chunking import ChunkingConfig
    from docseek.ingest.tokens import HFTokenCounter, RegexTokenCounter, TokenCounter

    settings = Settings()
    corpus = load_corpus_config(settings.configs_dir)
    ingest_cfg = load_ingest_config(settings.configs_dir)
    if ingest_cfg.chunking.strategy != "section":
        console.print(f"[red]unsupported chunking strategy:[/red] {ingest_cfg.chunking.strategy}")
        raise typer.Exit(code=1)
    chunking = ChunkingConfig(ingest_cfg.chunking.max_tokens, ingest_cfg.chunking.overlap_tokens)
    counter: TokenCounter = (
        RegexTokenCounter()
        if offline_tokens
        else HFTokenCounter(load_models_config(settings.configs_dir).embedding.default)
    )
    manifest = corpus.manifest_path
    commit = (
        json.loads(manifest.read_text(encoding="utf-8"))["commit"] if manifest.is_file() else None
    )
    out_dir = ingest_cfg.output_dir
    digest = pipeline.input_hash(corpus, chunking, counter, commit=commit)
    if not force and pipeline.is_up_to_date(out_dir, digest):
        console.print(f"{out_dir}: up to date (input hash {digest[:12]}), nothing to do")
        return
    try:
        result = pipeline.run_pipeline(
            corpus, load_principals(settings.configs_dir), counter, chunking, repo_root=Path.cwd()
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    n_sec = pipeline.write_jsonl(out_dir / "sections.jsonl", result.sections)
    n_chunk = pipeline.write_jsonl(out_dir / "chunks.jsonl", result.chunks)
    stats = pipeline.corpus_stats(result, counter)
    stats.update(
        input_hash=digest,
        corpus_commit=commit,
        chunking=chunking.__dict__,
        created_at=datetime.now(UTC).isoformat(),
    )
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    table = Table(title="docseek ingest")
    table.add_column("source")
    for col in ("documents", "sections", "chunks"):
        table.add_column(col, justify="right")
    for src, row in stats["by_source"].items():
        table.add_row(src, *(str(row[c]) for c in ("documents", "sections", "chunks")))
    console.print(table)
    console.print(
        f"{n_sec} sections, {n_chunk} chunks -> {out_dir}; issues {stats['issues']}; "
        f"links {stats['links']}"
    )
    if stats["unresolved_include_syntax"]:
        console.print("[red]unresolved {* ... *} include syntax left in output[/red]")
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


@data_app.command("gen-internal")
def data_gen_internal(
    force: Annotated[bool, typer.Option(help="Regenerate docs that already exist.")] = False,
    only: Annotated[
        list[str] | None, typer.Option(help="Only these KB ids (repeatable), e.g. kb-0001.")
    ] = None,
    workers: Annotated[int, typer.Option(min=1, max=16, help="Parallel LLM calls.")] = 4,
) -> None:
    """Generate Corpus B (Acme internal KB) from configs/internal_kb_plan.yaml."""
    from docseek.core.runs import environment_info, make_run_dir, write_json
    from docseek.core.tracing import JsonlTraceSink, Tracer, read_traces, summarize_llm_costs
    from docseek.data.gen_internal import KBPlan, generate_internal_kb, load_plan
    from docseek.ingest.internal_kb import KBValidationError, load_kb
    from docseek.llm.client import build_llm_client

    settings = Settings()
    corpus = load_corpus_config(settings.configs_dir)
    role = load_models_config(settings.configs_dir).llm.generator
    plan = load_plan(settings.configs_dir / "internal_kb_plan.yaml")
    if only:
        unknown = set(only) - {d.id for d in plan.docs}
        if unknown:
            console.print(f"[red]unknown KB ids:[/red] {sorted(unknown)}")
            raise typer.Exit(code=1)
        plan = KBPlan(company=plan.company, docs=[d for d in plan.docs if d.id in only])
    run_dir = make_run_dir(settings.experiments_dir, "gen-internal")
    traces = run_dir / "traces.jsonl"
    llm = build_llm_client(settings, Tracer(JsonlTraceSink(traces)))
    out_dir = corpus.internal_kb.dir
    results = generate_internal_kb(
        plan, llm=llm, role=role, out_dir=out_dir, force=force, workers=workers
    )
    events = read_traces(traces) if traces.exists() else []
    cost = summarize_llm_costs(events)
    write_json(run_dir / "report.json", {"results": [r.__dict__ for r in results]})
    write_json(
        run_dir / "env.json",
        environment_info(command="data gen-internal", model=role.model, cost=cost.__dict__),
    )
    table = Table(title="gen-internal")
    for col in ("id", "status", "attempts", "problems"):
        table.add_column(col, overflow="fold")
    for r in results:
        table.add_row(r.id, r.status, str(r.attempts), "; ".join(r.problems))
    console.print(table)
    console.print(
        f"LLM calls: {cost.llm_calls} ({cost.cached_calls} cached), "
        f"tokens in/out {cost.tokens_in}/{cost.tokens_out}, spent ${cost.cost_usd:.4f} "
        f"-> {run_dir}"
    )
    failed = [r.id for r in results if r.status == "failed"]
    if failed:
        console.print(f"[red]failed:[/red] {failed}")
        raise typer.Exit(code=1)
    try:
        docs = load_kb(out_dir, corpus.internal_kb, load_principals(settings.configs_dir))
    except KBValidationError as exc:
        if only:  # a partial KB may legitimately reference docs not generated yet
            console.print(f"[yellow]partial KB not validated:[/yellow] {exc}")
            return
        console.print(f"[red]KB validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"{len(docs)} KB docs valid in {out_dir}")
