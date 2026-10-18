"""DocSeek command line interface (Typer)."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console
from rich.table import Table

from docseek import __version__, doctor
from docseek.core.config import Settings, load_corpus_config, load_models_config, load_principals
from docseek.core.logging import configure_logging
from docseek.data.fetch import FetchError, fetch_fastapi_docs
from docseek.eval import human_cli

app = typer.Typer(
    name="docseek",
    help="Measurable agentic retrieval over technical documentation.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Fetch and generate corpora.", no_args_is_help=True)
app.add_typer(data_app, name="data")
eval_app = typer.Typer(help="Golden set, judges and experiments.", no_args_is_help=True)
graph_app = typer.Typer(help="Knowledge graph (Neo4j).", no_args_is_help=True)
app.add_typer(graph_app, name="graph")
app.add_typer(eval_app, name="eval")
human_cli.register(eval_app)

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
    _tolerate_unencodable_output()
    configure_logging(log_level.upper())


def _tolerate_unencodable_output() -> None:
    """Docs contain characters a Windows cp1252 console cannot print; replace, never crash."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


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


@app.command("index")
def index_cmd(
    config: Annotated[Path, typer.Option(help="Experiment YAML whose retriever to build.")],
) -> None:
    """Build (or reuse from cache) the index behind an experiment config's retriever."""
    import time

    from docseek.eval.runner import build_retriever, load_experiment

    cfg = load_experiment(config)
    inputs = _eval_inputs(cfg.golden, cfg.processed_dir, cfg.chunks_file)
    start = time.perf_counter()
    retriever = build_retriever(cfg, inputs["chunks"], [], inputs["policy"])  # type: ignore[arg-type]
    console.print(f"{retriever.name}: ready in {time.perf_counter() - start:.2f}s")


@app.command("search")
def search_cmd(
    text: Annotated[str, typer.Argument(help="Query text.")],
    config: Annotated[Path, typer.Option(help="Experiment YAML.")] = Path(
        "configs/experiments/bm25f.yaml"
    ),
    principal: Annotated[str, typer.Option(help="Who is asking (ACL).")] = "anonymous",
    k: Annotated[int, typer.Option(help="Number of hits.")] = 10,
) -> None:
    """Run one query against a configured retriever and print the hits."""
    from docseek.core.models import Query
    from docseek.eval.runner import build_retriever, load_experiment

    cfg = load_experiment(config)
    inputs = _eval_inputs(cfg.golden, cfg.processed_dir, cfg.chunks_file)
    retriever = build_retriever(cfg, inputs["chunks"], [], inputs["policy"])  # type: ignore[arg-type]
    hits = retriever.search(Query(qid="cli", text=text, principal=principal), k)
    table = Table(title=f"{retriever.name}, principal={principal}")
    for col in ("rank", "score", "section"):
        table.add_column(col)
    for h in hits:
        table.add_row(str(h.rank), f"{h.score:.3f}", h.section_id)
    console.print(table)


@app.command("project")
def project_cmd(
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("data/projected"),
) -> None:
    """Write the markdown projection used by the FS tools (one file per page)."""
    from docseek.core.models import Section
    from docseek.ingest.pipeline import read_jsonl
    from docseek.ingest.projection import project

    manifest = project(read_jsonl(Path("data/processed/sections.jsonl"), Section), out)
    console.print(f"{len(manifest['files'])} files -> {out}")


@graph_app.command("build")
def graph_build(
    concept_pages: Annotated[
        int, typer.Option(help="Pages to send to the query LLM for Concept extraction (0 = none).")
    ] = 0,
) -> None:
    """Extract the graph from sections, docs_src and release notes and load it into Neo4j."""
    import json

    from docseek.core.models import Section
    from docseek.core.runs import environment_info, make_run_dir, write_json
    from docseek.core.tracing import JsonlTraceSink, Tracer, read_traces, summarize_llm_costs
    from docseek.graph.concepts import make_concept_fn
    from docseek.graph.extract import build_graph, load_code_files
    from docseek.graph.graph_retriever import connect_neo4j
    from docseek.graph.loader import graph_counts, load_graph
    from docseek.ingest.pipeline import read_jsonl
    from docseek.llm.client import build_llm_client

    settings = Settings()
    corpus = load_corpus_config(settings.configs_dir)
    raw = corpus.fastapi.dest
    sections = read_jsonl(Path("data/processed/sections.jsonl"), Section)
    run_dir = make_run_dir(settings.experiments_dir, "graph_build")
    concept_fn = None
    if concept_pages > 0:
        llm = build_llm_client(settings, Tracer(JsonlTraceSink(run_dir / "traces.jsonl")))
        concept_fn = make_concept_fn(llm, load_models_config(settings.configs_dir).llm.query_role)
    notes = raw / "docs" / "en" / "docs" / "release-notes.md"
    g = build_graph(
        sections,
        load_code_files(raw / "docs_src", raw),
        release_notes=notes.read_text(encoding="utf-8") if notes.is_file() else None,
        concept_fn=concept_fn,
        max_concept_pages=concept_pages,
    )
    with connect_neo4j() as driver:
        load_graph(driver, g, wipe=True)
        counts = graph_counts(driver)
    traces = run_dir / "traces.jsonl"
    cost = summarize_llm_costs(read_traces(traces)).__dict__ if traces.exists() else None
    write_json(run_dir / "report.json", {"extracted": g.counts(), "loaded": counts, "cost": cost})
    write_json(
        run_dir / "env.json", environment_info(command="graph build", concept_pages=concept_pages)
    )
    console.print(json.dumps(counts, indent=1))
    console.print(f"-> {run_dir}")


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
    max_attempts: Annotated[
        int, typer.Option(min=1, max=10, help="Tries per doc (each retry gets feedback).")
    ] = 3,
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
        plan,
        llm=llm,
        role=role,
        out_dir=out_dir,
        force=force,
        workers=workers,
        max_attempts=max_attempts,
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


def _eval_inputs(
    golden_path: Path, processed_dir: Path, chunks_file: str = "chunks.jsonl"
) -> dict[str, object]:
    """Golden set, sections, chunks and access policy for ``eval run`` / ``eval grid``."""
    from docseek.core.acl import AccessPolicy
    from docseek.core.models import Chunk, Section
    from docseek.eval.golden import load_golden
    from docseek.ingest.pipeline import read_jsonl

    for path in (golden_path, processed_dir / "sections.jsonl"):
        if not path.is_file():
            console.print(
                f"[red]missing {path}[/red] (run `docseek ingest` / build the golden set)"
            )
            raise typer.Exit(code=1)
    return {
        "golden": load_golden(golden_path),
        "sections": read_jsonl(processed_dir / "sections.jsonl", Section),
        "chunks": read_jsonl(processed_dir / chunks_file, Chunk),
        "policy": AccessPolicy(load_principals(Settings().configs_dir)),
        "golden_path": golden_path,
    }


def _check_split(split: str) -> Literal["dev", "test", "all"]:
    if split not in ("dev", "test", "all"):
        console.print(f"[red]--split must be dev, test or all, got {split!r}[/red]")
        raise typer.Exit(code=1)
    return cast(Literal["dev", "test", "all"], split)


@eval_app.command("run")
def eval_run(
    config: Annotated[Path, typer.Option(help="Experiment YAML, e.g. configs/experiments/x.yaml.")],
    split: Annotated[str, typer.Option(help="dev (tuning), test (reporting) or all.")] = "dev",
    golden: Annotated[Path | None, typer.Option(help="Override the config's golden set.")] = None,
) -> None:
    """Evaluate one retrieval configuration and write experiments/<ts>_<name>_<split>/."""
    import json

    from docseek.eval.runner import load_experiment, run_experiment

    checked = _check_split(split)
    cfg = load_experiment(config)
    inputs = _eval_inputs(golden or cfg.golden, cfg.processed_dir, cfg.chunks_file)
    run_dir = run_experiment(
        cfg,
        split=checked,
        experiments_dir=Settings().experiments_dir,
        **inputs,  # type: ignore[arg-type]
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    overall = metrics["overall"]
    table = Table(title=f"{cfg.name} on {split} ({overall['n_queries']} queries)")
    for col in ("metric", "mean", "95% CI", "n"):
        table.add_column(col)
    for name, v in overall.items():
        if name == "n_queries":
            continue
        if "mean" in v:
            table.add_row(
                name, f"{v['mean']:.4f}", f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]", str(v["n"])
            )
        else:
            table.add_row(name, f"total {v['total']}", f"max {v['max']}", "")
    console.print(table)
    console.print(f"-> {run_dir}")


@eval_app.command("grid")
def eval_grid(
    config: Annotated[Path, typer.Option(help="Base experiment YAML.")],
    param: Annotated[
        list[str], typer.Option(help="Swept key, e.g. retriever.k1=0.9,1.2 (repeatable).")
    ],
    split: Annotated[str, typer.Option(help="Tune on dev; never sweep on test.")] = "dev",
    metric: Annotated[str, typer.Option(help="Metric to rank grid points by.")] = "ndcg@10",
) -> None:
    """Run a cartesian sweep; writes one run per point plus <ts>_<name>_grid_<split>/."""
    import json

    from docseek.eval.grid import parse_param, run_grid
    from docseek.eval.runner import ExperimentConfig, load_experiment, run_experiment

    checked = _check_split(split)
    if checked == "test":
        console.print("[red]grid search on the test split is not allowed[/red]")
        raise typer.Exit(code=1)
    base = load_experiment(config)
    params = dict(parse_param(p) for p in param)
    experiments_dir = Settings().experiments_dir
    loaded: dict[tuple[Path, Path, str], dict[str, object]] = {}

    def run_one(cfg: ExperimentConfig) -> Path:
        # Grid points may change the golden set or the chunk file; load inputs per point.
        key = (cfg.golden, cfg.processed_dir, cfg.chunks_file)
        if key not in loaded:
            loaded[key] = _eval_inputs(*key)
        console.print(f"  {cfg.name}")
        return run_experiment(cfg, split=checked, experiments_dir=experiments_dir, **loaded[key])  # type: ignore[arg-type]

    out = run_grid(base, params, run_one, split=checked, experiments_dir=experiments_dir)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    points = sorted(summary["points"], key=lambda r: -r.get(metric, 0.0))
    table = Table(title=f"grid {base.name} on {split}, by {metric}")
    table.add_column("params")
    table.add_column(metric)
    for r in points:
        table.add_row(json.dumps(r["params"]), f"{r.get(metric, float('nan')):.4f}")
    console.print(table)
    console.print(f"-> {out}")


@eval_app.command("compare")
def eval_compare(
    run_a: Annotated[Path, typer.Argument(help="Baseline run directory (A).")],
    run_b: Annotated[Path, typer.Argument(help="Candidate run directory (B).")],
    metric: Annotated[
        list[str] | None, typer.Option(help="Metrics to compare (default: all shared).")
    ] = None,
    top: Annotated[int, typer.Option(help="Biggest wins/losses to list.")] = 10,
) -> None:
    """Paired comparison of two runs: delta, bootstrap CI, randomization p-value, top swings."""
    from docseek.eval.runner import compare_runs

    try:
        rows, swings = compare_runs(run_a, run_b, metrics=metric)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    table = Table(title=f"B={run_b.name} vs A={run_a.name}")
    for col in ("metric", "n", "A", "B", "delta", "delta 95% CI", "p", "verdict"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r.metric,
            str(r.n),
            f"{r.mean_a:.4f}",
            f"{r.mean_b:.4f}",
            f"{r.delta:+.4f}",
            f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
            f"{r.p_value:.4f}",
            r.verdict,
        )
    console.print(table)
    if swings and rows:
        console.print(f"Biggest losses of B on {rows[0].metric}:")
        for qid, qtype, d in swings[:top]:
            if d < 0:
                console.print(f"  {qid} ({qtype}) {d:+.4f}")
        console.print(f"Biggest wins of B on {rows[0].metric}:")
        for qid, qtype, d in reversed(swings[-top:]):
            if d > 0:
                console.print(f"  {qid} ({qtype}) {d:+.4f}")


@eval_app.command("pool")
def eval_pool(
    runs: Annotated[list[Path], typer.Argument(help="Run directories to pool from.")],
    depth: Annotated[int, typer.Option(help="Top sections per run and query.")] = 10,
    golden: Annotated[Path, typer.Option()] = Path("data/golden/golden_v1.jsonl"),
    out: Annotated[Path, typer.Option()] = Path("data/golden/golden_v2.jsonl"),
    judgments: Annotated[Path, typer.Option()] = Path("data/golden/pool_judgments_v2.jsonl"),
    cost_per_pair: Annotated[
        float, typer.Option(help="Estimate for the dry run (M1 judge average).")
    ] = 0.000148,
    execute: Annotated[bool, typer.Option(help="Call the judge (default: dry run).")] = False,
) -> None:
    """Golden v2 by pooling (§5.3): judge unlabelled pooled pairs, merge into v1 labels."""
    from docseek.core.models import Section
    from docseek.core.runs import environment_info, make_run_dir, write_json
    from docseek.core.tracing import JsonlTraceSink, Tracer, read_traces, summarize_llm_costs
    from docseek.eval.golden import GoldenQuery, load_golden, write_golden
    from docseek.eval.judge import RelevanceJudge
    from docseek.eval.pooling import (
        PoolJudgment,
        build_v2,
        pool_sections,
        read_judgments,
        unlabelled_pairs,
        write_judgments,
    )
    from docseek.eval.synth import parallel_map
    from docseek.ingest.pipeline import read_jsonl
    from docseek.llm.client import build_llm_client

    settings = Settings()
    gold = load_golden(golden)
    pairs = unlabelled_pairs(gold, pool_sections(runs, depth))
    done = read_judgments(judgments)
    todo = [p for p in pairs if (p[0].qid, p[1]) not in done]
    console.print(
        f"{len(pairs)} unlabelled pooled pairs, {len(todo)} not yet judged; "
        f"estimated cost ${len(todo) * cost_per_pair:.3f}"
    )
    if not execute:
        console.print("dry run; pass --execute to call the judge")
        return
    sections = {s.section_id: s for s in read_jsonl(Path("data/processed/sections.jsonl"), Section)}
    run_dir = make_run_dir(settings.experiments_dir, "pool_v2")
    traces = run_dir / "traces.jsonl"
    llm = build_llm_client(settings, Tracer(JsonlTraceSink(traces)))
    judge = RelevanceJudge(llm, load_models_config(settings.configs_dir).llm.judge)

    def one(pair: tuple[GoldenQuery, str, list[str]]) -> PoolJudgment:
        q, sid, pooled = pair
        j = judge.judge(q.text, sections[sid])
        return PoolJudgment(q.qid, sid, j.score, j.reason, j.model, j.rubric_version, pooled)

    new = parallel_map(one, todo, 4)
    merged = {**done, **{(j.qid, j.section_id): j for j in new}}
    ordered = [merged[(q.qid, sid)] for q, sid, _ in pairs]
    write_judgments(judgments, ordered)
    v2 = build_v2(gold, ordered)
    write_golden(out, v2)
    cost = summarize_llm_costs(read_traces(traces) if traces.exists() else [])
    n_added = sum(len(b.relevant) - len(a.relevant) for a, b in zip(gold, v2, strict=True))
    report = {
        "runs": [r.as_posix() for r in runs],
        "depth": depth,
        "pairs": len(pairs),
        "judged_now": len(new),
        "grade_counts": {g: sum(1 for j in ordered if j.score == g) for g in (0, 1, 2)},
        "labels_added": n_added,
        "queries_changed": sum(1 for a, b in zip(gold, v2, strict=True) if a != b),
        "cost": cost.__dict__,
    }
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "env.json", environment_info(command="eval pool", **report))
    console.print(report)
    console.print(f"-> {out}, {judgments}, {run_dir}")


@eval_app.command("synth")
def eval_synth(
    out: Annotated[Path, typer.Option(help="Draft golden set to write.")] = Path(
        "data/golden/golden_v1.draft.jsonl"
    ),
    scale: Annotated[
        float, typer.Option(min=0.01, max=1.0, help="Multiply targets (e.g. 0.1 for a preview).")
    ] = 1.0,
) -> None:
    """Generate synthetic golden queries (draft, unreviewed) from data/processed/."""
    import yaml

    from docseek.core.config import load_ingest_config
    from docseek.core.models import Section
    from docseek.core.runs import environment_info, make_run_dir, write_json
    from docseek.core.tracing import JsonlTraceSink, Tracer, read_traces, summarize_llm_costs
    from docseek.data.gen_internal import load_plan
    from docseek.dense.embedders import FastEmbedEmbedder
    from docseek.eval.golden import write_golden
    from docseek.eval.judge import RelevanceJudge
    from docseek.eval.synth import SynthConfig, SynthContext, synthesize
    from docseek.ingest.pipeline import read_jsonl
    from docseek.llm.client import build_llm_client

    settings = Settings()
    models = load_models_config(settings.configs_dir)
    sections_path = load_ingest_config(settings.configs_dir).output_dir / "sections.jsonl"
    if not sections_path.is_file():
        console.print(f"[red]missing {sections_path}[/red]; run `docseek ingest` first")
        raise typer.Exit(code=1)
    sections = read_jsonl(sections_path, Section)
    if not any(s.source == "internal-kb" for s in sections):
        console.print("[red]no internal KB sections[/red]; run `docseek data gen-internal`")
        raise typer.Exit(code=1)
    with (settings.configs_dir / "synth.yaml").open(encoding="utf-8") as fh:
        cfg = SynthConfig.model_validate(yaml.safe_load(fh))
    if scale < 1.0:
        cfg = cfg.model_copy(
            update={"targets": {k: max(1, round(v * scale)) for k, v in cfg.targets.items()}}
        )
    run_dir = make_run_dir(settings.experiments_dir, "synth")
    traces = run_dir / "traces.jsonl"
    llm = build_llm_client(settings, Tracer(JsonlTraceSink(traces)))
    ctx = SynthContext(
        sections=sections,
        plan=load_plan(settings.configs_dir / "internal_kb_plan.yaml"),
        principals=load_principals(settings.configs_dir),
        llm=llm,
        generator=models.llm.generator,
        judge=RelevanceJudge(llm, models.llm.judge),
        embedder=FastEmbedEmbedder(models.embedding.default),
        cfg=cfg,
    )
    result = synthesize(ctx)
    write_golden(out, result.queries)
    cost = summarize_llm_costs(read_traces(traces) if traces.exists() else [])
    report = {
        **result.report,
        "config": cfg.model_dump(mode="json"),
        "generator": models.llm.generator.model,
        "judge": models.llm.judge.model,
        "judge_rubric_version": ctx.judge.version,
        "embedder": ctx.embedder.name,
        "cost": cost.__dict__,
        "run_dir": run_dir.as_posix(),
    }
    write_json(run_dir / "report.json", report)
    write_json(out.with_suffix(".report.json"), report)
    write_json(run_dir / "env.json", environment_info(command="eval synth", seed=cfg.seed))
    table = Table(title=f"eval synth -> {out}")
    for col in ("qtype", "generated", "dup", "not answerable", "kept rows"):
        table.add_column(col)
    for qtype, row in report["per_qtype"].items():
        table.add_row(
            qtype,
            str(row.get("generated", "")),
            str(row.get("dropped_duplicate", "")),
            str(row.get("dropped_not_answerable", "")),
            str(row.get("kept_rows", "")),
        )
    console.print(table)
    console.print(
        f"{len(result.queries)} queries; LLM calls {cost.llm_calls} ({cost.cached_calls} cached), "
        f"spent ${cost.cost_usd:.4f} -> {run_dir}"
    )
