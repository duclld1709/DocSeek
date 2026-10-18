"""Interactive commands for the human steps of the golden set; registered under ``docseek eval``.

``review`` (accept / edit / drop synthetic queries), ``find-sections`` (look up ids for hand-
written queries), ``build-golden`` (merge everything into golden_v1.jsonl), ``label`` (blind
0/1/2 labels for calibration) and ``calibrate`` (judge vs human kappa and bias checks).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from docseek.core.acl import AccessPolicy
from docseek.core.config import Settings, load_ingest_config, load_models_config, load_principals
from docseek.core.models import Section
from docseek.eval.golden import GoldenValidationError, load_golden, write_golden
from docseek.eval.review import (
    CalibrationPair,
    HumanLabel,
    ReviewDecision,
    append_jsonl,
    build_golden,
    calibration_report,
    dump_jsonl,
    find_sections,
    latest_decisions,
    load_human_queries,
    read_jsonl_models,
    review_order,
    sample_calibration_pairs,
)
from docseek.ingest.pipeline import read_jsonl

console = Console()
GOLDEN_DIR = Path("data/golden")


def _sections() -> list[Section]:
    path = load_ingest_config(Settings().configs_dir).output_dir / "sections.jsonl"
    if not path.is_file():
        console.print(f"[red]missing {path}[/red]; run `docseek ingest` first")
        raise typer.Exit(code=1)
    return read_jsonl(path, Section)


def _policy() -> AccessPolicy:
    return AccessPolicy(load_principals(Settings().configs_dir))


def _show(section: Section, max_chars: int = 1500) -> None:
    text = section.text if len(section.text) <= max_chars else section.text[:max_chars] + " [...]"
    path = " > ".join(section.section_path)
    title = f"{section.section_id}  ({path}; {section.status}; acl {section.acl})"
    console.print(Panel(text, title=title, title_align="left"))


def review(
    draft: Annotated[Path, typer.Option()] = GOLDEN_DIR / "golden_v1.draft.jsonl",
    decisions: Annotated[Path, typer.Option()] = GOLDEN_DIR / "golden_v1.review.jsonl",
    limit: Annotated[int, typer.Option(help="Stop after this many decisions.")] = 1000,
) -> None:
    """Review draft queries one by one: accept, edit, drop or skip (progress is saved)."""
    sections = {s.section_id: s for s in _sections()}
    done = latest_decisions(read_jsonl_models(decisions, ReviewDecision))
    todo = [q for q in review_order(load_golden(draft)) if q.qid not in done]
    console.print(f"{len(done)} reviewed so far, {len(todo)} to go. Decisions -> {decisions}")
    made = 0
    for q in todo:
        if made >= limit:
            break
        console.rule(f"{q.qid}  {q.qtype}  principal={q.principal}  style={q.style}")
        console.print(f"[bold]{q.text}[/bold]")
        if q.reference_answer:
            console.print(f"[dim]reference: {q.reference_answer}[/dim]")
        if q.notes:
            console.print(f"[dim]notes: {q.notes}[/dim]")
        for sid, grade in q.relevant.items():
            console.print(f"relevant (grade {grade}):")
            _show(sections[sid])
        for sid in q.stale:
            console.print(f"[yellow]stale:[/yellow] {sid}")
        for sid in q.hard_negatives:
            console.print(f"[dim]hard negative: {sid} - {sections[sid].title}[/dim]")
        choice = typer.prompt("[a]ccept [e]dit [d]rop [s]kip [q]uit", default="a")
        choice = choice.strip().lower()
        if choice.startswith("q"):
            break
        if choice.startswith("s"):
            continue
        if choice.startswith("e"):
            text = typer.prompt("new query text", default=q.text)
            note = typer.prompt("note (optional)", default="", show_default=False) or None
            decision = ReviewDecision(qid=q.qid, action="edit", text=text, note=note)
        elif choice.startswith("d"):
            note = typer.prompt("why drop? (optional)", default="", show_default=False) or None
            decision = ReviewDecision(qid=q.qid, action="drop", note=note)
        else:
            decision = ReviewDecision(qid=q.qid, action="accept")
        append_jsonl(decisions, decision)
        made += 1
    total = len(latest_decisions(read_jsonl_models(decisions, ReviewDecision)))
    console.print(f"{total} queries reviewed in total.")


def find(
    needle: Annotated[str, typer.Argument(help="Text to look for in titles, paths and text.")],
    limit: Annotated[int, typer.Option()] = 20,
) -> None:
    """Find section ids to use as relevant sections in hand-written queries."""
    for s in find_sections(_sections(), needle, limit):
        console.print(f"{s.section_id}  [dim]{' > '.join(s.section_path)}  acl={s.acl}[/dim]")


def build(
    draft: Annotated[Path, typer.Option()] = GOLDEN_DIR / "golden_v1.draft.jsonl",
    decisions: Annotated[Path, typer.Option()] = GOLDEN_DIR / "golden_v1.review.jsonl",
    human: Annotated[Path, typer.Option()] = GOLDEN_DIR / "human_queries_v1.yaml",
    out: Annotated[Path, typer.Option()] = GOLDEN_DIR / "golden_v1.jsonl",
) -> None:
    """Merge draft + review decisions + hand-written queries into the golden set."""
    try:
        golden = build_golden(
            load_golden(draft),
            latest_decisions(read_jsonl_models(decisions, ReviewDecision)),
            load_human_queries(human),
            sections=_sections(),
            policy=_policy(),
        )
    except (GoldenValidationError, ValueError) as exc:
        console.print(f"[red]golden set invalid:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    write_golden(out, golden)
    console.print(
        f"{len(golden)} queries -> {out}: reviewed {sum(q.reviewed for q in golden)}, "
        f"human {sum(q.origin == 'human' for q in golden)}, "
        f"splits {dict(Counter(q.split for q in golden))}"
    )
    console.print(dict(Counter(str(q.qtype) for q in golden)))


def label(
    golden: Annotated[Path, typer.Option()] = GOLDEN_DIR / "golden_v1.jsonl",
    pairs: Annotated[Path, typer.Option()] = GOLDEN_DIR / "calibration_pairs_v1.jsonl",
    labels: Annotated[Path, typer.Option()] = GOLDEN_DIR / "human_labels_v1.jsonl",
    n: Annotated[int, typer.Option(help="Pairs to sample (first run only).")] = 100,
) -> None:
    """Label (query, section) pairs 0/1/2 by hand, blind to the judge (resumable)."""
    sections = _sections()
    by_id = {s.section_id: s for s in sections}
    if not pairs.is_file():
        sampled = sample_calibration_pairs(load_golden(golden), sections, _policy(), n=n)
        dump_jsonl(pairs, sampled)
        console.print(f"sampled {len(sampled)} pairs -> {pairs}")
    all_pairs = read_jsonl_models(pairs, CalibrationPair)
    done = {lab.pair_id for lab in read_jsonl_models(labels, HumanLabel)}
    todo = [p for p in all_pairs if p.pair_id not in done]
    console.print(
        "Scale: 2 = answers directly, 1 = related/partial, 0 = not relevant. "
        f"{len(done)} labelled, {len(todo)} to go."
    )
    for p in todo:
        console.rule(p.pair_id)
        console.print(f"[bold]QUERY:[/bold] {p.query}")
        _show(by_id[p.section_id], max_chars=2500)
        answer = typer.prompt("score 0/1/2 (q to quit)").strip().lower()
        if answer.startswith("q"):
            break
        if answer not in ("0", "1", "2"):
            console.print("[yellow]skipped: answer 0, 1 or 2[/yellow]")
            continue
        note = typer.prompt("note (optional)", default="", show_default=False) or None
        append_jsonl(labels, HumanLabel(pair_id=p.pair_id, score=int(answer), note=note))
    console.print(f"{len(read_jsonl_models(labels, HumanLabel))} labels in {labels}")


def calibrate(
    pairs: Annotated[Path, typer.Option()] = GOLDEN_DIR / "calibration_pairs_v1.jsonl",
    labels: Annotated[Path, typer.Option()] = GOLDEN_DIR / "human_labels_v1.jsonl",
) -> None:
    """Judge the calibration pairs (both orders) and report kappa against human labels."""
    from docseek.core.runs import environment_info, make_run_dir, write_json
    from docseek.core.tracing import JsonlTraceSink, Tracer, read_traces, summarize_llm_costs
    from docseek.eval.judge import RelevanceJudge
    from docseek.eval.synth import parallel_map
    from docseek.llm.client import build_llm_client

    settings = Settings()
    human = {lab.pair_id: lab for lab in read_jsonl_models(labels, HumanLabel)}
    todo = [p for p in read_jsonl_models(pairs, CalibrationPair) if p.pair_id in human]
    if len(todo) < 2:
        console.print(f"[red]need human labels in {labels}[/red]; run `docseek eval label`")
        raise typer.Exit(code=1)
    sections = {s.section_id: s for s in _sections()}
    run_dir = make_run_dir(settings.experiments_dir, "calibrate")
    traces = run_dir / "traces.jsonl"
    llm = build_llm_client(settings, Tracer(JsonlTraceSink(traces)))
    judge = RelevanceJudge(llm, load_models_config(settings.configs_dir).llm.judge)
    qf = parallel_map(lambda p: judge.judge(p.query, sections[p.section_id]), todo, 4)
    sf = parallel_map(
        lambda p: judge.judge(p.query, sections[p.section_id], order="section_first"), todo, 4
    )
    report = calibration_report(
        todo,
        human,
        {p.pair_id: j for p, j in zip(todo, qf, strict=True)},
        {p.pair_id: j for p, j in zip(todo, sf, strict=True)},
        sections,
    )
    report["cost"] = summarize_llm_costs(read_traces(traces) if traces.exists() else []).__dict__
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "env.json", environment_info(command="eval calibrate"))
    console.print(
        f"n={report['n_pairs']}  kappa linear={report['kappa_linear']:.3f}  "
        f"quadratic={report['kappa_quadratic']:.3f}  exact={report['exact_agreement']:.3f}  "
        f"rubric {report['rubric_version']}"
    )
    console.print(
        f"confusion (rows human 0/1/2, cols judge): {report['confusion_human_rows_judge_cols']}"
    )
    if "position_bias" in report:
        console.print(f"position bias: {report['position_bias']}")
    console.print(f"length bias (Spearman): {report['length_bias_spearman']}")
    console.print(f"{len(report['disagreements'])} disagreements -> {run_dir / 'report.json'}")


def register(app: typer.Typer) -> None:
    app.command("review")(review)
    app.command("find-sections")(find)
    app.command("build-golden")(build)
    app.command("label")(label)
    app.command("calibrate")(calibrate)
