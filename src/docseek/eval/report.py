"""Numbers and figures for docs/results: golden-set statistics, the lexical-overlap chart and
LLM cost per run (summed from every ``experiments/*/traces.jsonl``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from docseek.core.tracing import read_traces
from docseek.eval.golden import GoldenQuery

STYLES = ("human", "machine", "paraphrase")
N_BINS = 10


def golden_stats(queries: Sequence[GoldenQuery]) -> dict[str, Any]:
    answerable = [q for q in queries if q.relevant]
    overlaps = sorted(q.lexical_overlap for q in queries if q.lexical_overlap is not None)
    by_style: defaultdict[str, list[float]] = defaultdict(list)
    for q in queries:
        if q.lexical_overlap is not None:
            by_style[str(q.style or "none")].append(q.lexical_overlap)
    return {
        "n_queries": len(queries),
        "reviewed": sum(q.reviewed for q in queries),
        "by_origin": dict(sorted(Counter(q.origin for q in queries).items())),
        "by_split": dict(sorted(Counter(q.split for q in queries).items())),
        "by_qtype": dict(sorted(Counter(str(q.qtype) for q in queries).items())),
        "by_principal": dict(sorted(Counter(q.principal for q in queries).items())),
        "by_style": dict(sorted(Counter(str(q.style) for q in queries).items())),
        "relevant_per_query": dict(sorted(Counter(len(q.relevant) for q in answerable).items())),
        "with_hard_negatives": sum(bool(q.hard_negatives) for q in queries),
        "hard_negatives_total": sum(len(q.hard_negatives) for q in queries),
        "relevant_by_source": dict(
            sorted(Counter(s.split(":", 1)[0] for q in queries for s in q.relevant).items())
        ),
        "lexical_overlap_median": overlaps[len(overlaps) // 2] if overlaps else None,
        "lexical_overlap_mean_by_style": {
            s: round(sum(v) / len(v), 3) for s, v in sorted(by_style.items())
        },
    }


def overlap_bins(queries: Iterable[GoldenQuery]) -> dict[str, list[int]]:
    """Counts per style in N_BINS equal-width bins of lexical overlap over [0, 1]."""
    out = {s: [0] * N_BINS for s in STYLES}
    for q in queries:
        if q.lexical_overlap is None or q.style not in out:
            continue
        out[q.style][min(N_BINS - 1, int(q.lexical_overlap * N_BINS))] += 1
    return out


def overlap_svg(queries: Sequence[GoldenQuery], title: str) -> str:
    """Small multiples (one panel per query style) of lexical overlap; one hue, thin bars,
    shared y scale, light and dark themes via prefers-color-scheme."""
    bins = overlap_bins(queries)
    ymax = max(1, *(max(v) for v in bins.values()))
    panel_w, panel_h, gap = 260, 150, 24
    left, top, bottom = 34, 52, 40
    width = left + 3 * panel_w + 2 * gap + 12
    height = top + panel_h + bottom
    bar_w = (panel_w - (N_BINS - 1) * 2) / N_BINS  # 2 px surface gap between bars
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        "<style>"
        ":root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--bar:#2a78d6}"
        "@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#f0efec;"
        "--ink2:#c3c2b7;--grid:#3a3936;--bar:#3987e5}}"
        "text{font-family:system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--ink2);"
        "font-size:11px}.t{fill:var(--ink);font-size:14px;font-weight:600}"
        ".p{fill:var(--ink);font-size:12px;font-weight:600}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="var(--surface)"/>',
        f'<text class="t" x="{left}" y="20">{escape(title)}</text>',
    ]
    for i, style in enumerate(STYLES):
        x0 = left + i * (panel_w + gap)
        counts = bins[style]
        n = sum(counts)
        parts.append(f'<text class="p" x="{x0}" y="{top - 12}">{style} (n={n})</text>')
        for tick in sorted({0, ymax // 2, ymax}):
            y = top + panel_h - tick / ymax * panel_h
            parts.append(
                f'<line x1="{x0}" x2="{x0 + panel_w}" y1="{y:.1f}" y2="{y:.1f}" '
                'stroke="var(--grid)" stroke-width="1"/>'
            )
            if i == 0:
                parts.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')
        for b, c in enumerate(counts):
            if c == 0:
                continue
            h = c / ymax * panel_h
            x = x0 + b * (bar_w + 2)
            y = top + panel_h - h
            r = min(4.0, h / 2, bar_w / 2)
            # Rounded top only: the data end, anchored to the baseline.
            parts.append(
                f'<path d="M{x:.1f},{top + panel_h} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},'
                f"{y:.1f} H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{y:.1f} {x + bar_w:.1f},"
                f'{y + r:.1f} V{top + panel_h} Z" fill="var(--bar)">'
                f"<title>{style}, overlap {b / N_BINS:.1f}-{(b + 1) / N_BINS:.1f}: {c}</title>"
                "</path>"
            )
        for frac in (0.0, 0.5, 1.0):
            x = x0 + frac * panel_w
            parts.append(
                f'<text x="{x:.1f}" y="{top + panel_h + 16}" text-anchor="middle">{frac:g}</text>'
            )
    parts.append(
        f'<text x="{left + (width - left) / 2:.0f}" y="{height - 6}" text-anchor="middle">'
        "share of query content words that also appear in the relevant section(s)</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def run_costs(experiments_dir: Path) -> list[dict[str, Any]]:
    """Per run directory: paid vs cached LLM calls, tokens and cost by provider."""
    rows: list[dict[str, Any]] = []
    for traces in sorted(experiments_dir.glob("*/traces.jsonl")):
        events = [e for e in read_traces(traces) if e.kind == "llm_call"]
        if not events:
            continue
        by_provider: defaultdict[str, dict[str, float]] = defaultdict(
            lambda: {
                "paid_calls": 0,
                "cached_calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
            }
        )
        for e in events:
            p = by_provider[str(e.input.get("model", "?")).split("/", 1)[0]]
            if e.output.get("cached"):
                p["cached_calls"] += 1
                continue
            p["paid_calls"] += 1
            p["tokens_in"] += e.tokens_in
            p["tokens_out"] += e.tokens_out
            p["cost_usd"] += e.cost_usd
        rows.append({"run": traces.parent.name, "providers": dict(by_provider)})
    return rows
