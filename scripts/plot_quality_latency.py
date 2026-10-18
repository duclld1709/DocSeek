"""Quality vs latency scatter (log-x) from explicit (label, latency_ms, ndcg) points.

    uv run python scripts/plot_quality_latency.py --out docs/results/img/m4_quality_latency.png \
        --point "bm25f=2.4,0.652" --point "rerank@20=1604,0.755" ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", action="append", required=True, help="label=latency_ms,ndcg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="nDCG@10 (dev) vs p50 latency per query")
    args = ap.parse_args()
    pts = []
    for p in args.point:
        label, vals = p.split("=", 1)
        lat, q = (float(v) for v in vals.split(","))
        pts.append((label, lat, q))
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=150)
    ax.plot([p[1] for p in pts], [p[2] for p in pts], "o", color="#2a6fdb", markersize=7)
    for label, lat, q in pts:
        ax.annotate(
            label, (lat, q), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#1f2328"
        )
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency per query, ms (log scale)")
    ax.set_ylabel("nDCG@10")
    ax.set_title(args.title, fontsize=10)
    ax.grid(True, which="major", color="#d0d7de", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
