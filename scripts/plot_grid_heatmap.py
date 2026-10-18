"""Heatmap of a 2-parameter grid sweep (``docseek eval grid``) as a PNG.

    uv run python scripts/plot_grid_heatmap.py GRID_DIR --x retriever.b --y retriever.k1 \
        --metric ndcg@10 --out docs/results/img/m2_bm25_grid.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_dir", type=Path)
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    s = json.loads((args.grid_dir / "summary.json").read_text(encoding="utf-8"))
    xs = sorted({p["params"][args.x] for p in s["points"]})
    ys = sorted({p["params"][args.y] for p in s["points"]})
    grid = np.full((len(ys), len(xs)), np.nan)
    for p in s["points"]:
        grid[ys.index(p["params"][args.y]), xs.index(p["params"][args.x])] = p[args.metric]

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=150)
    im = ax.imshow(grid, cmap="Blues", origin="lower", aspect="auto")
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    for (i, j), v in np.ndenumerate(grid):
        dark = (v - lo) / (hi - lo + 1e-12) > 0.6
        weight = "bold" if (i, j) == best else "normal"
        ax.text(
            j,
            i,
            f"{v:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            weight=weight,
            color="white" if dark else "#1f2328",
        )
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_yticks(range(len(ys)), [str(y) for y in ys])
    ax.set_xlabel(args.x.split(".")[-1])
    ax.set_ylabel(args.y.split(".")[-1])
    ax.set_title(args.title or f"{s['base']}: {args.metric} on {s['split']}", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.85).outline.set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
