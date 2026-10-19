"""Bar chart of OCR CER per engine (all pages, two-column pages, non-empty pages).

uv run python scripts/plot_ocr.py <local_bench> <vision_bench> docs/results/M7_ocr_cer.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    local, vision, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    engines = json.loads((local / "report.json").read_text(encoding="utf-8"))["engines"]
    engines |= json.loads((vision / "report.json").read_text(encoding="utf-8"))["engines"]
    names = ["tesseract", "rapidocr", "vision"]
    series = [
        ("all pages", "cer", "#2a6fdb"),
        ("two-column pages", "cer_two_columns", "#e08a2b"),
        ("non-empty output", "cer_non_empty", "#3a9e6f"),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.26
    for i, (label, key, color) in enumerate(series):
        xs = [n + (i - 1) * width for n in range(len(names))]
        vals = [engines[e].get(key, engines[e]["cer"]) for e in names]
        bars = ax.bar(xs, vals, width - 0.03, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(range(len(names)), ["Tesseract", "RapidOCR (en)", "Vision LLM"])
    ax.set_ylabel("CER (lower is better)")
    ax.set_title("OCR on 46 degraded scan pages")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
