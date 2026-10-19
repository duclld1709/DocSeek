"""OCR benchmark on Corpus C: CER, WER, s/page, $/page and table cell accuracy per engine.

    uv run python scripts/m7_ocr_bench.py [--engines tesseract rapidocr vision]

Writes experiments/<ts>_m7_ocr_bench/{report.json, per_page.jsonl, traces.jsonl, outputs/}.
Vision LLM cost comes from its trace (Gemini free tier: list price, not charged).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from docseek.core.config import Settings, load_models_config
from docseek.core.runs import environment_info, make_run_dir, write_json
from docseek.core.tracing import JsonlTraceSink, Tracer, read_traces, trace_scope
from docseek.eval.ocr_metrics import cer, table_cell_accuracy, wer
from docseek.ingest.ocr import OcrEngine, RapidOcrEngine, TesseractEngine, VisionLlmEngine
from docseek.ingest.pdf_parser import page_png
from docseek.llm.client import build_llm_client

SCANS = Path("data/scans")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["tesseract", "rapidocr", "vision"])
    args = ap.parse_args()
    manifest = json.loads((SCANS / "ground_truth" / "manifest.json").read_text(encoding="utf-8"))
    run = make_run_dir(Path("experiments"), "m7_ocr_bench")
    tracer = Tracer(JsonlTraceSink(run / "traces.jsonl"))
    engines: dict[str, OcrEngine] = {}
    for name in args.engines:
        if name == "tesseract":
            engines[name] = TesseractEngine()
        elif name == "rapidocr":
            engines[name] = RapidOcrEngine()
        elif name == "vision":
            engines[name] = VisionLlmEngine(
                build_llm_client(Settings(), tracer), load_models_config().llm.vision
            )
    rows: list[dict[str, Any]] = []
    texts: dict[tuple[str, str], list[str]] = {}
    n_pages = sum(s["pages"] for s in manifest["scans"])
    for key, engine in engines.items():
        done, started = 0, time.perf_counter()
        out_dir = run / "outputs" / key
        out_dir.mkdir(parents=True)
        for scan in manifest["scans"]:
            pdf = SCANS / "pdf" / f"{scan['name']}.pdf"
            for p in range(1, scan["pages"] + 1):
                gt = (SCANS / "ground_truth" / f"{scan['name']}_p{p}.txt").read_text(
                    encoding="utf-8"
                )
                png = page_png(pdf, p)
                with trace_scope():
                    t = time.perf_counter()
                    text = engine.ocr(png)
                    secs = time.perf_counter() - t
                (out_dir / f"{scan['name']}_p{p}.txt").write_text(
                    text, encoding="utf-8", newline="\n"
                )
                texts.setdefault((key, scan["name"]), []).append(text)
                rows.append(
                    {
                        "engine": key,
                        "scan": scan["name"],
                        "page": p,
                        "two_columns": scan["two_columns"],
                        "cer": cer(gt, text),
                        "wer": wer(gt, text),
                        "seconds": secs,
                        # Empty output = the engine returned nothing (for Gemini: the
                        # recitation content filter on public docs text).
                        "empty": not text.strip(),
                    }
                )
                done += 1
                eta = (time.perf_counter() - started) / done * (n_pages - done)
                print(
                    f"[{key}] {done}/{n_pages} {scan['name']} p{p} "
                    f"cer={rows[-1]['cer']:.3f} {secs:.1f}s eta={eta:.0f}s",
                    flush=True,
                )
    with (run / "per_page.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    events = read_traces(run / "traces.jsonl") if (run / "traces.jsonl").exists() else []
    vision_cost = sum(e.cost_usd for e in events if e.kind == "llm_call")
    vision_list_price = sum(
        e.output.get("original_cost_usd", e.cost_usd) for e in events if e.kind == "llm_call"
    )
    report: dict[str, Any] = {"n_pages": sum(s["pages"] for s in manifest["scans"]), "engines": {}}
    for key in engines:
        rs = [r for r in rows if r["engine"] == key]

        def mean(field: str, subset: list[dict[str, Any]]) -> float:
            return statistics.mean(r[field] for r in subset) if subset else float("nan")

        one, two = [r for r in rs if not r["two_columns"]], [r for r in rs if r["two_columns"]]
        answered = [r for r in rs if not r["empty"]]
        correct = total = 0
        per_table: dict[str, float] = {}
        for scan in manifest["scans"]:
            tpath = SCANS / "ground_truth" / f"{scan['name']}.tables.json"
            if tpath.exists():
                score = table_cell_accuracy(
                    json.loads(tpath.read_text(encoding="utf-8")),
                    "\n".join(texts[(key, scan["name"])]),
                )
                correct += score.correct
                total += score.total
                per_table[scan["name"]] = score.accuracy
        n = len(rs)
        cost = vision_list_price if key == "vision" else 0.0
        report["engines"][key] = {
            "cer": mean("cer", rs),
            "wer": mean("wer", rs),
            "cer_one_column": mean("cer", one),
            "cer_two_columns": mean("cer", two),
            "wer_two_columns": mean("wer", two),
            "empty_pages": len(rs) - len(answered),
            "cer_non_empty": mean("cer", answered),
            "wer_non_empty": mean("wer", answered),
            "seconds_per_page": mean("seconds", rs),
            "usd_per_page_list_price": cost / n if n else 0.0,
            "usd_charged_total": vision_cost if key == "vision" else 0.0,
            "table_cells": {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else None,
            },
            "table_accuracy_by_doc": per_table,
        }
    write_json(run / "report.json", report)
    write_json(
        run / "env.json", environment_info(command="scripts/m7_ocr_bench.py", engines=args.engines)
    )
    print(json.dumps(report["engines"], indent=1))
    print(f"-> {run}")


if __name__ == "__main__":
    main()
