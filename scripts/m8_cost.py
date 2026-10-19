"""$/1k queries per answer mode and one-time ingest/eval costs.

    uv run python scripts/m8_cost.py experiments/<ts>_m8_answers

- Per-query tokens come from the answer run's report (measured).
- ``usd_per_1k_list``: the answer model's LiteLLM list price (Gemma on the Gemini free tier
  is priced 0). ``usd_per_1k_if_paid``: the same tokens at a paid model's list price
  (default ``gemini/gemini-3.1-flash-lite``), an estimate for a deployment without free tier.
- Retrieval-only configs make no LLM call: $0 per 1k queries.
- One-time costs: the spend ledger grouped by provider and call name (all runs so far).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import litellm

from docseek.core.config import Settings
from docseek.core.runs import make_run_dir, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--paid-model", default="gemini/gemini-3.1-flash-lite")
    args = ap.parse_args()
    rep = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    paid = litellm.model_cost[args.paid_model]
    p_in, p_out = paid["input_cost_per_token"], paid["output_cost_per_token"]
    out: dict[str, Any] = {
        "answer_run": args.run.name,
        "answer_model": rep["model"],
        "paid_model": args.paid_model,
        "modes": {},
    }
    n = rep["n_queries"]
    for mode, m in rep["modes"].items():
        tin, tout = m["tokens_in"] / n, m["tokens_out"] / n
        out["modes"][mode] = {
            "tokens_in_per_query": tin,
            "tokens_out_per_query": tout,
            "llm_calls_per_query": m["llm_calls"] / n,
            "usd_per_1k_list": m["llm_cost_usd_list"] / n * 1000,
            "usd_per_1k_if_paid": (tin * p_in + tout * p_out) * 1000,
        }
    con = sqlite3.connect(Settings().llm_cache_path)
    rows = con.execute(
        "SELECT provider, name, COUNT(*), SUM(cost_usd) FROM spend GROUP BY provider, name"
    ).fetchall()
    out["ledger"] = [{"provider": p, "name": nm, "calls": c, "usd": u} for p, nm, c, u in rows]
    out["ledger_total_by_provider"] = {
        p: sum(r["usd"] for r in out["ledger"] if r["provider"] == p)
        for p in {r["provider"] for r in out["ledger"]}
    }
    dest = make_run_dir(Path("experiments"), "m8_cost")
    write_json(dest / "report.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "ledger"}, indent=1))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
