# DocSeek

**Measurable agentic retrieval over technical documentation.** BM25 written from scratch,
dense and hybrid search, reranking, query understanding, a knowledge graph and filesystem
tools for an agent, all evaluated against a golden set with statistical tests.

[![CI](https://github.com/duclld1709/DocSeek/actions/workflows/ci.yml/badge.svg)](https://github.com/duclld1709/DocSeek/actions/workflows/ci.yml)

> **Status:** M1 (ingestion + evaluation harness) in progress. The retrieval pipeline and the benchmark are built milestone
> by milestone; see [Roadmap](#roadmap).

## Results at a glance

No results yet. This table is filled only from runs recorded under `experiments/` (from
M2 on): BM25 → dense → hybrid → +rerank → +router → agentic, with nDCG@10, Recall@10,
p95 latency and $ per 1k queries.

## Demo

Coming in M8.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the diagram and component list.

- **Ingestion** with source tracing: markdown sections with stable ids, resolved code
  includes, OCR for scans, images.
- **Retrieval**: own BM25, Qdrant dense vectors, fusion (RRF/DBSF), cross-encoder rerank.
- **Query understanding and routing** by query type, metadata filters, freshness.
- **Knowledge graph** (Neo4j) for multi-hop questions and **sandboxed filesystem tools**.
- **Agent loop** with tool calling through LiteLLM, citations and abstention.
- **Evaluation harness** with section-level relevance, calibrated LLM judges, bootstrap CIs
  and a CI regression gate.

## What makes it production-grade

Evaluation methodology, ACL enforced inside retrievers (never by prompt), freshness-aware
ranking, tracing, cost tracking with a hard budget, and a CI regression gate. These are
built across milestones; [docs/overview.md](docs/overview.md) explains why.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
git clone https://github.com/duclld1709/DocSeek.git && cd DocSeek
cp .env.example .env          # add API keys later (needed from M1)
uv sync
docker compose up -d          # Qdrant + Neo4j
uv run docseek doctor         # checks configs, services, keys (set / not set), Tesseract
```

Fetch the pinned FastAPI docs corpus (writes `data/raw/MANIFEST.json`):

```bash
uv run docseek data fetch
```

Build sections and chunks (Corpus B, the internal KB, is committed in `data/internal_kb/`),
then check the evaluation harness with the sanity retrievers:

```bash
uv run docseek ingest
uv run docseek eval run --config configs/experiments/oracle.yaml --split dev
uv run docseek eval run --config configs/experiments/random.yaml --split dev
```

LLM-backed steps (`data gen-internal`, `eval synth`, `eval calibrate`) go through a response
cache, a per-run budget and a project-wide spend ledger (ADR 0003); `uv run docseek eval
costs` shows what has been spent.

Development:

```bash
uv run pre-commit install
uv run pytest                      # unit tests (no network, no keys)
uv run pytest -m integration       # needs docker compose up
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

## Project structure

```
configs/            corpus pin, principals -> ACL groups, model names
src/docseek/
  core/             models, stable ids, interfaces, settings, logging
  data/             corpus fetch
  testing/          deterministic fakes for tests
  doctor.py cli.py  environment checks, Typer CLI
tests/
  unit/ integration/ fixtures/mini_corpus/
docs/               overview, architecture, ADRs, milestones, results
```

## Key engineering decisions

- [ADR 0001: own BM25 implementation](docs/adr/0001-own-bm25.md) (proposed)

## Benchmark report

Coming in M8 (`docs/benchmark_report.md`).

## Roadmap

| Milestone | Scope |
|---|---|
| M0 | Foundation: repo, CI, configs, fakes, `doctor`, corpus fetch |
| M1 | Ingestion v1 + evaluation harness |
| M2 | Lexical search: TF-IDF → BM25 |
| M3 | Dense search with Qdrant |
| M4 | Hybrid retrieval + reranking |
| M5 | Query understanding, metadata, chunking |
| M6 | Knowledge graph + filesystem search |
| M7 | OCR + multimodal |
| M8 | Agent, API, UI, benchmark, release |

## Acknowledgements

- [FastAPI documentation](https://github.com/fastapi/fastapi) (MIT License), used as the
  main corpus, pinned to commit `b23d78f2ce4e7b3ea48a59f51ca249690d9fd02c`.
- Syllabus reference: the "Production-Grade Agentic RAG" course.
