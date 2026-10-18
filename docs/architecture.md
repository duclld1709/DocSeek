# Architecture (draft)

> Draft written in M0. Sections are filled in as milestones land; the "Status" column says
> what exists today.

## System overview

```mermaid
flowchart TB
    subgraph Ingestion
        F[fetch / generate] --> P[parse: markdown, PDF + OCR, images]
        P --> N[normalize + metadata] --> C[chunk]
        C --> PR[markdown projection]
    end
    C --> BM25[(BM25 index, own)]
    C --> QD[(Qdrant: dense / sparse / image vectors + payload)]
    C --> KG[(Neo4j knowledge graph)]
    PR --> FS[(Filesystem view)]

    Q[Query + principal] --> QU[Query understanding: classify, rewrite, decompose, filters]
    QU --> R[Router]
    R --> RET[Retrievers] --> FU[Fusion RRF / DBSF / weighted] --> RR[Reranker]
    BM25 & QD & KG --> RET
    RR --> A[Agent loop: tool calling via LiteLLM]
    FS --> A
    A --> ANS[Answer + citations + trace]

    ANS --> API[FastAPI] & UI[Streamlit] & CLI[Typer CLI]
    EV[Eval harness: golden set, metrics, judges, stats, gate] -.-> RET & A
```

## Components

| Package | Responsibility | Status |
|---|---|---|
| `core/` | Pydantic models, stable ids, `Protocol` interfaces, settings, structured logging | M0 |
| `testing/` | Deterministic fakes (`HashEmbedder`, `IdentityReranker`, `ScriptedLLM`) | M0 |
| `data/` | `data fetch` (pinned sparse clone + manifest) | M0 |
| `doctor.py`, `cli.py` | Environment checks and CLI entry point | M0 |
| `ingest/` | Markdown parser, chunking, metadata | M1 |
| `llm/` | `CachedLLMClient`, `BudgetGuard`, cost accounting | M1 |
| `eval/` | Golden set, synthetic queries, metrics, judges, statistics, runner | M1 |
| `lexical/` | Tokenizer, inverted index, TF-IDF, BM25 | M2 |
| `dense/` | Embedders, Qdrant store | M3 |
| `hybrid/`, `rerank/` | Fusion, normalisation, cross-encoder | M4 |
| `query/` | Classifier, rewrite, decomposition, filters, router | M5 |
| `graph/`, `fs/` | Neo4j graph retrieval, sandboxed filesystem tools | M6 |
| `multimodal/` | OCR, CLIP, captions | M7 |
| `agent/`, `api/`, `ui/` | Agent loop, HTTP API, demo UI | M8 |

## Data units and identifiers

```
Document  doc_id     = "<source>:<relative_path>"        fastapi-docs:tutorial/dependencies/index.md
Section   section_id = "<doc_id>#<anchor>"               ...index.md#first-steps
Chunk     chunk_id   = "<section_id>::<strategy>::<n>"   ...index.md#first-steps::section::0
```

Relevance is labelled per **section**; chunk hits are mapped back to sections (best rank
per section, duplicates dropped) during evaluation, so chunking strategies can be compared
on the same golden set. Helpers live in `docseek.core.ids`.

## Cross-cutting rules

- **ACL is enforced in retrievers and tools**, from the server-side principal
  (`configs/principals.yaml`). `MetadataFilter` has no ACL field, so a query or an
  LLM-extracted filter cannot widen access. Unknown principals are rejected.
- **Every LLM call** goes through `CachedLLMClient` + `BudgetGuard` (from M1). Model names
  live only in `configs/models.yaml`; the judge must come from a different provider family
  than the generator, which the config loader enforces.
- **Determinism:** fixed seeds; tests use fakes only (no network, no keys).
- **Reproducible data:** the corpus is pinned to a commit and recorded in
  `data/raw/MANIFEST.json` with per-file sha256 (line endings preserved byte-for-byte).
- **Observability:** structlog JSON logs on stderr with secret masking; `trace_id` bound
  through context variables.

## Local services

`docker-compose.yml` runs Qdrant `v1.19.2` and Neo4j `5.26-community`, bound to
`127.0.0.1`, with healthchecks and named volumes. Clients default to `127.0.0.1` rather
than `localhost`, because on Windows `localhost` tries IPv6 first and adds about 2 s to
every new connection.
