# Changelog

## v1.0.0 — 2026-10-19

First complete release. Milestone tags `m0-setup` … `m7-multimodal` mark the steps below;
per-milestone results are in `docs/results/`.

- **M8 Agent, API, UI, benchmark**: own tool-calling agent (max 6 steps, timeout, budget,
  principal bound by the server, citation post-check, abstention, one-shot fallback); FastAPI
  `/search` `/ask` `/sections/{id}` `/health` `/traces/{id}`; Streamlit Compare / Answer /
  Trace / Eval; `docseek bench`; DeepEval through our cached, budgeted client; Docker image
  and compose `api`/`ui` with a quick demo; benchmark report, interview notes, ADR 0007.
- **M7 OCR + multimodal** (`m7-multimodal`): synthetic degraded scans with exact ground
  truth; Tesseract vs RapidOCR vs vision LLM (CER/WER, tables); scan page chunks; image
  captions + OCR + CLIP; scan/image queries in the overall eval.
- **M6 Graph + FS** (`m6-graph-fs`): Neo4j graph (AST, release notes, LLM concepts), fixed
  Cypher templates with ACL, markdown projection and sandboxed `ls/find/grep/cat`.
- **M5 Query understanding** (`m5-query`): classifier, rewrite/HyDE/decompose, filters,
  freshness/superseded penalty, router, chunking variants; local LLM for query steps.
- **M4 Hybrid + rerank** (`m4-hybrid`): RRF and weighted fusion, cross-encoder reranking,
  golden v2 by pooling + LLM judge.
- **M3 Dense** (`m3-dense`): fastembed embeddings in Qdrant with ACL filters.
- **M2 BM25** (`m2-bm25`): own BM25/BM25F with a code-aware tokenizer, TF-IDF.
- **M1 Ingestion + eval** (`m1-eval`): sections with provenance, internal KB, golden set,
  calibrated judge, metrics, bootstrap CIs, CI eval gate.
- **M0 Setup** (`m0-setup`): repo, CI, configs, fakes, `doctor`, pinned corpus fetch.
