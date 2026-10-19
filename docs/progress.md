# DocSeek progress (read this first after a context reset)

## Milestones

| Milestone | Tag | Status |
|---|---|---|
| M0 Foundation | `m0-setup` | done |
| M1 Ingestion + eval harness | `m1-eval` | done (golden_v1: 248 q, dev 99 / test 149) |
| M2 BM25 | `m2-bm25` | done |
| M3 Dense | `m3-dense` | done |
| M4 Hybrid + rerank | `m4-hybrid` | done (golden_v2: +1,512 judge labels, unreviewed) |
| M5 Query understanding | `m5-query` | done |
| M6 Graph + FS | `m6-graph-fs` | done |
| M7 OCR + multimodal | `m7-multimodal` | done |
| M8 Agent + API + bench | `v1.0.0` | done |

## Modules and CLI

- `lexical/`: `tokenizer.py` (code-aware), `inverted_index.py` (fields, pickle cache in
  `data/index/bm25/<hash>.pkl`), `bm25.py` (BM25/BM25F), `tfidf.py`, `retriever.py`
  (`LexicalRetriever`, `VisibilityMask` ACL+filter pre-filter, `top_k_hits`, `Bm25sScorer`),
  `factory.py` (config → retriever).
- `dense/`: `embedders.py` (`FastEmbedEmbedder` with model-card prefixes, `CachedEmbedder`
  → `data/index/emb/<model>/{passage,query}/`), `qdrant_store.py` (named vector `dense`,
  payload indexes, UUID5 ids, ACL `MatchAny` in every filter), `retriever.py`
  (`DenseRetriever`, `build_dense`; `embedder: hash` option for CI). Qdrant: `docker compose up -d qdrant`.
- `hybrid/`: `fusion.py` (`rrf`, `weighted`, `HybridRetriever`), `normalization.py`,
  `qdrant_native.py` (sparse BM25 + dense prefetch, Qdrant RRF/DBSF). `rerank/cross_encoder.py`
  (`CrossEncoderReranker` with SQLite score cache `data/index/rerank_scores.sqlite`,
  `RerankedRetriever`). Runner types: hybrid (children, method, alpha|weights, rrf_k, depth),
  rerank (first_stage, model, top_n), qdrant_hybrid. `eval/runner.py::build_child` builds nested configs.
- `core/onnx.py`: fastembed on GPU via onnxruntime-directml (Windows); `DOCSEEK_ONNX_PROVIDERS`.
- `eval/pooling.py` + `docseek eval pool RUN... --depth 10 [--execute]` (dry run shows cost).
- `query/`: `classifier.py` (rules → LLM), `filters.py` (FilterExtractor, search_with_fallback),
  `rewrite.py` (MultiQuery/HyDE/Decompose, RRF), `freshness.py` (FreshnessReranker,
  `score_transform: sigmoid` after a cross-encoder), `router.py`. Runner types via
  `eval/pipelines.py`: freshness, expand, filtered, router (LLM = `llm.query_role`).
- `ingest/strategies.py` + `scripts/m5_chunk_variants.py` → `data/processed/chunks_{fixed_256,section_ctx}.jsonl`;
  select with `chunks_file:` in an experiment YAML. `qtypes:` scores a subset.
- Local LLM: Ollama (`%LOCALAPPDATA%/Programs/Ollama/ollama.exe`, serves 127.0.0.1:11434),
  model `qwen2.5:3b-instruct`. Experiments: `DOCSEEK_LLM_STRICT=1`.
- `graph/`: `schema.py`, `extract.py` (structure, AST, release notes, LLM concepts + pruning),
  `concepts.py`, `loader.py`, `graph_retriever.py` (EntityLinker, TEMPLATES, API_CHANGES,
  GraphRetriever, `min_entities`). `docseek graph build --concept-pages 200`; Neo4j via
  `docker compose up -d neo4j`. Integration tests wipe Neo4j: `DOCSEEK_NEO4J_WIPE_OK=1
  uv run pytest -m integration`, then rebuild the graph.
- `ingest/projection.py` + `docseek project` → `data/projected/`; `fs/tools.py` (FsTools:
  ls/find/grep/cat, sandbox + ACL + limits + trace).
- `core/filtering.py`: `matches_filter(metadata, MetadataFilter)`.
- M7: `data/make_scans.py` (`docseek data make-scans` → `data/scans/`), `ingest/pdf_parser.py`
  (text layer ≥ 20 chars else OCR), `ingest/ocr.py` (Tesseract, RapidOCR + English rec model
  from HF, vision LLM), `eval/ocr_metrics.py` (CER/WER/table cells), `scripts/m7_ocr_bench.py`.
  `ingest/images.py` + `docseek data images` (captions via `llm.vision`, Tesseract OCR →
  `chunks_section_ctx_images.jsonl`, `chunks_images.jsonl`, `images.jsonl`);
  `dense/image_retriever.py` (type `clip`, Qdrant named vector `image`). `ingest/scans.py` +
  `docseek data scan-chunks` (→ `*_scans.jsonl`, `scan_map.json`, `golden_v1_scans.jsonl`,
  `golden_m7.jsonl`). Runner type `subset` (modality). Classifier rules for image/scan.
- `testing/acl_checks.py`: `probe_leaks()` — use for every new retriever/tool test.
- `eval/grid.py` + `docseek eval grid --config X --param retriever.k1=0.9,1.2 ...` (dev only).
- `docseek index --config X`, `docseek search "q" --config X --principal P --k N`.
- Retriever types in `eval/runner.py::RETRIEVERS`: random, oracle, bm25, tfidf, bm25s, dense, hybrid, rerank, qdrant_hybrid.
- Scripts: `scripts/results_table.py RUN... --ref RUN` (markdown tables + p-values),
  `scripts/plot_grid_heatmap.py`, `scripts/m2_lexical_bench.py`, `scripts/m3_embed_bench.py`,
  `scripts/m4_rerank_latency.py`, `scripts/hard_negatives.py`, `scripts/plot_quality_latency.py`.
- CI: jobs lint, unit, eval-gate (`tests/unit/eval/test_eval_gate.py` vs
  `tests/fixtures/baseline_metrics.json`; refresh with `DOCSEEK_UPDATE_BASELINE=1`).

## Best configs and test numbers

- M2 `configs/experiments/bm25f.yaml` (k1 1.5, b 0.5, title 2 / section_path 0.5 / body 1 /
  code 0.5): test nDCG@10 0.644, Recall@10 0.831, MRR@10 0.592, Leakage 0
  (`experiments/20261018T125341Z_bm25f_test`).

- M3 `configs/experiments/dense_bge_small.yaml`: test nDCG@10 0.641, Recall@10 0.823,
  MRR@10 0.606, Leakage 0 (`experiments/20261018T133514Z_dense_bge_small_test`).

- M4 `configs/experiments/rerank_bge20.yaml` (weighted min-max α 0.5 → bge-reranker-base top 20):
  test v1 nDCG@10 0.739, Recall@10 0.902, MRR@10 0.701, Leakage 0
  (`experiments/20261018T142702Z_rerank_bge20_test`); v2 0.799. Fast default:
  `hybrid_weighted.yaml` test v1 0.709 (`…134221Z_hybrid_weighted_test`).

- M5 `configs/experiments/fresh_ctx.yaml`: test nDCG@10 0.709, StaleHit@5 0.00 (0.20 without
  the superseded penalty), Leakage 0 (`experiments/20261018T164921Z_fresh_ctx_test`). Dev gains
  did not transfer to test; no query-time LLM component helped with the 3B local model.

- M6 `configs/experiments/hybrid_graph_w.yaml` (fresh_ctx + graph 0.2, ≥2 entities): test
  nDCG@10 0.707, multihop R@10 0.667 = fresh_ctx; graph not adopted as a default retriever.

- M7 OCR: Tesseract CER 0.032 / RapidOCR 0.098 / vision 0.121 (0.014 on non-refused pages;
  5/46 pages refused by Gemini's recitation filter); tables: vision 0.817 best. Image queries
  (test): caption + OCR 0.869, CLIP 0.393, text-only 0.386. Overall M7 test (169 q):
  fresh_ctx on text+image+scan chunks 0.716, router 0.713; Leakage 0
  (`experiments/20261019T010714Z_m7_overall_fresh_test`).

- M8 (`docs/benchmark_report.md`): golden_m7 test, 147 answerable: BM25F 0.693, fresh_ctx
  0.716, rerank 0.758 (p = 0.007 vs BM25F), router 0.713. Answers (60 q): faithfulness
  0.95-0.96, evidence R@10 agentic 0.915 vs one-shot 0.872, 13/13 correct abstentions.
  Runs `20261019T034445Z_bench`, `20261019T041107Z_bench`, `20261019T024413Z_m8_answers`.

## Default decisions taken (no ADR needed)

- M2: Lucene idf `ln(1 + (N-df+.5)/(df+.5))`; query terms de-duplicated in BM25; BM25F =
  per-field length norm then one saturation (same b for all fields).
- M2: `section_path` field = ancestor headings only (title is its own field); `all` field =
  path + text for single-field indexes.
- M2: whole identifiers never stemmed or stop-worded; parts are. `FastAPI` splits to `fast`, `api`.
- M2: code-aware kept on despite no measurable gain on dev (+0.002, p=0.85 for turning it off).
- M2: ACL as pre-filter (mask before top-k) in lexical retrievers.
- M2: test split run once for tfidf, bm25_default, bm25_tuned, bm25f; ablations dev-only.

- M3: second embedding model = snowflake-arctic-embed-s (nomic too slow on this CPU: ~1 h).
- M3: dense chunks embedded from `text_for_embedding` (= text for `section` strategy).
- M3: CPU here is slow for transformers (~0.6 s per 400-token chunk for bge-small); plan
  reranker/LLM-free work accordingly; caches under `data/index/` make reruns free.

- M4: weighted fusion gives missing chunks the list's minimum normalised score.
- M4: golden v2 pool = top-10 of bm25f, dense, hybrid, rerank@20 (rerank added after a
  measured pool bias); unanswerable queries not pooled; v1 stays the primary metric.
- M4: switched fastembed to GPU (DirectML) at the user's request; reranker batch 8.
- Budget: OpenAI ledger $0.933 of $2.00 after M4 → ~$1.07 left for M5–M8 (DeepEval in M8 is
  the main remaining OpenAI consumer; generator/agent/vision use Gemini free tier).

- M5: chunking = section_ctx, α 0.3 (ADR 0004; did not transfer to test, kept as it is not worse).
- M5: freshness stage = superseded penalty 0.5 only (β 0, deprecated 0) from the dev grid.
- M5: query role on local Ollama qwen2.5:3b (ADR 0005); Gemini free tier = 500 req/day/model;
  3.1-flash-lite quota for 2026-10-18 exhausted (generator/agent/vision role).
- M5: OpenAI cap lowered to $1.88 (owner has ~$1 left).
- Docker's Qdrant container stopped once when drive C: filled up; `docker compose up -d qdrant`.

- M6: concept pruning = stoplist + not an API name + df ≤ 5% of sections.
- M6: one-word API names link only with exact casing; FS hides restricted files as missing.
- M6: CI has an `integration` job (Neo4j + Qdrant service containers).

- M7: vision role moved to `gemini-3.5-flash-lite` (3.1-flash-lite daily quota used up).
- M7: scan unit = one chunk per page; scanned originals removed in the variant corpus; golden
  relevance remapped to pages (derived files; golden_v1 never edited).
- M7: default stays fresh_ctx over one index holding text, image and scan chunks; router
  image/scan routes measured, not adopted.
- Docker Desktop may be stopped after the laptop sleeps/reboots: start it, then
  `docker compose up -d qdrant` (and neo4j).

- M8: agent role = Gemma 4 26B-A4B (Gemini API free tier; 16k input tokens/min -> rpm 4);
  all three answer modes on it (ADR 0007). DeepEval via `eval/answer_judge.py` (judge role).
- M8: `docker compose up` = quick demo (`configs/service_demo.yaml`, mini corpus, hash
  embedder); `DOCSEEK_SERVICE_CONFIG=configs/service.yaml` for the full corpus. Verified in
  CI (`docker-demo`), not locally (drive C: nearly full).
- M8 modules: `agent/` (tools, loop, answer, runtime), `api/app.py`, `ui/app.py`,
  `eval/bench.py` + `bench_rss.py`, `eval/answer_judge.py`; scripts `m8_answer_eval.py`,
  `m8_judge.py`, `m8_answer_latency.py`, `m8_answer_summary.py`, `m8_cost.py`.
- Restart after reboot: Docker Desktop, `docker compose up -d qdrant neo4j`. Do not run the
  API/UI servers during long evals (RAM: they load the same models).

## Open items

- See `USER_TODO.md` for human steps.
