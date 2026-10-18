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
| M6 Graph + FS | `m6-graph-fs` | next |

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
- `core/filtering.py`: `matches_filter(metadata, MetadataFilter)`.
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

## Open items

- See `USER_TODO.md` for human steps.
