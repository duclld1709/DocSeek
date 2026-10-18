# DocSeek progress (read this first after a context reset)

## Milestones

| Milestone | Tag | Status |
|---|---|---|
| M0 Foundation | `m0-setup` | done |
| M1 Ingestion + eval harness | `m1-eval` | done (golden_v1: 248 q, dev 99 / test 149) |
| M2 BM25 | `m2-bm25` | done |
| M3 Dense | `m3-dense` | next |

## Modules and CLI

- `lexical/`: `tokenizer.py` (code-aware), `inverted_index.py` (fields, pickle cache in
  `data/index/bm25/<hash>.pkl`), `bm25.py` (BM25/BM25F), `tfidf.py`, `retriever.py`
  (`LexicalRetriever`, `VisibilityMask` ACL+filter pre-filter, `top_k_hits`, `Bm25sScorer`),
  `factory.py` (config → retriever).
- `core/filtering.py`: `matches_filter(metadata, MetadataFilter)`.
- `testing/acl_checks.py`: `probe_leaks()` — use for every new retriever/tool test.
- `eval/grid.py` + `docseek eval grid --config X --param retriever.k1=0.9,1.2 ...` (dev only).
- `docseek index --config X`, `docseek search "q" --config X --principal P --k N`.
- Retriever types in `eval/runner.py::RETRIEVERS`: random, oracle, bm25, tfidf, bm25s.
- Scripts: `scripts/results_table.py RUN... --ref RUN` (markdown tables + p-values),
  `scripts/plot_grid_heatmap.py`, `scripts/m2_lexical_bench.py`.
- CI: jobs lint, unit, eval-gate (`tests/unit/eval/test_eval_gate.py` vs
  `tests/fixtures/baseline_metrics.json`; refresh with `DOCSEEK_UPDATE_BASELINE=1`).

## Best configs and test numbers

- M2 `configs/experiments/bm25f.yaml` (k1 1.5, b 0.5, title 2 / section_path 0.5 / body 1 /
  code 0.5): test nDCG@10 0.644, Recall@10 0.831, MRR@10 0.592, Leakage 0
  (`experiments/20261018T125341Z_bm25f_test`).

## Default decisions taken (no ADR needed)

- M2: Lucene idf `ln(1 + (N-df+.5)/(df+.5))`; query terms de-duplicated in BM25; BM25F =
  per-field length norm then one saturation (same b for all fields).
- M2: `section_path` field = ancestor headings only (title is its own field); `all` field =
  path + text for single-field indexes.
- M2: whole identifiers never stemmed or stop-worded; parts are. `FastAPI` splits to `fast`, `api`.
- M2: code-aware kept on despite no measurable gain on dev (+0.002, p=0.85 for turning it off).
- M2: ACL as pre-filter (mask before top-k) in lexical retrievers.
- M2: test split run once for tfidf, bm25_default, bm25_tuned, bm25f; ablations dev-only.

## Open items

- See `USER_TODO.md` for human steps.
