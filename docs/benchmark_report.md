# DocSeek benchmark report (v1.0.0)

Every number below comes from an `experiments/` run named in the text. Runs from 2026-10-19.

## Setup

| | |
|---|---|
| Machine | Windows 11 laptop, Intel64 Family 6 Model 183 CPU (10 cores / 16 threads), 23.8 GB RAM, NVIDIA RTX 3050 6 GB Laptop GPU |
| GPU use | fastembed embeddings + `bge-reranker-base` via onnxruntime-directml (batch 8); Qdrant/Neo4j in Docker Desktop (WSL2) |
| Services | Qdrant 1.19.2, Neo4j 5.26 community (docker compose) |
| Corpus | 1,435 chunks: FastAPI docs + internal KB (text, `section_ctx` chunks), 103 image chunks, 46 OCR'd scan pages (M7 variant) |
| Queries | `golden_m7.jsonl` test split: 169 queries, 147 with relevant sections (v1 remapped + 10 scan + 10 image) |
| Answer model | `gemini/gemma-4-26b-a4b-it` (Gemini API free tier: 16k input tokens/min, client capped at 4 calls/min) |
| Judge | `openai/gpt-5.4-nano` through our cached, budgeted client (DeepEval 4.3) |

Runs: `20261019T034445Z_bench` (quality, cold build, latency), `20261019T041107Z_bench` (RSS per
fresh process, vector sweep, docker stats, disk), `20261019T024413Z_m8_answers` +
`…_judged` (answers, DeepEval), `20261019T033205Z_m8_answer_latency`,
`20261019T034533Z_m8_answer_summary`, `20261019T041123Z_m8_cost`.

## Retrieval configurations (test, 147 answerable queries)

Latency: warm, 3 timed passes over all 169 queries (507 searches) after one warm-up pass,
query-embedding and reranker score caches off. Cold build: all caches off (no BM25 pickle, no
embedding cache, in-memory Qdrant). RSS: one fresh Python process per config, after building
the retriever (imports alone: ~61 MB).

| Config | nDCG@10 [95% CI] | Recall@10 | MRR@10 | p50 / p95 / p99 ms | Cold build | RSS | $/1k q |
|---|---|---|---|---|---|---|---|
| BM25F (`bm25f`) | 0.693 [0.634, 0.749] | 0.850 | 0.649 | 2.3 / 6.7 / 13.8 | 2.7 s | 78 MB | 0 |
| Dense bge-small (`dense_bge_small`) | 0.675 [0.617, 0.735] | 0.830 | 0.636 | 240 / 318 / 402 | 54.8 s | 255 MB | 0 |
| Hybrid (`hybrid_ctx`, weighted α 0.3) | 0.718 [0.664, 0.774] | 0.864 | 0.680 | 117 / 286 / 361 | 63.7 s | 254 MB | 0 |
| + freshness (`fresh_ctx`, default) | 0.716 [0.663, 0.771] | 0.864 | 0.677 | 88 / 120 / 143 | 38.7 s | 261 MB | 0 |
| + rerank top-20 (`rerank_fresh_ctx`) | **0.758** [0.703, 0.810] | **0.895** | **0.727** | 1,129 / 1,229 / 1,269 | 42.2 s | 518 MB | 0 |
| Router (rules → image route, `m7_overall_router`) | 0.713 [0.660, 0.768] | 0.864 | 0.673 | 82 / 109 / 211 | 51.8 s | 579 MB | 0 |

Paired tests (`eval compare`, nDCG@10, n = 147): rerank vs BM25F **+0.065, p = 0.007**; rerank
vs fresh_ctx +0.042, p = 0.055; hybrid vs BM25F +0.025, p = 0.27; dense vs BM25F −0.018,
p = 0.52; router vs fresh_ctx −0.003, p = 1.0. On this corpus (scan pages and image chunks
included) fusion alone is not significantly better than BM25F, unlike on the text-only corpus
in M4 (+0.065, p ≤ 0.001); the reranker is what makes the difference.

nDCG@10 by query type (test):

| Config | concept. | factoid | fresh. | howto | image | multihop | perm. | scan | symbol |
|---|---|---|---|---|---|---|---|---|---|
| BM25F | 0.578 | 0.732 | 0.796 | 0.559 | **0.963** | 0.415 | 0.902 | 0.900 | 0.807 |
| Dense | 0.644 | 0.781 | 0.745 | 0.587 | 0.741 | 0.337 | **1.000** | 0.828 | 0.698 |
| Hybrid | 0.670 | 0.820 | 0.762 | **0.648** | 0.913 | 0.423 | 0.926 | 0.832 | 0.702 |
| fresh_ctx | 0.635 | 0.822 | 0.769 | **0.648** | 0.913 | 0.423 | 0.926 | 0.852 | 0.702 |
| + rerank | **0.720** | **0.835** | **0.876** | 0.623 | 0.889 | **0.433** | **1.000** | **0.963** | **0.836** |
| Router | 0.635 | 0.822 | 0.769 | **0.648** | 0.876 | 0.423 | 0.926 | 0.852 | 0.702 |

Latency notes: dense queries pay ~0.2 s for embedding the query on DirectML (the reranker
dominates its config at ~1.1 s for 20 pairs); BM25 is in-process. Hybrid and fresh_ctx run the
same children, yet hybrid's measured p95 is higher (286 vs 120 ms); the cause was not
isolated (GPU query-embedding variance is the likely suspect). Resident memory of the services after the benchmark: Qdrant 2.15 GiB (all
experiment collections kept), Neo4j 1.10 GiB (`docker stats`).

## Answer modes (60 stratified test queries; 47 answerable, 13 must abstain)

`scripts/m8_answer_eval.py` → `m8_answer_summary`, DeepEval via `scripts/m8_judge.py`
(answered, answerable questions only; reasons off). Latency: first uncached execution of each
(question, mode) across the answer runs (`m8_answer_latency`); "compute" = LLM call time +
tool time from the trace, i.e. without the client-side waits for the free tier's 4 calls/min.

| Mode | Faithfulness | Answer relev. | Ctx recall | Citation support | Cites a relevant section | Evidence R@10 | Correct abstain | Wrong abstain | Steps (mean / max) |
|---|---|---|---|---|---|---|---|---|---|
| (a) one-shot RAG (rerank_fresh_ctx top-5) | 0.952 | 0.912 | 0.767 | **0.976** | 87.2% | 0.872 | 13/13 | 2/47 | 1 / 1 |
| (b) agentic | **0.961** | **0.917** | 0.853 | 0.907 | 87.2% | **0.915** | 13/13 | 1/47 | 2.8 / 6 |
| (c) agentic, no FS tools | 0.956 | 0.910 | **0.907** | 0.946 | 85.1% | 0.883 | 13/13 | 1/47 | 2.5 / 6 |

95% CIs (bootstrap) overlap for every DeepEval metric except citation support, where one-shot
[0.953, 0.993] is above agentic [0.856, 0.952]. n judged: 45 / 46 / 46 (contextual recall
needs a reference answer: 39 each). No citation had to be removed by the post-check in any
mode (0 / 0 / 0); unanswerable and no-access questions were all abstained on.

| Mode | wall p50 / p95 (s) | compute p50 / p95 (s) | LLM calls / q | tokens in / out per q | $/1k q (free tier) | $/1k q if paid* |
|---|---|---|---|---|---|---|
| one-shot | 16.9 / 80.2 | 13.3 / 79.0 | 1.0 | 1,642 / 529 | 0 | 1.20 |
| agentic | 35.1 / 102.3 | 16.1 / 52.6 | 2.8 | 5,488 / 679 | 0 | 2.39 |
| agentic, no FS | 33.3 / 63.8 | 15.1 / 33.3 | 2.5 | 3,792 / 612 | 0 | 1.87 |

\* Measured tokens × LiteLLM list price of `gemini/gemini-3.1-flash-lite` ($0.25 / $1.50 per
M tokens), an estimate for a deployment without a free tier. LLM time is dominated by the
free-tier Gemma endpoint (p50 ≈ 13 s for one ~2k-token call, including provider retries);
tool calls are fast (agentic p50 0.28 s, p95 0.84 s in total per question).

Tools used by the agent (60 questions): search 84, get_section 16, ls 5, grep 3; without FS:
search 77, get_section 12, image_search 1. The graph tool was offered but never called.

## Vector index tradeoffs (dense bge-small, 1,435 vectors × 384)

Separate Qdrant collections with HNSW forced (`indexing_threshold` 1, `full_scan_threshold`
10 KB; by default Qdrant brute-forces collections this small and `ef` does nothing). No ACL
filter (index-level measurement only). 147 test queries × 3 passes.

| Variant | hnsw_ef | p50 / p95 ms | ANN recall@10 vs exact | nDCG@10 | Vector RAM (computed) |
|---|---|---|---|---|---|
| float32 | 16 / 64 / 256 | 8.8 / 26.2 · 8.5 / 27.9 · 9.4 / 27.8 | 0.990 / 1.000 / 1.000 | 0.663 / 0.667 / 0.667 | 2.10 MB |
| int8 + rescore | 16 / 64 / 256 | 8.9 / 26.8 · 8.6 / 28.1 · 9.9 / 27.6 | 0.984 / 0.996 / 0.997 | 0.664 / 0.667 / 0.667 | 0.53 MB (+ originals on disk) |
| int8, no rescore | 16 / 64 / 256 | 9.2 / 27.5 · 9.0 / 27.4 · 7.5 / 28.6 | 0.949 / 0.955 / 0.955 | 0.679 / 0.683 / 0.683 | 0.53 MB |

- At this size latency is the HTTP round trip (~9 ms); `ef` changes recall, not speed.
- int8 cuts vector RAM 4× and loses ~4.5 points of ANN recall without rescoring; rescoring
  with the original vectors recovers it (0.997). nDCG@10 moves the other way by +0.016 without
  rescoring, which is noise at this scale (the ANN errors happened to swap in labelled
  neighbours), not a reason to drop rescoring.
- Each collection takes ~347 MB on disk regardless of its 2 MB of vectors (Qdrant's
  pre-allocated WAL/segments); the Qdrant volume holds 12.7 GB because every experiment's
  collection is kept.

## Disk and one-time costs

| Item | Size |
|---|---|
| `data/processed` (sections, chunk variants) | 20.9 MB |
| BM25 index cache (`data/index/bm25`) | 13.9 MB |
| Embedding cache (`data/index/emb`) | 10.8 MB |
| Reranker score cache | 2.4 MB |
| Neo4j data | 546 MB |
| Qdrant storage (all experiment collections) | 12.7 GB |

| One-time LLM spend (ledger, list prices) | USD |
|---|---|
| Golden set relevance judge, golden v2 pooling (OpenAI, 3,980 calls) | 0.933 |
| DeepEval answer judging, M8 (OpenAI) | 0.410 |
| Internal KB generation (Gemini, 68 docs) | 0.043 (free tier) |
| Synthetic queries (Gemini) | 0.055 (free tier) |
| Image captions (103) + vision OCR (46 pages) (Gemini) | 0.117 (free tier) |
| Concepts, query understanding (local Ollama) | 0 |

OpenAI total over the project: $1.343 (cap $1.88). From `m8_cost` (ledger grouped by name).

## Failure analysis (answer modes, test)

Remaining errors in the agentic run, by cause, with examples:

- **Data / labels.** `q2019` ("telemetry for WebSocket connections") and `q0065` ("can
  sub-dependencies use yield?") were answered correctly from text sections that stayed in the
  corpus, but the labels point only at the scan page that also contains the answer, so the
  answer "misses" the relevant section. Same fact, two places: the scan variant replaced only
  the sections rendered within the 2-page limit.
- **Agent policy.** `q0104` "why is this template deprecated" (no template named): both
  agentic modes abstained at step 1 *without calling any tool*, judging the question
  unanswerable; one-shot, which always retrieves first, found and cited the right KB section.
  The query is a synthetic paraphrase that lost its subject, but searching would have worked.
- **Ranking / granularity.** `q0049` (JSON Lines use cases): the right page was retrieved,
  but the `#use-cases` section ranked below its siblings; the answer cites a neighbour.
- **Retrieval (one-shot).** `q0084` ("verify endpoints without starting a server"): the
  `TestClient` reference section was not in one-shot's top-5, so it abstained; the agent found
  it with a second search and answered correctly.
- **Generation.** One-shot abstained on `q2120` (image query) although the relevant section's
  caption chunk was in its context: the model judged it insufficient. Agentic citation support is
  lower than one-shot's (0.907 vs 0.976): with more evidence the model sometimes cites a
  neighbouring section for a claim.
- **Retrieval (multihop).** The weakest slice for every retriever (nDCG@10 ≈ 0.42–0.43): two
  facts in different sections, with paraphrased questions that name few entities; the graph
  tool was never chosen by the agent.

## What I'd do next

1. Split scan pages by detected headings so a scan chunk maps to one section (fixes most
   scan-label mismatches and the 0.733 → 0.619 drop of answers that exist only as scans).
2. A stronger answer model for the agent (or few-shot tool-use prompts) so it uses
   `graph_search` on multihop questions; measure on the multihop slice only.
3. Learned sparse retrieval (SPLADE) as the lexical child, and a cheaper reranker for the
   default mode (the cross-encoder costs ~1 s per query on this GPU).
4. Human review of the Claude-written labels and of a sample of DeepEval verdicts.
