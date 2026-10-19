# Interview notes: DocSeek

Short answers grounded in this project's own measurements (numbers from `docs/results/` and
`docs/benchmark_report.md`, each traceable to an `experiments/` run).

**1. What is DocSeek in one sentence?**
An agentic retrieval system over the FastAPI docs, a synthetic internal KB with ACL and
freshness, scanned PDFs and screenshots, built around an evaluation harness so every change
is measured on a fixed golden set with confidence intervals.

**2. Why score relevance at section level, not chunk level?** (ADR 0002)
Chunking is a variable we tune; labels must not change when chunk size does. Chunks map back
to their section before scoring, so `fixed_256` and `section_ctx` are compared on identical
labels, and scans/images map to the section (or page) that shows them.

**3. Why write BM25 yourself?** (ADR 0001)
To control the tokenizer (code-aware: identifiers kept whole, parts stemmed) and BM25F field
weights, and to test the scoring against hand-computed values. `bm25s` is the reference
implementation in the tests. BM25F gave test nDCG@10 0.644 and is the best single retriever
on `symbol` queries (0.837 on test).

**4. RRF or weighted sum?**
Both were measured: weighted min-max with α = 0.5 and RRF with small k are equivalent within
noise on dev (M4). Fusion itself is the big win: +0.065 nDCG@10 over either retriever on test,
p ≤ 0.001. I kept weighted min-max because its α is interpretable and tunable per chunking
(α 0.3 with `section_ctx`, ADR 0004); RRF is the safer default when score scales drift.

**5. Did the reranker help?**
+0.030 nDCG@10 on test (0.709 → 0.739), not significant at n = 127 (p = 0.19), but hard
negatives are ordered correctly 98.6% vs 88.7% and `freshness` gains +0.18. More candidates
hurt (top-20 > 50 > 100). It costs ~1 s per query on the laptop GPU, so it is the quality
mode, not the default.

**6. How do you know the LLM judge is trustworthy?**
Weighted κ 0.922 against reference labels and a position-bias check (97% same grade when the
prompt order is swapped). Caveat I state up front: the reference labels were written by
Claude, not a human, so this κ is optimistic; 100 pooled labels are queued for human review.

**7. What was the pooling bias you found?**
The first golden v2 pool omitted reranker results, so 226 reranked hits were unjudged and
counted as non-relevant, penalising the reranker. Adding rerank@20 to the pool fixed it for
$0.083 of judge calls.

**8. How is ACL enforced?**
In code at the retriever/tool layer, never in prompts: Qdrant filters on `acl`, lexical
retrievers mask before top-k, Cypher templates filter on `$groups` and the retriever
re-checks, FS tools hide files as "missing", and the agent's principal is fixed by the server.
Leakage@10 is 0 in every run, and every retriever/tool has a probe test.

**9. Can the agent escalate its own access?**
No. The `ToolBox` is built for one principal from the `X-Principal` header; tools have no
principal parameter, a tool call that passes one is refused, and the API rejects a body
principal that differs from the header (tests in `tests/unit/agent`, `tests/unit/api`).

**10. How do you stop hallucinated citations?**
A citation survives only if that section was shown to the model in this request; others are
removed and reported. If nothing valid remains, the system abstains and lists the closest
sections.

**11. Freshness: what worked?**
A superseded penalty after retrieval (score × 0.5 for superseded KB docs) removed stale hits
(StaleHit@5 0.20 → 0.00 on test), but with n = 10 the drop is not significant (p = 0.49).
A sigmoid on cross-encoder logits was needed first: multiplying a negative logit by 0.5
*raised* the stale doc.

**12. When does the graph beat vector retrieval?**
When the query names two specific entities that co-occur in one section. On the test
`multihop` slice the graph did not raise Recall@10 (0.667 → 0.667): paraphrased questions
name few linkable entities, and generic ones ("FastAPI") link everywhere. Kept as an agent
tool, not as a default retriever.

**13. Why only parameterised Cypher templates?** (ADR 0006)
No injection surface and ACL inside every query; the cost is coverage limited to the shapes
we wrote.

**14. OCR: which engine?**
Tesseract: CER 0.032 at 1.4 s/page on CPU. The vision LLM is the best transcriber where it
answers (CER 0.014, tables 0.82 cell accuracy) but Gemini's recitation filter returned empty
output for 5 of 46 pages of public docs text, which is a production risk on its own.

**15. Image retrieval: CLIP or captions?**
Captions + OCR text: test nDCG@10 0.869 vs CLIP 0.393 on `image` queries (text-only
baseline 0.386). Screenshots differ by the words on them, which CLIP does not read; captions
cost one vision call per image once ($0.058 list price for 103 images, free tier).

**16. Biggest remaining weakness?**
Multihop (test nDCG@10 ≈ 0.42) and content that exists only as OCR'd scans (the 17 affected
test queries fell from 0.733 to 0.619). Next steps: section-aware page splitting for scans,
query decomposition with a stronger model, and learned sparse retrieval (SPLADE).

**17. Dev gains that did not transfer to test?**
M5's chunking and query-rewriting gains on dev disappeared on test; I reported that rather
than re-tuning on test. Test is run once per final config; grids refuse the test split.

**18. How is LLM cost controlled?**
Every call goes through one client: SQLite cache, per-run `BudgetGuard`, a project-wide spend
ledger with per-provider caps, and traces with tokens and cost. Free tiers (Gemini, Gemma,
local Ollama) carry the bulk; OpenAI (judge) stayed under the owner's ~$2 total.
