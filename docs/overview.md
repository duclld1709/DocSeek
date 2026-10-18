# Overview: from keyword search to agentic retrieval

This page frames the problem DocSeek works on and why it is measured the way it is.

## 1. Traditional search

Classic search engines rank documents with **lexical** signals: an inverted index maps
terms to the documents containing them, and a scoring function such as TF-IDF or **BM25**
rewards rare terms that occur often in a short document. Lexical search is fast,
explainable and very strong on exact terms (API names, error codes, config keys). It fails
when the query and the document use different words for the same idea (*vocabulary
mismatch*): "run code after the response is sent" vs `BackgroundTasks`.

Production search engines add more than ranking: field boosts (title > body), metadata
filters, freshness, permissions, spelling correction and query understanding.

## 2. Semantic search

**Dense retrieval** embeds queries and passages into the same vector space with a
bi-encoder and ranks by cosine similarity, retrieved with an approximate nearest neighbour
index (HNSW in Qdrant). It handles paraphrase well but can miss exact identifiers and is
harder to debug. **Hybrid search** fuses lexical and dense rankings (RRF, DBSF, weighted
sums), and a **cross-encoder reranker** rescores the top candidates by reading the query
and passage together: slower, but more precise.

## 3. Retrieval-Augmented Generation (RAG)

RAG retrieves passages and asks an LLM to answer *from them*, with citations. The answer
can only be as good as the retrieved context, so most RAG failures are retrieval
failures: the right section was never retrieved, was ranked too low, was stale, or should
not have been visible to this user at all.

## 4. Agentic retrieval

An agent treats retrieval as a set of **tools** and decides what to call next: rewrite or
decompose the query, extract metadata filters, search lexically or semantically, follow a
knowledge graph for multi-hop questions, or browse the corpus like a developer would
(`ls`, `grep`, `cat`). This adds recall on hard questions but costs latency, tokens and
predictability, so it has to earn its place against simpler pipelines **on measurements**.

## 5. What makes modern search hard

| Problem | Example in DocSeek |
|---|---|
| Relevance vs exact match | `Annotated` Query alias vs "how do I rename a query parameter" |
| Structure | Answers live in a *section*, not a whole page; code is included from other files |
| Metadata | `doc_type`, `source`, version, status (active / deprecated) |
| Freshness | Two internal docs disagree; the newer one supersedes the older |
| Permissions | `security-team` documents must never reach an `engineer` |
| Modality | Answers inside scanned PDFs, tables and screenshots |
| Intent | Factoid vs how-to vs conceptual vs multi-hop queries need different pipelines |
| Unanswerable | The system must say "not found" instead of inventing an answer |

## 6. Why measure first

Without a benchmark you cannot tell whether a change helped, or whether a failure comes
from data, retrieval, ranking or generation. DocSeek therefore builds the evaluation
harness (golden set, section-level relevance labels, IR metrics checked against `ranx`,
calibrated LLM judges, bootstrap confidence intervals, paired significance tests and a CI
regression gate) **before** optimising retrieval. Every claim in the README and results
pages comes from a run recorded under `experiments/`.

## 7. Why this corpus

- **FastAPI docs** (MIT): realistic technical writing with a mix of tutorials, reference
  pages and code. Code examples are *included* from `docs_src/*.py`, headings carry
  explicit anchors, and pages link to each other, which gives stable section ids and a
  natural link graph. The corpus is pinned to commit
  `b23d78f2ce4e7b3ea48a59f51ca249690d9fd02c` so results are reproducible.
- It is small enough to iterate on a laptop CPU, yet has both exact-term queries (where
  BM25 should win) and conceptual ones (where dense retrieval should win).
- Extended corpora add what public docs lack: a simulated **internal KB** with ACLs and
  conflicting versions, **scanned PDFs** with ground-truth text for OCR error rates, and
  **images** for multimodal retrieval.
