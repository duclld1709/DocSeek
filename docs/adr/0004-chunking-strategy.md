# ADR 0004: Chunk by section and prepend the heading path to the embedded text

- **Status:** Accepted (M5, 2026-10-18)
- **Deciders:** project owner (decision delegated to Claude in the autonomous M2–M8 run)

## Context

Relevance is judged per section (ADR 0002), so every chunking strategy can be compared on the
same golden set. M5 compares three strategies on the dev split (85 answerable queries),
with the M4 retrievers and no other change:

- `section` (M1): one chunk per heading-delimited section, long sections split on paragraphs
  (≤ 512 tokens, 64 overlap). 1,426 chunks covering 1,287 non-empty sections.
- `fixed_256`: each document cut into 256-token windows (32 overlap, bge tokenizer) ignoring
  headings; a window is attributed to the section holding most of its tokens. 1,276 chunks
  covering only 835 sections.
- `section_ctx`: the `section` chunks, with `title path > … > section` prepended to
  `text_for_embedding` only (`text`, BM25 fields and display are unchanged).

## Measurements (dev nDCG@10; `experiments/` run directories)

| strategy | dense bge-small | hybrid α = 0.5 | hybrid, α re-tuned |
|---|---|---|---|
| section | 0.662 | 0.735 | 0.735 (α 0.5) |
| fixed_256 | 0.512 | 0.547 | — |
| **section_ctx** | **0.715** | 0.731 | **0.769** (α 0.3) |

Runs: `20261018T145216Z_dense_bge_small_grid_dev`, `20261018T145142Z_hybrid_weighted_grid_dev`,
`20261018T145257Z_hybrid_weighted_grid_dev` (α ∈ {0.2, 0.3, 0.4} → 0.752 / 0.769 / 0.755).
With `section` chunks α = 0.3 gives 0.724 (`20261018T133918Z_hybrid_weighted_grid_dev`).

## Decision

Use `section_ctx` with hybrid weight α = 0.3 on BM25F (0.7 on dense).

## Consequences

- Short sections ("Recap", "Technical details", "Timeline") are ambiguous on their own; the
  path header tells the embedder which page and topic they belong to. The gain is on the dense
  side only (+0.053), so the fusion weight moves towards dense.
- Fixed windows lose a third of the sections outright (a section that never holds the majority
  of a window cannot be returned) and mix neighbouring topics; they are only worth it for
  corpora without usable structure (e.g. OCR output, M7).
- Re-embedding is needed whenever headings change; the embedding cache keys on the full
  embedded text, so unchanged sections are reused.
- Not tested: semantic chunking (out of scope) and header-only-for-BM25 variants.
