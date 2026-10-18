# ADR 0002: Judge relevance per section, map chunk hits back to sections

- **Status:** Accepted (M1)
- **Date:** 2026-10-18
- **Deciders:** project owner

## Context

DocSeek indexes **chunks**, and chunking changes between experiments: `section` in M1,
`fixed_256`, `section+context_header` and maybe `semantic` in M5. A golden set has to stay
valid across those experiments, or two chunking strategies cannot be compared on the same
labels.

Three possible units for relevance labels:

1. **Chunk.** Labels break whenever chunk boundaries move, and a strategy that splits a
   section in two looks like it "found" two relevant results.
2. **Document (page).** Stable, but too coarse: FastAPI pages hold up to dozens of sections,
   and "the right page" is not "the right answer".
3. **Section** (`section_id = "<doc_id>#<anchor>"`, from headings and their `{ #anchor }`).
   Stable across chunking strategies, as fine-grained as the docs' own structure, and what a
   citation should point to.

## Decision

- Golden labels are graded per **section**: 2 = answers directly, 1 = related/partial,
  absent = 0 (`data/golden/golden_v{N}.jsonl`).
- At evaluation time, chunk hits are mapped to their section with
  `ids.section_id_of_chunk`, and each section keeps its **best (first) rank**; later hits
  of the same section are dropped (`eval.metrics.hits_to_sections`). Metrics are then
  computed on that de-duplicated section ranking.
- Section ids are stable by construction: explicit anchors from the docs, otherwise a
  mkdocs-style slug, made unique per page (`ingest.markdown_parser`).

## Consequences

- Chunking strategies are compared fairly: splitting a section never adds relevant hits,
  and merging never loses one.
- Precision@k counts sections, not chunks, so a retriever cannot fill the top k with
  near-identical chunks of one section and look precise.
- A section that is long and only partly relevant is still one unit. The grade (1 vs 2)
  carries that nuance; finer passage-level labels are out of scope.
- If upstream docs rename an anchor, labels pointing at it break. The corpus is pinned to a
  commit (PROJECT_SPEC §3.1), and `eval.golden.validate_against_corpus` fails loudly on
  unknown section ids.
