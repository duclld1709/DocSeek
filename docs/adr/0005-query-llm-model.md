# ADR 0005: A separate model for query understanding (free-tier quota)

- **Status:** Accepted (M5, 2026-10-18)
- **Deciders:** project owner (chose "switch to another Gemini model" when asked)

## Context

The generator, agent and vision roles use `gemini/gemini-3.1-flash-lite` on the Gemini free
tier (ADR 0003). On 2026-10-18 the M5 classifier run hit HTTP 429
`GenerateRequestsPerDayPerProjectPerModel-FreeTier` with quota 500: M1 synthesis and the
first 158 classifier calls had used the day's requests. The quota is per model. M5–M8 need
on the order of 2,000 more generator-side calls.

Options considered: wait for the daily reset (multi-day schedule), enable paid billing,
use OpenAI for generation (same family as the judge, and the OpenAI budget is capped at $2),
or use another Gemini model with its own free quota.

## Decision

Add an optional `llm.query` role in `configs/models.yaml` for query understanding (classify,
filters, rewrite, HyDE, decompose), set to `gemini/gemini-3.5-flash-lite` (structured output,
thinking off, rpm 10). The existing roles stay on `gemini-3.1-flash-lite`, so M1's cached
synthesis remains reproducible. Later milestones may assign agent/vision to whichever Gemini
model still has quota; each such change is recorded here.

Experiments run with `DOCSEEK_LLM_STRICT=1`: a provider error (429, timeout) fails the run
instead of degrading to the un-expanded query, so no reported number silently measures a
degraded pipeline. Interactive use keeps the degrade behaviour (PROJECT_SPEC §7).

## Consequences

- The 158 classifier calls cached under 3.1-flash-lite are not reused (cache keys include the
  model); the classifier is re-measured with 3.5-flash-lite.
- M5 LLM work is sized to stay within one model's daily quota: classifier accuracy on dev
  (test accuracy comes from the final router run), decomposition scored on multihop queries.
- Results tables name the model next to every LLM-backed number.
