# ADR 0005: Run query understanding on a local model (Ollama on the laptop GPU)

- **Status:** Accepted (M5, 2026-10-18). Supersedes the same-day draft that used
  `gemini-3.5-flash-lite` for this role.
- **Deciders:** project owner (proposed running a local model on the GPU; approved the plan)

## Context

The generator, agent and vision roles use `gemini/gemini-3.1-flash-lite` on the Gemini free
tier (ADR 0003). On 2026-10-18 the M5 classifier run hit HTTP 429
`GenerateRequestsPerDayPerProjectPerModel-FreeTier` (quota 500): M1 synthesis and the first
classifier calls had used the day's requests. The free tier also caps requests per minute
(the client throttles to 10 rpm, about 50 minutes per 400 calls). The owner has about $1 of
OpenAI credit left for the rest of the project, reserved for the M8 judge/DeepEval work.
M5 alone needs several hundred short structured calls (classify, filters, rewrite, HyDE,
decompose).

## Decision

Add an optional `llm.query` role in `configs/models.yaml` for query understanding, served by
Ollama on the local RTX 3050 6 GB: `ollama_chat/qwen2.5:3b-instruct` (4-bit, 1.9 GB),
temperature 0, `api_base: http://127.0.0.1:11434`. It goes through the same LiteLLM →
`CachedLLMClient` path (cache, trace, budget), with cost recorded as $0. Other roles are
unchanged, so M1's cached synthesis stays reproducible. A second candidate
(`qwen3:4b-instruct`) was planned for a pilot but not pulled: drive C: ran out of space.

Experiments run with `DOCSEEK_LLM_STRICT=1`: provider errors fail the run instead of degrading
to the un-expanded query, so no reported number silently measures a degraded pipeline.
Interactive use keeps the degrade behaviour (PROJECT_SPEC §7). The OpenAI cap is lowered to
$1.88 (ledger $0.933 + $0.95) so the ledger stops before the remaining credit runs out.

## Consequences

- No quota and no cost for M5 LLM steps; a warm call takes 0.2–0.5 s (first call ~70 s while
  the model loads). Results name the model next to every LLM-backed number.
- A 3B model is weaker than Gemini at rewriting and HyDE passages; M5 measures each component
  rather than assuming a gain, and the router only uses components that help on dev.
- Reproducing M5 requires Ollama with this model (`docs/progress.md`); CI does not (tests use
  `ScriptedLLM`).
- Agent and judge roles for M8 are decided separately with measurements; a 3B model is not
  assumed to be a usable judge without its own calibration.
