# ADR 0003: Cheapest structured-output models, thinking off, and a project-wide spend ledger

- **Status:** Accepted (M1)
- **Date:** 2026-10-18
- **Deciders:** project owner

## Context

The project has **$2 of OpenAI credit for M1-M8**. Gemini runs on the free tier (rate
limited, not billed). M0 picked `gemini/gemini-3.8-flash` as generator and
`openai/gpt-5.4-mini` as judge. Before any real call in M1, the plan was re-estimated:

- M1 makes on the order of 1,000 judge calls (answerability checks, hard-negative checks,
  calibration in two orders) and a few hundred generator calls.
- With the M0 models the estimate was $1.2-3.5 for M1 alone: over the whole budget.
- Both M0 models are reasoning models. Thinking tokens are billed as output tokens, so the
  real cost could be several times the visible output.
- Checked offline with LiteLLM 1.106.0 (`mock_response`, no network): `gpt-5-nano` rejects any
  `temperature` other than 1 while reasoning is on, and its reasoning cannot be turned off,
  so a deterministic judge is impossible with it. `gpt-5.4-nano` and `gpt-5.4-mini` accept
  `reasoning_effort="none"` with `temperature=0`. `gemini-3.1-flash-lite` accepts
  `reasoning_effort="none"`.

List prices in LiteLLM 1.106.0's bundled map (USD per million tokens, input / output):

| Model | Input | Output | Role |
|---|---|---|---|
| `gemini/gemini-3.8-flash` | 0.75 | 3.75 | M0 generator |
| `gemini/gemini-3.1-flash-lite` | 0.25 | 1.50 | **new generator (free tier)** |
| `openai/gpt-5.4-mini` | 0.75 | 4.50 | M0 judge |
| `openai/gpt-5.4-nano` | 0.20 | 1.25 | **new judge** |
| `openai/gpt-5-nano` | 0.05 | 0.40 | rejected: reasoning always on, temperature fixed at 1 |

## Decision

1. **Generator / agent / vision:** `gemini/gemini-3.1-flash-lite`, `reasoning_effort: none`,
   client-side cap of 10 requests per minute for the free tier (`rpm` in
   `configs/models.yaml`; a conservative guess, to adjust if the provider allows more).
2. **Judge:** `openai/gpt-5.4-nano`, `reasoning_effort: none`, `temperature: 0`. It still comes
   from a different provider family than the generator (self-preference bias, §5.4).
3. **Per-model parameters are part of the cache key**, so changing `reasoning_effort` never
   returns an answer produced under other settings.
4. **Spend ledger:** every paid call of every run is appended to a `spend` table in the LLM
   cache database. Before a real call, `CachedLLMClient` refuses if the provider's
   cumulative list-price spend reached `spend_limits_usd` (`openai: 2.0`). This is on top of
   the per-run `DOCSEEK_BUDGET_USD` cap. Cache hits are free and not recorded.
5. **Smaller M1 workload:** oversample 1.3 (was 1.5), 2 hard-negative candidates per query
   (was 3), section text sent to the generator and the judge capped at 3,000 characters
   (2.3% of the 1,146 Corpus A sections are longer).

## Consequences

- The judge is a small model. Its agreement with human labels is **measured**, not assumed:
  Cohen's kappa on 100 hand-labelled pairs (`docseek eval calibrate`). If kappa < 0.6, the
  rubric is revised first; switching the judge back to `gpt-5.4-mini` is the fallback, and it
  costs about 3.6x more per call.
- Gemini free-tier costs in traces are LiteLLM list prices, not money spent. Reports state
  both: list price per provider, and actual spend (OpenAI only).
- Free-tier rate limits make generation slower (minutes, not seconds). The cache makes a
  rerun after a 429 resume where it stopped, at no cost.
- Later milestones (pooling in M4, query understanding in M5, captions in M7, agent and
  DeepEval in M8) share what M1 leaves. Each must plan its LLM calls against the ledger.
