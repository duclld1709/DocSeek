# ADR 0007: Own agent loop with server-bound principal, citation post-check and DeepEval on our client

- **Status:** Accepted (M8, 2026-10-19)
- **Deciders:** project owner (PROJECT_SPEC §6 M8, §7)

## Context

M8 needs an agent that searches the corpus with tools and answers with citations, compared
against one-shot RAG. Frameworks (LangGraph, LlamaIndex agents) would hide the loop that the
spec asks us to own, and make ACL, step limits and cost accounting harder to prove in tests.
DeepEval scores answers but calls OpenAI by itself, outside our cache, budget and ledger.
The free Gemini quotas (500 requests/day per model) and ~$0.95 of OpenAI budget limit how many
LLM calls M8 can make.

## Decision

- **Loop** (`agent/loop.py`): plain tool calling through `CachedLLMClient`; at most
  `max_steps` LLM calls (default 6, the last one asks for the answer now), a wall-clock
  timeout and a per-question spend cap. A degradable provider error falls back to one-shot
  RAG (§7); a run-budget error always stops the run.
- **Principal** (`agent/tools.py`): fixed when the server builds the `ToolBox` (API header
  `X-Principal`, eval golden query). No tool has a principal parameter; a call that passes
  one is refused. Every tool enforces ACL itself (retrievers, FS tools, graph, `get_section`);
  hidden sections look exactly like missing ones.
- **Citations** (`agent/answer.py`): `[section_id]` only; a citation survives only if that
  section was shown to the model in this request (the toolbox's evidence set). No valid
  citation, or the abstain sentence, gives an abstention with the closest sections.
- **Tool output** is wrapped in `<tool_output>` delimiters; the system prompt says document
  text is data, never instructions (prompt-injection risk from documents).
- **Answer model**: the `agent` role, the same model for all three modes so they differ only
  in the pipeline. Gemma 4 26B-A4B via the Gemini API (free tier) after the flash-lite daily
  quotas ran out and 2.5-flash-lite was closed to new users.
- **DeepEval** runs through `DocSeekJudge`, a `DeepEvalBaseLLM` adapter over our client
  (judge role, `gpt-5.4-nano`): every judge call is cached, budgeted and in the ledger.
  Reasons are off; only answered, answerable questions are judged; abstention is measured
  without a judge.

## Consequences

- Tests prove the security properties without any real LLM (`ScriptedLLM`): a scripted tool
  call that sets `principal` is refused, restricted sections never reach the evidence, the
  loop stops at `max_steps`, and fabricated citations are removed.
- The loop is ~150 lines we must maintain (no parallel tool calls, no streaming).
- Gemma's free tier allows 16k input tokens per minute, so evaluation runs at ~4 calls/min;
  latency numbers for the agent must separate rate-limit waits from compute.
- DeepEval results depend on a small judge model; they are reported with the judge name and
  as relative comparisons between modes, not as absolute quality.
