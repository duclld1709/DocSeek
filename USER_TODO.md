# Things that need a person

Claude kept working with a stand-in for each item; nothing here blocks the milestones.

## M2

- [ ] Skim the failure-analysis causes in `docs/results/M2.md` (assigned by Claude, not a
  person) and correct any category you disagree with.

## M4

- [ ] Review a random sample of 100 pooled judge labels in `data/golden/pool_judgments_v2.jsonl`
  (all `reviewed: false`; PROJECT_SPEC §5.3 step 3). v2 numbers in `docs/results/M4.md` rest
  on these unreviewed labels.

## Machine (found in M5)

- [ ] Drive C: is full (202 GB used, ~0.2 GB free on 2026-10-18). Ollama put its program
  (2.8 GB) and models (1.9 GB, `qwen2.5:3b-instruct`) there. Optional: set the user env var
  `OLLAMA_MODELS=D:\ollama\models`, restart Ollama and re-pull, then free C: as you see fit.
  The project's own caches (TEMP, fastembed, indexes) are already on D:.
