# Thin aliases over the docseek CLI (works with GNU make; on Windows run the commands directly).
.PHONY: setup services quick-demo demo data index serve ui test lint bench answers

setup:            ## install dependencies
	uv sync

services:         ## Qdrant + Neo4j
	docker compose up -d qdrant neo4j

quick-demo:       ## clean clone -> mini corpus demo in docker (< 2 min): UI on :8501
	docker compose up -d --build --wait qdrant api ui
	@echo "UI: http://127.0.0.1:8501   API docs: http://127.0.0.1:8000/docs"

data:             ## fetch -> ingest -> chunk variants -> scans/images -> projection
	uv run docseek data fetch
	uv run docseek ingest
	uv run python scripts/m5_chunk_variants.py
	uv run docseek data make-scans
	uv run docseek data images            # vision-LLM captions (Gemini free tier, cached)
	uv run docseek data scan-chunks
	uv run docseek project

index: services   ## dense/hybrid indexes and the knowledge graph
	uv run docseek index --config configs/experiments/fresh_ctx.yaml
	uv run docseek graph build --concept-pages 200

serve:            ## API on :8000 (full corpus, configs/service.yaml)
	uv run uvicorn docseek.api.app:app --port 8000

ui:               ## Streamlit UI on :8501 (talks to the API)
	uv run streamlit run src/docseek/ui/app.py

demo: data index  ## full pipeline: fetch -> ingest -> index -> open the UI
	DOCSEEK_SERVICE_CONFIG=configs/service.yaml docker compose up -d --build --wait api ui
	@echo "UI: http://127.0.0.1:8501"

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy src

bench:            ## benchmark report data -> experiments/<ts>_bench/report.json
	uv run docseek bench

answers:          ## answer-quality eval (3 modes) then DeepEval judge
	uv run python scripts/m8_answer_eval.py
	uv run python scripts/m8_judge.py $$(ls -d experiments/*_m8_answers | tail -1)
