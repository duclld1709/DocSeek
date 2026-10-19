# DocSeek API / UI image. `docker compose up` builds it once and runs it as `api` and `ui`.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /app

# Dependencies first (cached layer), then the project.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
COPY tests/fixtures ./tests/fixtures
RUN uv sync --frozen --no-dev

EXPOSE 8000 8501
CMD ["uvicorn", "docseek.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
