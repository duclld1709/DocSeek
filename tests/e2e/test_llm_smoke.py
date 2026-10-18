"""Real provider calls (marker ``e2e``): verifies the model names in configs/models.yaml.

Costs well under one cent per run. Never runs in CI; needs keys in ``.env``:
``uv run pytest -m e2e tests/e2e``.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from docseek.core.config import LLMRoleConfig, Settings, load_models_config
from docseek.core.tracing import MemoryTraceSink, Tracer
from docseek.llm.client import build_llm_client

pytestmark = pytest.mark.e2e


class Echo(BaseModel):
    word: str


def _roles(configs_dir: Path) -> dict[str, LLMRoleConfig]:
    llm = load_models_config(configs_dir).llm
    return {"generator": llm.generator, "judge": llm.judge}


@pytest.mark.parametrize("role", ["generator", "judge"])
def test_model_answers_with_structured_output_and_cost(
    role: str, configs_dir: Path, tmp_path: Path
) -> None:
    settings = Settings()
    cfg = _roles(configs_dir)[role]
    if not settings.api_key_status()[f"{cfg.provider.upper()}_API_KEY"]:
        pytest.skip(f"no key for {cfg.provider}")
    sink = MemoryTraceSink()
    client = build_llm_client(
        settings.model_copy(update={"llm_cache_path": tmp_path / "c.sqlite"}), Tracer(sink)
    )
    messages = [{"role": "user", "content": 'Reply with JSON {"word": "pong"}.'}]
    resp = client.complete(
        messages, model=cfg.model, temperature=cfg.temperature, response_format=Echo
    )
    assert resp.parsed is not None and resp.parsed["word"].lower() == "pong"
    assert resp.cost_usd > 0 and resp.tokens_in > 0
    again = client.complete(
        messages, model=cfg.model, temperature=cfg.temperature, response_format=Echo
    )
    assert again.cached and [e.cost_usd for e in sink.events][1] == 0.0
