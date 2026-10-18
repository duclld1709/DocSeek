from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from docseek.core.config import (
    ModelsConfig,
    PrincipalsConfig,
    Settings,
    UnknownPrincipalError,
    load_corpus_config,
    load_models_config,
    load_principals,
    load_yaml,
)

PINNED_COMMIT = "b23d78f2ce4e7b3ea48a59f51ca249690d9fd02c"
ENV_VARS = [
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "QDRANT_URL",
    "NEO4J_PASSWORD",
    "DOCSEEK_BUDGET_USD",
    "DOCSEEK_LLM_CACHE",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_settings_defaults_without_env() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.budget_usd == 5.0
    assert settings.llm_cache_path == Path(".cache/llm.sqlite")
    assert settings.api_key_status() == {"GEMINI_API_KEY": False, "OPENAI_API_KEY": False}


def test_settings_read_env_and_hide_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-very-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setenv("DOCSEEK_BUDGET_USD", "1.5")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.budget_usd == 1.5
    assert settings.api_key_status() == {"GEMINI_API_KEY": True, "OPENAI_API_KEY": False}
    assert "sk-very-secret" not in repr(settings)
    assert "sk-very-secret" not in settings.model_dump_json()


def test_settings_read_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("QDRANT_URL=http://qdrant:6333\nNEO4J_PASSWORD=pw\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.neo4j_password.get_secret_value() == "pw"


def test_settings_reject_non_positive_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCSEEK_BUDGET_USD", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_repo_corpus_config_is_pinned(configs_dir: Path) -> None:
    corpus = load_corpus_config(configs_dir)

    assert corpus.fastapi.commit == PINNED_COMMIT
    assert corpus.fastapi.sparse_paths == ["docs/en/docs", "docs_src"]
    assert "release-notes.md" in corpus.fastapi.exclude_files
    assert corpus.manifest_path == Path("data/raw/MANIFEST.json")


def test_corpus_config_rejects_short_sha(configs_dir: Path, tmp_path: Path) -> None:
    data = yaml.safe_load((configs_dir / "corpus.yaml").read_text(encoding="utf-8"))
    data["fastapi"]["commit"] = "b23d78f"
    (tmp_path / "corpus.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValidationError, match="40-char"):
        load_corpus_config(tmp_path)


def test_repo_principals(configs_dir: Path) -> None:
    principals = load_principals(configs_dir)

    assert principals.groups_for("anonymous") == {"public"}
    assert principals.groups_for("engineer") == {"public", "eng-internal"}
    assert principals.groups_for("security") == {"public", "eng-internal", "security-team"}


def test_unknown_principal_fails_closed(configs_dir: Path) -> None:
    with pytest.raises(UnknownPrincipalError):
        load_principals(configs_dir).groups_for("admin")


def test_principals_reject_undeclared_group() -> None:
    with pytest.raises(ValidationError, match="undeclared"):
        PrincipalsConfig(acl_groups=["public"], principals={"x": ["public", "root"]})


def test_repo_models_config(configs_dir: Path) -> None:
    models = load_models_config(configs_dir)

    assert models.llm.generator.provider != models.llm.judge.provider
    assert models.embedding.default == "BAAI/bge-small-en-v1.5"
    assert models.reranker.default in models.reranker.candidates


def _models(**llm_overrides: dict[str, object]) -> dict[str, object]:
    llm = {
        "generator": {"model": "gemini/a"},
        "judge": {"model": "openai/b"},
        "agent": {"model": "gemini/a"},
        "vision": {"model": "gemini/a"},
    }
    llm.update(llm_overrides)
    return {"llm": llm, "embedding": {"default": "e"}, "reranker": {"default": "r"}}


def test_models_config_requires_judge_from_other_family() -> None:
    with pytest.raises(ValidationError, match="different provider family"):
        ModelsConfig.model_validate(_models(judge={"model": "gemini/b"}))


def test_models_config_requires_provider_prefix() -> None:
    with pytest.raises(ValidationError, match="provider"):
        ModelsConfig.model_validate(_models(agent={"model": "flash"}))


def test_load_yaml_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_yaml(tmp_path / "empty.yaml", ModelsConfig)
