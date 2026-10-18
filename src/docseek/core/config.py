"""Configuration: secrets and endpoints from the environment, project config from YAML.

- ``Settings`` reads environment variables and ``.env`` (pydantic-settings). Secrets are
  ``SecretStr`` so they never show up in ``repr``, logs or tracebacks.
- ``configs/*.yaml`` hold versioned project config (corpus pin, principals, model names)
  and are validated into the typed models below.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class Settings(BaseSettings):
    """Runtime settings from environment variables (names as in ``.env.example``)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    # 127.0.0.1, not localhost: on Windows localhost tries ::1 first and docker-compose binds
    # IPv4 only, which adds ~2 s to every new connection.
    qdrant_url: str = "http://127.0.0.1:6333"
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("change-me")
    llm_cache_path: Path = Field(default=Path(".cache/llm.sqlite"), alias="DOCSEEK_LLM_CACHE")
    budget_usd: float = Field(default=5.0, gt=0, alias="DOCSEEK_BUDGET_USD")
    configs_dir: Path = Field(default=Path("configs"), alias="DOCSEEK_CONFIGS_DIR")

    def api_key_status(self) -> dict[str, bool]:
        """Whether each provider key is set. Never exposes the values themselves."""
        return {
            "GEMINI_API_KEY": _is_set(self.gemini_api_key),
            "OPENAI_API_KEY": _is_set(self.openai_api_key),
        }


def _is_set(secret: SecretStr | None) -> bool:
    return secret is not None and bool(secret.get_secret_value().strip())


class _YamlModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FastAPICorpusConfig(_YamlModel):
    source: str
    repo_url: str
    commit: str
    dest: Path
    sparse_paths: list[str] = Field(min_length=1)
    docs_root: str
    code_root: str
    exclude_files: list[str] = Field(default_factory=list)
    exclude_dirs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_commit(self) -> Self:
        if not _SHA1_RE.match(self.commit):
            raise ValueError(f"commit must be a full 40-char sha, got {self.commit!r}")
        return self


class InternalKBConfig(_YamlModel):
    source: str
    dir: Path
    doc_types: list[str] = Field(min_length=1)


class CorpusConfig(_YamlModel):
    manifest_path: Path
    fastapi: FastAPICorpusConfig
    internal_kb: InternalKBConfig


class UnknownPrincipalError(KeyError):
    """Raised for principals not declared in principals.yaml (access fails closed)."""


class PrincipalsConfig(_YamlModel):
    acl_groups: list[str] = Field(min_length=1)
    principals: dict[str, list[str]]

    @model_validator(mode="after")
    def _check_groups(self) -> Self:
        known = set(self.acl_groups)
        for name, groups in self.principals.items():
            unknown = set(groups) - known
            if unknown:
                raise ValueError(f"principal {name!r} uses undeclared ACL groups {sorted(unknown)}")
        return self

    def groups_for(self, principal: str) -> frozenset[str]:
        """ACL groups a principal may read.

        Raises:
            UnknownPrincipalError: if the principal is not declared.
        """
        try:
            return frozenset(self.principals[principal])
        except KeyError:
            raise UnknownPrincipalError(principal) from None


class LLMRoleConfig(_YamlModel):
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0] if "/" in self.model else ""

    @model_validator(mode="after")
    def _check_model(self) -> Self:
        if "/" not in self.model:
            raise ValueError(f"model must be '<provider>/<name>' for LiteLLM, got {self.model!r}")
        return self


class LLMConfig(_YamlModel):
    generator: LLMRoleConfig
    judge: LLMRoleConfig
    agent: LLMRoleConfig
    vision: LLMRoleConfig

    @model_validator(mode="after")
    def _check_judge_family(self) -> Self:
        if self.judge.provider == self.generator.provider:
            raise ValueError(
                "judge must use a different provider family than generator "
                f"(both are {self.judge.provider!r}) to limit self-preference bias"
            )
        return self


class ModelChoice(_YamlModel):
    default: str
    candidates: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_in_candidates(self) -> Self:
        if self.candidates and self.default not in self.candidates:
            raise ValueError(f"default {self.default!r} is not among candidates")
        return self


class ModelsConfig(_YamlModel):
    llm: LLMConfig
    embedding: ModelChoice
    reranker: ModelChoice


def load_yaml[M: BaseModel](path: Path, model: type[M]) -> M:
    """Load a YAML file and validate it into ``model``."""
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return model.model_validate(data if data is not None else {})


def load_corpus_config(configs_dir: Path = Path("configs")) -> CorpusConfig:
    return load_yaml(configs_dir / "corpus.yaml", CorpusConfig)


def load_principals(configs_dir: Path = Path("configs")) -> PrincipalsConfig:
    return load_yaml(configs_dir / "principals.yaml", PrincipalsConfig)


def load_models_config(configs_dir: Path = Path("configs")) -> ModelsConfig:
    return load_yaml(configs_dir / "models.yaml", ModelsConfig)
