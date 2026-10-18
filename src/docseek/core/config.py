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
    experiments_dir: Path = Field(default=Path("experiments"), alias="DOCSEEK_EXPERIMENTS_DIR")

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
    # Passed to LiteLLM as-is ("none" disables thinking, whose tokens are billed as output).
    reasoning_effort: str | None = None
    # Client-side cap on requests per minute (free tiers rate-limit hard); None = no cap.
    rpm: int | None = Field(default=None, gt=0)
    # Server URL for self-hosted models (Ollama); passed to LiteLLM as ``api_base``.
    api_base: str | None = None

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
    # Query understanding (classify, rewrite, HyDE, decompose, filters). Defaults to the
    # generator; set separately when a free-tier daily quota is per model (ADR 0005).
    query: LLMRoleConfig | None = None

    @property
    def query_role(self) -> LLMRoleConfig:
        return self.query or self.generator

    def roles(self) -> tuple[LLMRoleConfig, ...]:
        extra = (self.query,) if self.query is not None else ()
        return (self.generator, self.judge, self.agent, self.vision, *extra)

    @model_validator(mode="after")
    def _check_judge_family(self) -> Self:
        if self.judge.provider == self.generator.provider:
            raise ValueError(
                "judge must use a different provider family than generator "
                f"(both are {self.judge.provider!r}) to limit self-preference bias"
            )
        return self


class Prefixes(_YamlModel):
    query: str = ""
    passage: str = ""


class ModelChoice(_YamlModel):
    default: str
    candidates: list[str] = Field(default_factory=list)
    # Model-card instruction prefixes for embedding models (query vs passage side).
    prefixes: dict[str, Prefixes] = Field(default_factory=dict)

    def prefixes_for(self, model: str) -> Prefixes:
        return self.prefixes.get(model, Prefixes())

    @model_validator(mode="after")
    def _default_in_candidates(self) -> Self:
        if self.candidates and self.default not in self.candidates:
            raise ValueError(f"default {self.default!r} is not among candidates")
        return self


class ModelsConfig(_YamlModel):
    llm: LLMConfig
    embedding: ModelChoice
    reranker: ModelChoice
    # Cumulative spend cap per provider across every run (spend ledger); absent = no cap.
    spend_limits_usd: dict[str, float] = Field(default_factory=dict)
    # Local OCR model files (Hugging Face repo + file), e.g. the RapidOCR English recogniser.
    ocr: dict[str, dict[str, str]] = Field(default_factory=dict)
    # CLIP towers for text-to-image retrieval (M7): {"image": ..., "text": ...}.
    image_embedding: dict[str, str] = Field(default_factory=dict)

    def model_params(self) -> dict[str, dict[str, object]]:
        """Extra LiteLLM kwargs per model name (same model in two roles must agree)."""
        out: dict[str, dict[str, object]] = {}
        for role in self.llm.roles():
            params: dict[str, object] = {}
            if role.reasoning_effort is not None:
                params["reasoning_effort"] = role.reasoning_effort
            if role.api_base is not None:
                params["api_base"] = role.api_base
            if out.setdefault(role.model, params) != params:
                raise ValueError(f"roles using {role.model} disagree on extra parameters")
        return out

    def rpm_limits(self) -> dict[str, int]:
        roles = self.llm.roles()
        limits: dict[str, int] = {}
        for role in roles:
            if role.rpm is not None:
                limits[role.model] = min(role.rpm, limits.get(role.model, role.rpm))
        return limits


class ChunkingSettings(_YamlModel):
    strategy: str = "section"
    max_tokens: int = Field(default=512, gt=0)
    overlap_tokens: int = Field(default=64, ge=0)


class IngestConfig(_YamlModel):
    output_dir: Path
    chunking: ChunkingSettings


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


def load_ingest_config(configs_dir: Path = Path("configs")) -> IngestConfig:
    return load_yaml(configs_dir / "ingest.yaml", IngestConfig)
