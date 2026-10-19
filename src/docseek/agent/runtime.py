"""Build the search backend and answerers from ``configs/service.yaml`` (API, UI, eval, bench).

Optional parts degrade instead of failing: no Neo4j → no ``graph_search``; no image chunks
→ no ``image_search``; no projection → no FS tools. What was disabled is reported in
``Runtime.disabled``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from docseek.agent.loop import AgentLoop, OneShotRAG
from docseek.agent.tools import SearchBackend, ToolBox
from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.core.interfaces import Retriever
from docseek.core.models import Chunk
from docseek.core.tracing import Tracer
from docseek.llm.client import CachedLLMClient

log = structlog.get_logger(__name__)
ANSWER_MODES = ("one-shot", "agentic", "agentic-no-fs")


@dataclass
class ServiceConfig:
    chunks_file: Path
    search_modes: dict[str, Any]
    one_shot: Any
    image: Any | None = None
    graph: bool = False
    projected_dir: Path | None = None
    agent: dict[str, Any] = field(default_factory=dict)


def load_service_config(path: Path = Path("configs/service.yaml")) -> ServiceConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ServiceConfig(
        chunks_file=Path(raw["chunks_file"]),
        search_modes=dict(raw["search_modes"]),
        one_shot=raw["one_shot"],
        image=raw.get("image"),
        graph=bool(raw.get("graph", False)),
        projected_dir=Path(raw["projected_dir"]) if raw.get("projected_dir") else None,
        agent=dict(raw.get("agent") or {}),
    )


def service_config_path(configs_dir: Path = Path("configs")) -> Path:
    """``$DOCSEEK_SERVICE_CONFIG`` (e.g. configs/service_demo.yaml) or configs/service.yaml."""
    import os

    env = os.environ.get("DOCSEEK_SERVICE_CONFIG")
    return Path(env) if env else configs_dir / "service.yaml"


def retriever_block(entry: Any) -> dict[str, Any]:
    """An inline retriever block, or the ``retriever:`` block of an experiment YAML path."""
    if isinstance(entry, Mapping):
        return dict(entry)
    data = yaml.safe_load(Path(str(entry)).read_text(encoding="utf-8"))
    return dict(data["retriever"])


@dataclass
class Runtime:
    config: ServiceConfig
    policy: AccessPolicy
    backend: SearchBackend
    one_shot_retriever: Retriever
    disabled: dict[str, str] = field(default_factory=dict)

    def toolbox(
        self, principal: str, *, use_fs: bool = True, tracer: Tracer | None = None
    ) -> ToolBox:
        k = int(self.config.agent.get("k", 5))
        return ToolBox(self.backend, self.policy, principal, k=k, use_fs=use_fs, tracer=tracer)

    def answerer(self, mode: str, llm: CachedLLMClient, model: str) -> AgentLoop | OneShotRAG:
        if mode not in ANSWER_MODES:
            raise ValueError(f"mode must be one of {ANSWER_MODES}")
        rag = OneShotRAG(self.one_shot_retriever, llm, model, k=int(self.config.agent.get("k", 5)))
        if mode == "one-shot":
            return rag
        a = self.config.agent
        return AgentLoop(
            llm,
            model,
            max_steps=int(a.get("max_steps", 6)),
            timeout_s=float(a.get("timeout_s", 90)),
            budget_usd=float(a.get("budget_usd", 0.05)),
            fallback=rag,
            mode=mode,
        )

    def answer(
        self, mode: str, question: str, principal: str, llm: CachedLLMClient, model: str
    ) -> Any:
        tools = self.toolbox(principal, use_fs=mode != "agentic-no-fs", tracer=llm.tracer)
        return self.answerer(mode, llm, model).run(question, tools)


def build_runtime(
    config: ServiceConfig | None = None,
    *,
    chunks: list[Chunk] | None = None,
    configs_dir: Path = Path("configs"),
) -> Runtime:
    from docseek.eval.runner import build_child
    from docseek.ingest.pipeline import read_jsonl

    cfg = config or load_service_config(service_config_path(configs_dir))
    policy = AccessPolicy(load_principals(configs_dir))
    chunks = chunks if chunks is not None else read_jsonl(cfg.chunks_file, Chunk)
    disabled: dict[str, str] = {}
    retrievers = {
        mode: build_child(retriever_block(entry), chunks, [], policy, 13)
        for mode, entry in cfg.search_modes.items()
    }
    one_shot = build_child(retriever_block(cfg.one_shot), chunks, [], policy, 13)
    image = None
    if cfg.image is not None:
        try:
            image = build_child(retriever_block(cfg.image), chunks, [], policy, 13)
        except ValueError as exc:
            disabled["image_search"] = str(exc)
    graph = None
    if cfg.graph:
        try:
            from docseek.graph.graph_retriever import build_graph_retriever

            graph = build_graph_retriever({}, chunks, policy)
        except Exception as exc:  # Neo4j down or graph not built
            disabled["graph_search"] = f"{type(exc).__name__}: {exc}"
    fs_root = cfg.projected_dir
    if fs_root is not None and not (fs_root / "_manifest.json").is_file():
        disabled["fs"] = f"no projection at {fs_root} (run `docseek project`)"
        fs_root = None
    for tool, why in disabled.items():
        log.warning("tool_disabled", tool=tool, reason=why[:200])
    backend = SearchBackend(
        retrievers=retrievers, chunks=chunks, image=image, graph=graph, fs_root=fs_root
    )
    return Runtime(cfg, policy, backend, one_shot, disabled)
