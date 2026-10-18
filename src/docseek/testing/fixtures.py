"""Builders for the committed fixture corpora (tests, CI eval gate, quick demo).

``tests/fixtures/mini_corpus`` (FastAPI docs stand-in), ``tests/fixtures/mini_kb`` (internal KB
stand-in) and ``tests/fixtures/mini_golden.jsonl`` (hand-labelled queries over both).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from docseek.core.config import CorpusConfig, load_corpus_config, load_principals
from docseek.ingest.chunking import ChunkingConfig
from docseek.ingest.pipeline import IngestResult, run_pipeline
from docseek.ingest.tokens import RegexTokenCounter

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
MINI_GOLDEN = FIXTURES / "mini_golden.jsonl"


def mini_corpus_config(repo_root: Path = REPO_ROOT) -> CorpusConfig:
    base = load_corpus_config(repo_root / "configs")
    fixtures = repo_root / "tests" / "fixtures"
    return base.model_copy(
        update={
            "fastapi": base.fastapi.model_copy(update={"dest": fixtures / "mini_corpus"}),
            "internal_kb": base.internal_kb.model_copy(update={"dir": fixtures / "mini_kb"}),
        }
    )


@cache
def mini_ingest() -> IngestResult:
    """Sections and chunks of the fixture corpora (offline token counter, no git dates)."""
    return run_pipeline(
        mini_corpus_config(),
        load_principals(REPO_ROOT / "configs"),
        RegexTokenCounter(),
        ChunkingConfig(),
        repo_root=REPO_ROOT,
        use_git_dates=False,
    )
