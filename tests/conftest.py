from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Tests that call configure_logging bind a stream pytest later closes; undo that."""
    yield
    structlog.reset_defaults()


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def configs_dir() -> Path:
    return REPO_ROOT / "configs"
