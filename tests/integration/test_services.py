"""Checks against the real docker-compose services. Run: ``uv run pytest -m integration``."""

import pytest
from typer.testing import CliRunner

from docseek import doctor
from docseek.cli import app
from docseek.core.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture
def settings() -> Settings:
    return Settings()


def test_qdrant_reachable(settings: Settings) -> None:
    result = doctor.check_qdrant(settings.qdrant_url)
    assert result.status is doctor.Status.OK, result.detail


def test_neo4j_reachable(settings: Settings) -> None:
    result = doctor.check_neo4j(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password.get_secret_value()
    )
    assert result.status is doctor.Status.OK, result.detail


def test_doctor_exits_zero_with_services_up() -> None:
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
