import pytest
from typer.testing import CliRunner

from docseek import __version__, doctor
from docseek.cli import app
from docseek.doctor import CheckResult, Status

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def _patch_checks(monkeypatch: pytest.MonkeyPatch, results: list[CheckResult]) -> None:
    monkeypatch.setattr(doctor, "run_checks", lambda _settings: results)


def test_doctor_passes_with_only_optional_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_checks(
        monkeypatch,
        [
            CheckResult("qdrant", Status.OK, "v1.19.2"),
            CheckResult("tesseract", Status.WARN, "not found", required=False),
        ],
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "qdrant" in result.output and "tesseract" in result.output


def test_doctor_strict_fails_on_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_checks(monkeypatch, [CheckResult("tesseract", Status.WARN, "x", required=False)])
    assert runner.invoke(app, ["doctor", "--strict"]).exit_code == 1


def test_doctor_fails_when_required_service_down(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_checks(monkeypatch, [CheckResult("neo4j", Status.FAIL, "unreachable")])
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "docker compose up -d" in result.output


def test_doctor_never_prints_key_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw-should-not-leak")
    monkeypatch.setattr(doctor, "check_qdrant", lambda url: CheckResult("qdrant", Status.OK, url))
    monkeypatch.setattr(
        doctor, "check_neo4j", lambda uri, user, pw: CheckResult("neo4j", Status.OK, uri)
    )
    result = runner.invoke(app, ["doctor"])

    assert "GEMINI_API_KEY" in result.output
    assert "sk-should-not-leak" not in result.output
    assert "pw-should-not-leak" not in result.output
