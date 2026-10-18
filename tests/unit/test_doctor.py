from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from docseek import doctor
from docseek.core.config import Settings
from docseek.doctor import CheckResult, Status


def _settings(**env: Any) -> Settings:
    return Settings(_env_file=None, **env)  # type: ignore[call-arg]


def test_check_configs_ok_for_repo(configs_dir: Path) -> None:
    assert doctor.check_configs(configs_dir).status is Status.OK


def test_check_configs_fails_on_missing_dir(tmp_path: Path) -> None:
    result = doctor.check_configs(tmp_path)
    assert result.status is Status.FAIL
    assert "FileNotFoundError" in result.detail


class _FakeQdrant:
    def __init__(self, *, fail: bool, **_: Any) -> None:
        self.fail = fail
        self.closed = False

    def info(self) -> SimpleNamespace:
        if self.fail:
            raise ConnectionError("refused")
        return SimpleNamespace(version="1.19.2")

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(("fail", "status"), [(False, Status.OK), (True, Status.FAIL)])
def test_check_qdrant(monkeypatch: pytest.MonkeyPatch, fail: bool, status: Status) -> None:
    made: list[_FakeQdrant] = []

    def factory(**kw: Any) -> _FakeQdrant:
        made.append(_FakeQdrant(fail=fail, **kw))
        return made[-1]

    monkeypatch.setattr(doctor, "QdrantClient", factory)
    result = doctor.check_qdrant("http://q:6333")

    assert result.status is status
    assert made[0].closed
    if status is Status.OK:
        assert "1.19.2" in result.detail


class _FakeDriver:
    def __init__(self, fail: bool) -> None:
        self.fail = fail

    def __enter__(self) -> "_FakeDriver":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def verify_connectivity(self) -> None:
        if self.fail:
            raise OSError("no route")

    def get_server_info(self) -> SimpleNamespace:
        return SimpleNamespace(agent="Neo4j/5.26.32")


@pytest.mark.parametrize(("fail", "status"), [(False, Status.OK), (True, Status.FAIL)])
def test_check_neo4j(monkeypatch: pytest.MonkeyPatch, fail: bool, status: Status) -> None:
    seen: dict[str, Any] = {}

    def driver(uri: str, **kw: Any) -> _FakeDriver:
        seen.update(uri=uri, **kw)
        return _FakeDriver(fail)

    monkeypatch.setattr(doctor.GraphDatabase, "driver", driver)
    result = doctor.check_neo4j("bolt://n:7687", "neo4j", "pw-secret")

    assert result.status is status
    assert seen["auth"] == ("neo4j", "pw-secret")
    assert "pw-secret" not in result.detail


def test_check_api_keys_report_presence_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-abc")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    results = {r.name: r for r in doctor.check_api_keys(_settings())}

    assert results["GEMINI_API_KEY"].status is Status.OK
    assert results["OPENAI_API_KEY"].status is Status.WARN
    assert all(not r.required and "sk-abc" not in r.detail for r in results.values())


def test_check_tesseract_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    result = doctor.check_tesseract()
    assert (result.status, result.required) == (Status.WARN, False)


def test_check_tesseract_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "/usr/bin/tesseract")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="tesseract 5.5.1\n leptonica", stderr=""),
    )
    result = doctor.check_tesseract()
    assert result.status is Status.OK
    assert result.detail == "tesseract 5.5.1"


def _r(status: Status, required: bool = True) -> CheckResult:
    return CheckResult("x", status, "", required)


@pytest.mark.parametrize(
    ("results", "strict", "healthy"),
    [
        ([_r(Status.OK), _r(Status.WARN, required=False)], False, True),
        ([_r(Status.OK), _r(Status.WARN, required=False)], True, False),
        ([_r(Status.FAIL)], False, False),
        ([_r(Status.FAIL, required=False)], False, True),
    ],
)
def test_is_healthy(results: list[CheckResult], strict: bool, healthy: bool) -> None:
    assert doctor.is_healthy(results, strict=strict) is healthy
