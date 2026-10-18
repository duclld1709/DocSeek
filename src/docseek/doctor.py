"""Environment checks behind ``docseek doctor``.

Required checks (configs, Qdrant, Neo4j) make the command fail. Optional ones (API keys,
Tesseract) only warn, because later milestones need them, not M0. Secrets are reported as
set / not set and their values are never read into the output.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from docseek.core.config import (
    Settings,
    load_corpus_config,
    load_models_config,
    load_principals,
)

TIMEOUT_S = 3.0


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    required: bool = True


def check_configs(configs_dir: Path) -> CheckResult:
    try:
        load_corpus_config(configs_dir)
        load_principals(configs_dir)
        load_models_config(configs_dir)
    except Exception as exc:  # any parse/validation error is a config failure
        return CheckResult("configs", Status.FAIL, f"{type(exc).__name__}: {exc}")
    return CheckResult("configs", Status.OK, f"{configs_dir}/{{corpus,principals,models}}.yaml")


def check_qdrant(url: str) -> CheckResult:
    try:
        client = QdrantClient(url=url, timeout=int(TIMEOUT_S), check_compatibility=False)
        try:
            info = client.info()
        finally:
            client.close()
    except Exception as exc:
        return CheckResult("qdrant", Status.FAIL, f"{url} unreachable ({type(exc).__name__})")
    return CheckResult("qdrant", Status.OK, f"{url} (v{info.version})")


def check_neo4j(uri: str, user: str, password: str) -> CheckResult:
    try:
        with GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=TIMEOUT_S,
            connection_acquisition_timeout=TIMEOUT_S,
        ) as driver:
            driver.verify_connectivity()
            agent = driver.get_server_info().agent
    except Exception as exc:
        return CheckResult("neo4j", Status.FAIL, f"{uri} unreachable ({type(exc).__name__})")
    return CheckResult("neo4j", Status.OK, f"{uri} ({agent})")


def check_api_keys(settings: Settings) -> list[CheckResult]:
    return [
        CheckResult(
            name,
            Status.OK if is_set else Status.WARN,
            "set" if is_set else "not set (needed from M1 for LLM calls)",
            required=False,
        )
        for name, is_set in settings.api_key_status().items()
    ]


def check_tesseract() -> CheckResult:
    exe = shutil.which("tesseract")
    if exe is None:
        return CheckResult(
            "tesseract", Status.WARN, "not found on PATH (needed in M7 for OCR)", required=False
        )
    try:
        proc = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult("tesseract", Status.WARN, f"{exe} failed: {exc}", required=False)
    lines = (proc.stdout or proc.stderr).splitlines()
    return CheckResult("tesseract", Status.OK, lines[0] if lines else exe, required=False)


def run_checks(settings: Settings) -> list[CheckResult]:
    """Run every check in display order."""
    return [
        check_configs(settings.configs_dir),
        check_qdrant(settings.qdrant_url),
        check_neo4j(
            settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password.get_secret_value()
        ),
        *check_api_keys(settings),
        check_tesseract(),
    ]


def is_healthy(results: list[CheckResult], *, strict: bool = False) -> bool:
    """False if a required check failed, or (strict) if anything is not OK."""
    if strict:
        return all(r.status is Status.OK for r in results)
    return not any(r.required and r.status is Status.FAIL for r in results)
