"""Run directories and env.json content."""

import json
from datetime import UTC, datetime
from pathlib import Path

from docseek.core.runs import environment_info, make_run_dir, write_json

NOW = datetime(2026, 10, 18, 13, 5, 1, tzinfo=UTC)


def test_run_dirs_are_timestamped_and_never_reused(tmp_path: Path) -> None:
    a = make_run_dir(tmp_path, "gen-internal", now=NOW)
    b = make_run_dir(tmp_path, "gen-internal", now=NOW)
    assert a.name == "20261018T130501Z_gen-internal"
    assert b.name == "20261018T130501Z_gen-internal-1"
    assert a.is_dir() and b.is_dir()


def test_environment_info_has_reproducibility_fields(tmp_path: Path) -> None:
    info = environment_info(seed=13, command="x")
    assert info["seed"] == 13 and info["command"] == "x"
    assert info["python"].startswith("3.12")
    assert info["packages"]["pydantic"] and "litellm" in info["packages"]
    assert len(info["git_sha"] or "x" * 40) == 40
    write_json(tmp_path / "env.json", info)
    assert json.loads((tmp_path / "env.json").read_text(encoding="utf-8"))["seed"] == 13
