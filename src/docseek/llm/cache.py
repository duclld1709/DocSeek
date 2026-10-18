"""SQLite cache for LLM responses, keyed by a hash of everything that shapes the output."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from docseek.core.models import LLMResponse

# Bump when the key layout or the stored value format changes.
CACHE_KEY_VERSION = 1


def cache_key(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    response_format: type[BaseModel] | None,
) -> str:
    """sha256 of ``(model, messages, tools, temperature, response schema)`` (PROJECT_SPEC §4.3).

    The response schema is part of the key: the same prompt asked for a different structure
    must not return a stale shape.
    """
    payload = {
        "v": CACHE_KEY_VERSION,
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
        "response_format": response_format.model_json_schema() if response_format else None,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class SQLiteLLMCache:
    """Thread-safe key/value store of ``LLMResponse`` JSON in one SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                " key TEXT PRIMARY KEY, model TEXT NOT NULL, response TEXT NOT NULL,"
                " created_at TEXT NOT NULL DEFAULT (datetime('now')))"
            )

    def get(self, key: str) -> LLMResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
        return LLMResponse.model_validate_json(row[0]) if row else None

    def put(self, key: str, response: LLMResponse) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, model, response) VALUES (?, ?, ?)",
                (key, response.model, response.model_dump_json()),
            )

    def __len__(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        return int(n)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
