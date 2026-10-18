"""Spend tracking on real (uncached) LLM calls.

- ``BudgetGuard``: cap for one run (``DOCSEEK_BUDGET_USD``), in memory.
- ``SpendLedger``: every paid call of every run, persisted in SQLite next to the response
  cache, with caps per provider over the whole project (``spend_limits_usd`` in
  ``configs/models.yaml``). Costs are LiteLLM list prices; free-tier calls are recorded at
  list price too but not charged by the provider.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path


class BudgetExceededError(RuntimeError):
    """Raised before a call once the run has spent its budget (``DOCSEEK_BUDGET_USD``)."""


class BudgetGuard:
    """Accumulates spend for one run and refuses new calls once ``limit_usd`` is reached.

    The check happens *before* each real call, so a run can overshoot by at most the cost of
    the calls already in flight; that is the price of not guessing costs up front.
    """

    def __init__(self, limit_usd: float) -> None:
        if limit_usd <= 0:
            raise ValueError("limit_usd must be > 0")
        self.limit_usd = limit_usd
        self._spent = 0.0
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def check(self) -> None:
        with self._lock:
            if self._spent >= self.limit_usd:
                raise BudgetExceededError(
                    f"LLM budget exhausted: spent ${self._spent:.4f} of ${self.limit_usd:.2f} "
                    "(raise DOCSEEK_BUDGET_USD to continue; cached calls stay free)"
                )

    def add(self, cost_usd: float) -> None:
        if cost_usd < 0:
            raise ValueError("cost must be >= 0")
        with self._lock:
            self._spent += cost_usd


class SpendLedger:
    """Append-only record of paid calls; thread-safe, one SQLite file shared across runs."""

    def __init__(self, path: Path) -> None:
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS spend ("
                " id INTEGER PRIMARY KEY, at TEXT NOT NULL, provider TEXT NOT NULL,"
                " model TEXT NOT NULL, name TEXT NOT NULL, tokens_in INTEGER NOT NULL,"
                " tokens_out INTEGER NOT NULL, cost_usd REAL NOT NULL)"
            )

    def record(
        self, *, model: str, name: str, tokens_in: int, tokens_out: int, cost_usd: float
    ) -> None:
        provider = model.split("/", 1)[0]
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO spend (at, provider, model, name, tokens_in, tokens_out, cost_usd)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    provider,
                    model,
                    name,
                    tokens_in,
                    tokens_out,
                    cost_usd,
                ),
            )

    def total(self, provider: str | None = None) -> float:
        query = "SELECT COALESCE(SUM(cost_usd), 0) FROM spend"
        args: tuple[str, ...] = ()
        if provider is not None:
            query += " WHERE provider = ?"
            args = (provider,)
        with self._lock:
            (value,) = self._conn.execute(query, args).fetchone()
        return float(value)

    def summary(self) -> dict[str, dict[str, float]]:
        """Per provider: number of paid calls, tokens and list-price cost."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT provider, COUNT(*), SUM(tokens_in), SUM(tokens_out), SUM(cost_usd)"
                " FROM spend GROUP BY provider ORDER BY provider"
            ).fetchall()
        return {
            p: {"calls": n, "tokens_in": ti, "tokens_out": to, "cost_usd": c}
            for p, n, ti, to, c in rows
        }

    def check(self, model: str, limits: Mapping[str, float]) -> None:
        """Raise before a call whose provider already reached its project-wide limit."""
        provider = model.split("/", 1)[0]
        limit = limits.get(provider)
        if limit is not None and (spent := self.total(provider)) >= limit:
            raise BudgetExceededError(
                f"project spend limit for {provider} reached: ${spent:.4f} of ${limit:.2f} "
                "(spend_limits_usd in configs/models.yaml); cached calls stay free"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
