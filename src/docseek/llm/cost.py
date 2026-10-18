"""Spend tracking: a hard per-run budget on real (uncached) LLM calls."""

from __future__ import annotations

import threading


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
