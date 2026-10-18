"""Reusable ACL check: run probe queries as every principal and collect leaked hits.

Every retriever and tool test calls :func:`probe_leaks` and asserts the result is empty, and
also asserts that privileged principals *do* see restricted content so the check is not
vacuous (a retriever that returns nothing never leaks).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from docseek.core.acl import AccessPolicy
from docseek.core.interfaces import Retriever
from docseek.core.models import Query

# Queries aimed at the restricted fixture docs (tests/fixtures/mini_kb) plus generic ones.
PROBE_QUERIES: tuple[str, ...] = (
    "Payment API timeouts runbook",
    "JWT signing key rotation",
    "security policy signing keys",
    "local development setup",
    "dependencies with yield",
    "BackgroundTasks",
)


@dataclass(frozen=True)
class ProbeResult:
    leaks: list[tuple[str, str, str]]  # (principal, query, chunk_id)
    restricted_seen: dict[str, int]  # principal -> hits whose ACL is not just public


def probe_leaks(
    retriever: Retriever,
    chunk_acl: Mapping[str, Sequence[str]],
    policy: AccessPolicy,
    principals: Iterable[str] = ("anonymous", "engineer", "security"),
    queries: Iterable[str] = PROBE_QUERIES,
    k: int = 10,
) -> ProbeResult:
    leaks: list[tuple[str, str, str]] = []
    restricted: dict[str, int] = {}
    query_list = list(queries)
    for principal in principals:
        restricted[principal] = 0
        for i, text in enumerate(query_list):
            hits = retriever.search(Query(qid=f"probe{i}", text=text, principal=principal), k)
            for h in hits:
                acl = chunk_acl[h.chunk_id]
                if not policy.can_read(principal, acl):
                    leaks.append((principal, text, h.chunk_id))
                if set(acl) != {"public"}:
                    restricted[principal] += 1
    return ProbeResult(leaks, restricted)
