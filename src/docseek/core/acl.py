"""Access control shared by every retriever and tool (CLAUDE.md: enforce ACL in code, never
in prompts).

A principal may read an item when the item's ACL shares at least one group with the
principal's groups from ``configs/principals.yaml``. Unknown principals raise; items with an
empty ACL are never visible (fail closed).
"""

from __future__ import annotations

from collections.abc import Iterable

from docseek.core.config import PrincipalsConfig


class AccessPolicy:
    def __init__(self, principals: PrincipalsConfig) -> None:
        self._principals = principals

    def groups(self, principal: str) -> frozenset[str]:
        """Groups of ``principal``; raises ``UnknownPrincipalError`` if undeclared."""
        return self._principals.groups_for(principal)

    def can_read(self, principal: str, acl: Iterable[str]) -> bool:
        return not self.groups(principal).isdisjoint(acl)
