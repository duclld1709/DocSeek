"""Evaluate a :class:`MetadataFilter` against a chunk's flat metadata payload.

Used by retrievers that filter in Python (lexical, graph, FS). Vector stores translate the
same filter into their native query language instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from docseek.core.models import MetadataFilter


def as_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def matches_filter(metadata: Mapping[str, Any], flt: MetadataFilter | None) -> bool:
    """True when ``metadata`` satisfies every constraint set in ``flt``.

    Items without ``updated_at`` fail any date constraint (they cannot be shown to be fresh).
    """
    if flt is None:
        return True
    for key in ("doc_type", "source", "modality", "status"):
        allowed = getattr(flt, key)
        if allowed is not None and metadata.get(key) not in allowed:
            return False
    if flt.version is not None and metadata.get("version") != flt.version:
        return False
    if flt.updated_after is not None or flt.updated_before is not None:
        updated = as_datetime(metadata.get("updated_at"))
        if updated is None:
            return False
        if flt.updated_after is not None and updated < _aware(flt.updated_after):
            return False
        if flt.updated_before is not None and updated > _aware(flt.updated_before):
            return False
    return True
