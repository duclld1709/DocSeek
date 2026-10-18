"""Knowledge-graph schema (PROJECT_SPEC §6 M6), mirrored in ``docs/architecture.md``.

Nodes::

    (:Page {doc_id, title, doc_type, source, acl})
    (:Section {section_id, doc_id, title, path, acl, status})
    (:APIElement {name, kind})          // Depends, HTTPException, BackgroundTasks, ...
    (:Concept {name})                   // dependency injection, oauth2, cors, ...
    (:CodeExample {path})               // docs_src/*.py
    (:Version {name, date})             // from release-notes.md
    (:KBDoc {doc_id, acl, status, updated_at, version})

Edges::

    (Section)-[:PART_OF]->(Page)        (Page)-[:LINKS_TO]->(Page)
    (Section)-[:MENTIONS]->(APIElement|Concept)
    (Section)-[:SHOWS]->(CodeExample)   (CodeExample)-[:USES]->(APIElement)
    (APIElement)-[:CHANGED_IN]->(Version)
    (KBDoc)-[:SUPERSEDES]->(KBDoc)      (KBDoc)-[:REFERENCES]->(APIElement|Concept)

``acl`` is copied onto Page and Section so every query can filter by the principal's groups
inside Cypher (ACL is never left to the caller).
"""

from __future__ import annotations

NODE_KEYS: dict[str, str] = {
    "Page": "doc_id",
    "Section": "section_id",
    "APIElement": "name",
    "Concept": "name",
    "CodeExample": "path",
    "Version": "name",
    "KBDoc": "doc_id",
}

CONSTRAINTS: tuple[str, ...] = tuple(
    f"CREATE CONSTRAINT {label.lower()}_{key} IF NOT EXISTS "
    f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
    for label, key in NODE_KEYS.items()
)

EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("Section", "PART_OF", "Page"),
    ("Page", "LINKS_TO", "Page"),
    ("Section", "MENTIONS", "APIElement"),
    ("Section", "MENTIONS", "Concept"),
    ("Section", "SHOWS", "CodeExample"),
    ("CodeExample", "USES", "APIElement"),
    ("APIElement", "CHANGED_IN", "Version"),
    ("KBDoc", "SUPERSEDES", "KBDoc"),
    ("KBDoc", "REFERENCES", "APIElement"),
    ("KBDoc", "REFERENCES", "Concept"),
)
