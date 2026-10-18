"""Graph retriever: link query entities to graph nodes, run fixed Cypher templates.

Only the templates below ever run (ADR 0006): they are constants, user input reaches them
solely as ``$parameters``, and every template filters sections by ``$groups`` (the
principal's ACL groups) inside Cypher. Hits carry the graph path that produced them in
``debug["via"]`` as an explanation.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from docseek.core.acl import AccessPolicy
from docseek.core.models import Chunk, Query, ScoredHit
from docseek.graph.extract import normalize_api, normalize_concept

# name -> Cypher. Parameters: $names (linked entity names), $groups, $limit.
TEMPLATES: dict[str, str] = {
    # Sections that mention several of the query's entities (multi-hop intersection).
    "shared_mentions": """
        MATCH (s:Section)-[:MENTIONS]->(e)
        WHERE e.name IN $names AND any(g IN s.acl WHERE g IN $groups)
        WITH s, collect(DISTINCT e.name) AS via
        RETURN s.section_id AS section_id, size(via) AS n, via
        ORDER BY n DESC, section_id LIMIT $limit""",
    # Sections whose code examples use the linked API elements.
    "code_examples": """
        MATCH (s:Section)-[:SHOWS]->(:CodeExample)-[:USES]->(e:APIElement)
        WHERE e.name IN $names AND any(g IN s.acl WHERE g IN $groups)
        WITH s, collect(DISTINCT e.name) AS via
        RETURN s.section_id AS section_id, size(via) AS n, via
        ORDER BY n DESC, section_id LIMIT $limit""",
    # Current (not superseded) internal KB docs that reference the entities.
    "kb_references": """
        MATCH (k:KBDoc)-[:REFERENCES]->(e)
        WHERE e.name IN $names AND any(g IN k.acl WHERE g IN $groups)
          AND NOT EXISTS { MATCH (:KBDoc)-[:SUPERSEDES]->(k) }
        MATCH (s:Section {doc_id: k.doc_id})-[:MENTIONS]->(e)
        WHERE any(g IN s.acl WHERE g IN $groups)
        WITH s, collect(DISTINCT e.name) AS via
        RETURN s.section_id AS section_id, size(via) AS n, via
        ORDER BY n DESC, section_id LIMIT $limit""",
    # One hop over page links: sections of linked pages that also mention an entity.
    "linked_pages": """
        MATCH (s0:Section)-[:MENTIONS]->(e)
        WHERE e.name IN $names AND any(g IN s0.acl WHERE g IN $groups)
        MATCH (s0)-[:PART_OF]->(:Page)-[:LINKS_TO]-(:Page)<-[:PART_OF]-(s:Section)-[:MENTIONS]->(e2)
        WHERE e2.name IN $names AND any(g IN s.acl WHERE g IN $groups)
        WITH s, collect(DISTINCT e2.name) AS via
        RETURN s.section_id AS section_id, size(via) AS n, via
        ORDER BY n DESC, section_id LIMIT $limit""",
}
# Not a section retriever: used by tools (M8) to answer "when did X change".
API_CHANGES = """
    MATCH (e:APIElement {name: $name})-[:CHANGED_IN]->(v:Version)
    RETURN v.name AS version, v.date AS date ORDER BY v.date DESC LIMIT $limit"""

WEIGHTS: dict[str, float] = {
    "shared_mentions": 1.0,
    "code_examples": 0.7,
    "kb_references": 0.8,
    "linked_pages": 0.3,
}

_IDENT = re.compile(r"@?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?")


class CypherRunner(Protocol):
    def __call__(self, cypher: str, params: Mapping[str, Any]) -> list[dict[str, Any]]: ...


def neo4j_runner(driver: Any, database: str | None = None) -> CypherRunner:
    def run(cypher: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        with driver.session(database=database) as session:
            return [dict(r) for r in session.run(cypher, dict(params))]

    return run


def _compound(name: str) -> bool:
    return "_" in name or bool(re.search(r"[a-z][A-Z]|[A-Z]{2,}[a-z]", name))


class EntityLinker:
    """Exact (case-insensitive) and fuzzy matching of query text to graph node names."""

    def __init__(self, api_names: Iterable[str], concepts: Iterable[str], *, fuzzy: float = 0.88):
        self._api = {n.lower(): n for n in api_names}
        self._concepts = sorted({c for c in concepts if len(c) >= 4 or " " in c})
        self._fuzzy = fuzzy

    def link(self, text: str) -> list[str]:
        found: list[str] = []
        for tok in _IDENT.findall(text):
            norm = normalize_api(tok)
            key = norm.lower()
            if key in self._api:
                name = self._api[key]
                # A one-word API name (Header, Path, Security) also is an English word: only
                # link it when written with the API's exact casing.
                if _compound(name) or norm == name:
                    found.append(name)
            elif len(key) >= 6:
                close = difflib.get_close_matches(key, list(self._api), n=1, cutoff=self._fuzzy)
                if close:
                    found.append(self._api[close[0]])
        norm = f" {normalize_concept(text)} "
        found += [c for c in self._concepts if f" {c} " in norm]
        return list(dict.fromkeys(found))


class GraphRetriever:
    name = "graph"

    def __init__(
        self,
        run: CypherRunner,
        linker: EntityLinker,
        chunks: Sequence[Chunk],
        policy: AccessPolicy,
        *,
        per_template: int = 30,
        templates: Sequence[str] = tuple(TEMPLATES),
        min_entities: int = 1,
    ) -> None:
        self._min_entities = min_entities
        self._run = run
        self._linker = linker
        self._policy = policy
        self._per_template = per_template
        self._templates = list(templates)
        self._first_chunk: dict[str, Chunk] = {}
        for c in sorted(chunks, key=lambda c: c.chunk_id):
            self._first_chunk.setdefault(c.section_id, c)

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        groups = sorted(self._policy.groups(query.principal))  # unknown principal raises
        names = self._linker.link(query.text)
        if len(names) < max(1, self._min_entities):
            return []  # e.g. min_entities=2: only multi-entity (multi-hop style) queries
        score: defaultdict[str, float] = defaultdict(float)
        via: defaultdict[str, dict[str, list[str]]] = defaultdict(dict)
        for t in self._templates:
            params = {"names": names, "groups": groups, "limit": self._per_template}
            for row in self._run(TEMPLATES[t], params):
                sid = row["section_id"]
                score[sid] += WEIGHTS[t] * row["n"]
                via[sid][t] = list(row["via"])
        # Defence in depth: Cypher already filtered by $groups; re-check with chunk metadata.
        readable = [
            sid
            for sid in score
            if sid in self._first_chunk
            and self._policy.can_read(query.principal, self._first_chunk[sid].metadata["acl"])
        ]
        ranked = sorted(readable, key=lambda s: (-score[s], s))[:k]
        return [
            ScoredHit(
                chunk_id=self._first_chunk[sid].chunk_id,
                section_id=sid,
                score=score[sid],
                rank=i,
                retriever=self.name,
                debug={"entities": names, "via": via[sid]},
            )
            for i, sid in enumerate(ranked, start=1)
        ]


def connect_neo4j() -> Any:
    from neo4j import GraphDatabase

    from docseek.core.config import Settings

    st = Settings()
    return GraphDatabase.driver(
        st.neo4j_uri, auth=(st.neo4j_user, st.neo4j_password.get_secret_value())
    )


def build_graph_retriever(
    opts: Mapping[str, Any], chunks: Sequence[Chunk], policy: AccessPolicy
) -> GraphRetriever:
    """Runner factory; options: per_template (30), templates (all), fuzzy (0.88)."""
    run = neo4j_runner(connect_neo4j())
    apis = [r["name"] for r in run("MATCH (e:APIElement) RETURN e.name AS name", {})]
    concepts = [r["name"] for r in run("MATCH (c:Concept) RETURN c.name AS name", {})]
    if not apis:
        raise RuntimeError("the graph is empty; run `docseek graph build` first")
    linker = EntityLinker(apis, concepts, fuzzy=float(opts.get("fuzzy", 0.88)))
    return GraphRetriever(
        run,
        linker,
        chunks,
        policy,
        per_template=int(opts.get("per_template", 30)),
        templates=list(opts.get("templates") or TEMPLATES),
        min_entities=int(opts.get("min_entities", 1)),
    )
