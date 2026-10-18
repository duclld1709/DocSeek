"""Cypher templates on a fixture graph in a real Neo4j (``-m integration``).

The test wipes the database before loading the fixture graph, so it only runs when
``DOCSEEK_NEO4J_WIPE_OK=1`` (always true in CI's throwaway service container). Locally,
rebuild the real graph afterwards with ``docseek graph build`` (LLM concepts are cached).
"""

import os

import pytest

from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.core.models import Query
from docseek.graph.extract import build_graph, load_code_files
from docseek.graph.graph_retriever import (
    API_CHANGES,
    TEMPLATES,
    EntityLinker,
    GraphRetriever,
    connect_neo4j,
    neo4j_runner,
)
from docseek.graph.loader import graph_counts, load_graph
from docseek.testing.acl_checks import probe_leaks
from docseek.testing.fixtures import FIXTURES, REPO_ROOT, mini_ingest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("DOCSEEK_NEO4J_WIPE_OK") != "1",
        reason="wipes Neo4j; set DOCSEEK_NEO4J_WIPE_OK=1",
    ),
]

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
RESULT = mini_ingest()
NOTES = "## 0.2.0 (2026-01-02)\n* Fix `Depends` caching.\n"


@pytest.fixture(scope="module")
def retriever() -> GraphRetriever:
    root = FIXTURES / "mini_corpus"
    g = build_graph(
        RESULT.sections,
        load_code_files(root / "docs_src", root),
        release_notes=NOTES,
        concept_fn=lambda t, x: ["timeout", "dependency injection"],
        max_concept_pages=50,
    )
    driver = connect_neo4j()
    load_graph(driver, g, wipe=True)
    load_graph(driver, g)  # idempotent: MERGE does not duplicate
    assert graph_counts(driver)["Section"] == len(RESULT.sections)
    run = neo4j_runner(driver)
    names = [r["name"] for r in run("MATCH (e:APIElement) RETURN e.name AS name", {})]
    linker = EntityLinker(names, ["timeout", "dependency injection"])
    return GraphRetriever(run, linker, RESULT.chunks, POLICY)


def test_every_template_runs_and_filters_acl(retriever: GraphRetriever) -> None:
    run = retriever._run
    for name, cypher in TEMPLATES.items():
        params = {"names": ["Depends", "timeout"], "groups": ["public"], "limit": 50}
        rows = run(cypher, params)
        sids = {r["section_id"] for r in rows}
        assert not any(s.startswith("internal-kb:kb-0002") for s in sids), name
        assert not any(s.startswith("internal-kb:kb-0004") for s in sids), name
    assert run(API_CHANGES, {"name": "Depends", "limit": 5}) == [
        {"version": "0.2.0", "date": "2026-01-02"}
    ]


def test_kb_references_skip_superseded_docs(retriever: GraphRetriever) -> None:
    rows = retriever._run(
        TEMPLATES["kb_references"],
        {"names": ["timeout"], "groups": ["public", "eng-internal"], "limit": 50},
    )
    sids = {r["section_id"] for r in rows}
    assert any("kb-0003" in s for s in sids)
    assert not any("kb-0002" in s for s in sids)  # superseded by kb-0003


def test_graph_retriever_never_leaks(retriever: GraphRetriever) -> None:
    acl = {c.chunk_id: c.metadata["acl"] for c in RESULT.chunks}
    res = probe_leaks(retriever, acl, POLICY, queries=["timeout", "Depends", "JWT"])
    assert res.leaks == [] and res.restricted_seen["engineer"] > 0
    hits = retriever.search(Query(qid="q", text="Depends with dependency injection"), 5)
    assert hits and hits[0].debug["via"]
