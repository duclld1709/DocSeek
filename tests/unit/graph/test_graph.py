"""Graph extraction and retriever logic without Neo4j (a fake Cypher runner)."""

from typing import Any

from docseek.core.acl import AccessPolicy
from docseek.core.config import load_principals
from docseek.core.models import Query
from docseek.graph.extract import (
    api_mentions,
    build_graph,
    load_code_files,
    normalize_api,
    normalize_concept,
    parse_code_file,
    parse_release_notes,
)
from docseek.graph.graph_retriever import TEMPLATES, EntityLinker, GraphRetriever
from docseek.testing.acl_checks import probe_leaks
from docseek.testing.fixtures import FIXTURES, REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))
RESULT = mini_ingest()
VOCAB = {n.lower(): n for n in ("Depends", "BackgroundTasks", "HTTPException", "Query")}


def test_ast_imports_and_uses() -> None:
    src = (
        "from fastapi import Depends, FastAPI\n"
        "from fastapi.security import OAuth2PasswordBearer\n"
        "import os\napp = FastAPI()\n"
    )
    f = parse_code_file("docs_src/x.py", src)
    assert f is not None
    assert f.imports == {
        "Depends": "fastapi",
        "FastAPI": "fastapi",
        "OAuth2PasswordBearer": "fastapi.security",
    }
    assert f.used == {"FastAPI"}  # imported but unused names are not USES
    assert parse_code_file("bad.py", "def (:") is None


def test_name_normalisation_and_mentions() -> None:
    assert normalize_api("fastapi.Depends()") == "Depends"
    assert normalize_api("@app.get") == "app.get"
    assert normalize_concept("Dependencies  Injection") == "dependency injection"
    assert normalize_concept("OAuth2 Scopes") == "oauth2 scope"
    text = "Use `Depends()` and BackgroundTasks; a Query here is prose, `Query` is code."
    assert api_mentions(text, VOCAB) == {"Depends", "BackgroundTasks", "Query"}
    assert api_mentions("A Query parameter", VOCAB) == set()  # single words need backticks


def test_release_notes_versions() -> None:
    notes = "## Latest Changes\n* x\n## 0.2.0 (2026-01-02)\n* Fix `HTTPException`.\n## 0.1.0\n* y\n"
    assert parse_release_notes(notes, VOCAB) == [
        ("0.2.0", "2026-01-02", {"HTTPException"}),
        ("0.1.0", None, set()),
    ]


def test_build_graph_on_fixture_corpus() -> None:
    root = FIXTURES / "mini_corpus"
    g = build_graph(
        RESULT.sections,
        load_code_files(root / "docs_src", root),
        concept_fn=lambda title, text: ["Dependency Injection", "x"],
        max_concept_pages=50,
    )
    c = g.counts()
    assert c["Section"] == len(RESULT.sections) and c["Section-PART_OF->Page"] == c["Section"]
    assert c["KBDoc-SUPERSEDES->KBDoc"] == 1  # kb-0003 supersedes kb-0002
    assert "Depends" in g.nodes["APIElement"] and c["CodeExample-USES->APIElement"] > 0
    assert set(g.nodes["Concept"]) == {"dependency injection"}  # "x" is too short
    assert any(e[2] == "MENTIONS" and e[3] == "Concept" for e in g.edges)
    assert all(n["acl"] for n in g.nodes["Section"].values())


def test_entity_linking() -> None:
    linker = EntityLinker(["BackgroundTasks", "Depends"], ["dependency injection", "api"])
    assert linker.link("use backgroundtasks with Depends()") == ["BackgroundTasks", "Depends"]
    assert linker.link("BackgroundTask") == ["BackgroundTasks"]  # fuzzy
    assert linker.link("how does dependency injection work") == ["dependency injection"]
    assert linker.link("an api question") == []  # short single-word concepts are ignored


class FakeRunner:
    """Returns every section for every template, ignoring $groups (a broken template)."""

    def __init__(self, section_ids: list[str]) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._rows = [{"section_id": s, "n": 1, "via": ["Depends"]} for s in section_ids]

    def __call__(self, cypher: str, params: Any) -> list[dict[str, Any]]:
        self.calls.append((cypher, dict(params)))
        return self._rows


def test_templates_are_constant_and_parameterised() -> None:
    runner = FakeRunner([c.section_id for c in RESULT.chunks])
    r = GraphRetriever(runner, EntityLinker(["Depends"], []), RESULT.chunks, POLICY)
    evil = 'Depends"}) MATCH (n) DETACH DELETE n //'
    r.search(Query(qid="x", text=evil), 5)
    assert runner.calls
    for cypher, params in runner.calls:
        assert cypher in TEMPLATES.values()  # never built from input
        assert "DETACH" not in cypher and evil not in cypher
        assert set(params) == {"names", "groups", "limit"}
        assert params["names"] == ["Depends"]  # only linked node names reach Cypher
    for cypher in TEMPLATES.values():
        assert "$groups" in cypher and "$names" in cypher


def test_acl_is_rechecked_even_if_cypher_ignores_groups() -> None:
    runner = FakeRunner([c.section_id for c in RESULT.chunks])
    r = GraphRetriever(runner, EntityLinker(["Depends"], []), RESULT.chunks, POLICY)
    acl = {c.chunk_id: c.metadata["acl"] for c in RESULT.chunks}
    res = probe_leaks(r, acl, POLICY, queries=["Depends"], k=500)
    assert res.leaks == [] and res.restricted_seen["engineer"] > 0
    assert r.search(Query(qid="y", text="nothing linkable here"), 5) == []
