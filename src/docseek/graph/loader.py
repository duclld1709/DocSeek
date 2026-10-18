"""Load :class:`GraphData` into Neo4j with idempotent, batched ``MERGE`` statements.

Labels and relationship types come from the fixed schema (``graph/schema.py``), never from
data, so the only interpolated identifiers are constants; all values are parameters.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from neo4j import Driver

from docseek.graph.extract import GraphData
from docseek.graph.schema import CONSTRAINTS, EDGE_TYPES, NODE_KEYS

BATCH = 1000


def _batches(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [rows[i : i + BATCH] for i in range(0, len(rows), BATCH)]


def load_graph(
    driver: Driver, g: GraphData, *, database: str | None = None, wipe: bool = False
) -> None:
    allowed_edges = {(a, r, b) for a, r, b in EDGE_TYPES}
    with driver.session(database=database) as session:
        if wipe:
            session.run("MATCH (n) DETACH DELETE n")
        for stmt in CONSTRAINTS:
            session.run(stmt)
        for label, nodes in g.nodes.items():
            key = NODE_KEYS[label]  # KeyError for labels outside the schema
            rows = [{"key": k, "props": props} for k, props in nodes.items()]
            for batch in _batches(rows):
                session.run(
                    f"UNWIND $rows AS row MERGE (n:{label} {{{key}: row.key}}) SET n += row.props",
                    rows=batch,
                )
        grouped: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for src_label, src_key, rel, dst_label, dst_key in g.edges:
            if (src_label, rel, dst_label) not in allowed_edges:
                raise ValueError(f"edge type not in schema: {src_label}-{rel}->{dst_label}")
            grouped[(src_label, rel, dst_label)].append({"s": src_key, "d": dst_key})
        for (src_label, rel, dst_label), pairs in grouped.items():
            sk, dk = NODE_KEYS[src_label], NODE_KEYS[dst_label]
            for batch in _batches(pairs):
                session.run(
                    f"UNWIND $rows AS row "
                    f"MATCH (a:{src_label} {{{sk}: row.s}}) MATCH (b:{dst_label} {{{dk}: row.d}}) "
                    f"MERGE (a)-[:{rel}]->(b)",
                    rows=batch,
                )


def graph_counts(driver: Driver, *, database: str | None = None) -> dict[str, int]:
    with driver.session(database=database) as session:
        nodes = session.run("MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c").data()
        rels = session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c").data()
    return {**{r["l"]: r["c"] for r in nodes}, **{r["t"]: r["c"] for r in rels}}
