# ADR 0006: The graph is queried only through fixed, parameterised Cypher templates

- **Status:** Accepted (M6, 2026-10-19)
- **Deciders:** project owner (PROJECT_SPEC §6 M6, §7)

## Context

Multi-hop questions can be answered by walking the knowledge graph. One option is to let an
LLM write Cypher from the question ("text-to-Cypher"); the other is a small set of query
shapes written by us, with the question contributing only entity names.

## Decision

Only the templates in `src/docseek/graph/graph_retriever.py` run: `shared_mentions`,
`code_examples`, `kb_references`, `linked_pages` (section retrieval) and `API_CHANGES`
(versions of an API, for tools). They are string constants; query text never reaches Cypher.
Entity linking maps the text to existing node names, and only those names, the principal's
ACL groups and a limit are passed as `$parameters`. Every section-returning template filters
`any(g IN s.acl WHERE g IN $groups)` (and `KBDoc.acl` where KB docs are joined); the retriever
re-checks ACL on chunk metadata. Labels and relationship types in the loader come from the
fixed schema, never from data.

## Consequences

- No Cypher injection surface: a unit test sends a query containing Cypher and asserts the
  executed statements are exactly the templates with the expected parameters.
- ACL cannot be bypassed by a crafted question, and a template bug is caught by the
  post-filter (unit test with a runner that ignores `$groups`).
- Coverage is limited to the shapes we wrote; new question types need a new template, a test
  on the fixture graph (`tests/integration/test_graph.py`) and an eval run.
- Evaluation is meaningful: each template is a deterministic, inspectable retriever, and every
  hit carries the template(s) and entities that produced it.
