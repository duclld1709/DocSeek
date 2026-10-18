# mini_corpus

A tiny stand-in for the FastAPI docs (`docs/en/docs` + `docs_src`) with the same layout and
markdown dialect, used by unit tests, the CI eval gate and the quick demo. Content is
abridged and adapted from FastAPI's documentation (MIT License, (c) Sebastián Ramírez).

Layout mirrors upstream: code includes such as `{* ../../docs_src/x.py *}` resolve relative
to `docs/en/` (the mkdocs project directory), not relative to the markdown file.

Quirks covered on purpose (`tests/unit/test_mini_corpus.py` asserts each one is present):

| Quirk | Where |
|---|---|
| Heading anchors `## Title { #anchor }` | every page |
| Heading without anchor (H1 with inline code) | `reference/background.md` |
| `#` lines inside fenced code that are NOT headings | `tutorial/first-steps.md`, `deployment/concepts.md` |
| Code include, plain / `hl[...]` / `ln[...] hl[...]` | `tutorial/first-steps.md`, `tutorial/dependencies/*` |
| Variants `_py310` and `_an_py310` (prefer `_an_py310`) | `docs_src/dependencies`, `docs_src/security` |
| Admonitions `/// tip`, `/// note \| Title`, `/// warning`, `/// danger` | several |
| Tabs `//// tab \| Name` | `advanced/settings.md` |
| Inline HTML `<dfn>`, `<abbr>`, `<img>`, `<div class="termy">`, `<font>` | several |
| Markdown table | `tutorial/query-params.md` |
| Internal links, relative and with `#anchor` | several |
| Dangling internal link (target page not in corpus) | `tutorial/dependencies/dependencies-with-yield.md` |
| mkdocstrings directive `::: fastapi.X` | `reference/background.md` |
| YAML front matter | `release-notes.md` |
| Excluded files / dirs (`configs/corpus.yaml`) | `release-notes.md`, `fastapi-people.md`, `css/` |
| Hard-negative pair (similar but different) | `tutorial/query-params.md` vs `tutorial/path-params.md` |
| Multi-hop: OAuth2 tutorial + dependency with yield | `tutorial/security/first-steps.md` |
