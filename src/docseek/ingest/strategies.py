"""Extra chunking strategies for the M5 ablation (``section`` lives in ``chunking.py``).

- ``fixed_256``: each document's sections are concatenated in order and cut into windows of
  ``size`` tokens with ``overlap`` tokens of overlap, ignoring headings. A window is attributed
  to the section holding most of its tokens, so it can still be scored at section level.
- ``section_ctx``: the ``section`` chunks with ``doc title > section path`` prepended to
  ``text_for_embedding`` only (``text`` is unchanged, so BM25 and display do not change).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

from docseek.core.ids import make_chunk_id
from docseek.core.models import Chunk, Section
from docseek.ingest.chunking import chunk_metadata

Offsets = Callable[[str], list[tuple[int, int]]]
_REGEX_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def regex_offsets(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _REGEX_TOKEN_RE.finditer(text)]


def hf_offsets(model_name: str) -> Offsets:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(model_name)
    tok.no_truncation()

    def offsets(text: str) -> list[tuple[int, int]]:
        return [tuple(o) for o in tok.encode(text, add_special_tokens=False).offsets]  # type: ignore[misc]

    return offsets


def fixed_chunks(
    sections: Sequence[Section],
    offsets: Offsets,
    *,
    size: int = 256,
    overlap: int = 32,
    strategy: str = "fixed_256",
) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    by_doc: defaultdict[str, list[Section]] = defaultdict(list)
    for s in sections:
        if s.text.strip():
            by_doc[s.doc_id].append(s)
    out: list[Chunk] = []
    counters: Counter[str] = Counter()
    for doc_sections in by_doc.values():
        text_parts: list[str] = []
        owner: list[Section] = []  # section of each token
        spans: list[tuple[int, int]] = []
        base = 0
        for s in doc_sections:
            for a, b in offsets(s.text):
                spans.append((base + a, base + b))
                owner.append(s)
            text_parts.append(s.text)
            base += len(s.text) + 2
        doc_text = "\n\n".join(text_parts)
        step = size - overlap
        for start in range(0, max(len(spans), 1), step):
            window = spans[start : start + size]
            if not window:
                break
            majority = Counter(o.section_id for o in owner[start : start + size]).most_common(1)
            sec = next(o for o in owner[start : start + size] if o.section_id == majority[0][0])
            piece = doc_text[window[0][0] : window[-1][1]]
            n = counters[sec.section_id]
            counters[sec.section_id] += 1
            out.append(
                Chunk(
                    chunk_id=make_chunk_id(sec.section_id, strategy, n),
                    section_id=sec.section_id,
                    strategy=strategy,
                    text=piece,
                    text_for_embedding=piece,
                    metadata=chunk_metadata(sec),
                    n_tokens=len(window),
                )
            )
            if start + size >= len(spans):
                break
    return out


def with_context_header(
    chunks: Sequence[Chunk], sections: Sequence[Section], strategy: str = "section_ctx"
) -> list[Chunk]:
    by_id = {s.section_id: s for s in sections}
    out: list[Chunk] = []
    for c in chunks:
        s = by_id[c.section_id]
        header = " > ".join(s.section_path)
        n = int(c.chunk_id.rsplit("::", 1)[1])
        out.append(
            c.model_copy(
                update={
                    "chunk_id": make_chunk_id(c.section_id, strategy, n),
                    "strategy": strategy,
                    "text_for_embedding": f"{header}\n\n{c.text}",
                }
            )
        )
    return out
