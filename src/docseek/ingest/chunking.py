"""Chunking strategies (PROJECT_SPEC §6 M1; more strategies arrive in M5).

``section``: one chunk per section. A section longer than ``max_tokens`` is split on
paragraph boundaries; consecutive chunks share up to ``overlap_tokens`` of trailing
paragraphs. Fenced code blocks are kept whole when they fit, otherwise split by lines and
re-fenced so every piece is still valid markdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from docseek.core.ids import make_chunk_id
from docseek.core.models import Chunk, Section
from docseek.ingest.markdown_parser import FENCE_RE, FenceTracker
from docseek.ingest.tokens import TokenCounter

SECTION_STRATEGY = "section"


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("need max_tokens > 0 and 0 <= overlap_tokens < max_tokens")


DEFAULT_CHUNKING = ChunkingConfig()


def chunk_metadata(section: Section) -> dict[str, Any]:
    """Flat payload for vector-store filters (only JSON scalars and lists of strings)."""
    return {
        "section_id": section.section_id,
        "doc_id": section.doc_id,
        "source": section.source,
        "title": section.title,
        "section_path": " > ".join(section.section_path),
        "doc_type": section.doc_type,
        "modality": section.modality,
        "acl": list(section.acl),
        "status": section.status,
        "version": section.version,
        "supersedes": section.supersedes,
        "updated_at": section.updated_at.isoformat() if section.updated_at else None,
    }


def chunk_section(
    section: Section, counter: TokenCounter, config: ChunkingConfig = DEFAULT_CHUNKING
) -> list[Chunk]:
    """Chunks of one section with the ``section`` strategy; empty sections give none."""
    if not section.text.strip():
        return []
    if counter.count(section.text) <= config.max_tokens:
        pieces = [section.text]
    else:
        pieces = _pack(_blocks(section.text, counter, config.max_tokens), counter, config)
    meta = chunk_metadata(section)
    return [
        Chunk(
            chunk_id=make_chunk_id(section.section_id, SECTION_STRATEGY, n),
            section_id=section.section_id,
            strategy=SECTION_STRATEGY,
            text=piece,
            text_for_embedding=piece,
            metadata=meta,
            n_tokens=counter.count(piece),
        )
        for n, piece in enumerate(pieces)
    ]


def chunk_sections(
    sections: list[Section], counter: TokenCounter, config: ChunkingConfig = DEFAULT_CHUNKING
) -> list[Chunk]:
    return [c for s in sections for c in chunk_section(s, counter, config)]


def _blocks(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Paragraphs and whole fenced code blocks; anything above ``max_tokens`` is split."""
    blocks: list[list[str]] = []
    current: list[str] = []
    fences = FenceTracker()
    for line in text.splitlines():
        was_inside = fences.inside
        is_fence_line = fences.feed(line)
        if not is_fence_line and not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        if is_fence_line and not was_inside and current:
            blocks.append(current)  # a code block starts its own block
            current = []
        current.append(line)
        if is_fence_line and was_inside and not fences.inside:
            blocks.append(current)  # code block closed
            current = []
    if current:
        blocks.append(current)

    out: list[str] = []
    for block in blocks:
        joined = "\n".join(block)
        if counter.count(joined) <= max_tokens:
            out.append(joined)
        else:
            out.extend(_split_long_block(block, counter, max_tokens))
    return out


def _split_long_block(lines: list[str], counter: TokenCounter, max_tokens: int) -> list[str]:
    fence = FENCE_RE.match(lines[0])
    is_code = fence is not None and len(lines) > 1 and FENCE_RE.match(lines[-1]) is not None
    if is_code:
        opener, closer = lines[0], lines[-1]
        body = lines[1:-1]
        budget = max_tokens - counter.count(f"{opener}\n{closer}")
    else:
        opener = closer = ""
        body = lines
        budget = max_tokens
    pieces: list[list[str]] = []
    current: list[str] = []
    for line in _split_long_lines(body, counter, budget):
        if current and counter.count("\n".join([*current, line])) > budget:
            pieces.append(current)
            current = []
        current.append(line)
    if current:
        pieces.append(current)
    if is_code:
        return ["\n".join([opener, *p, closer]) for p in pieces]
    return ["\n".join(p) for p in pieces]


def _split_long_lines(lines: list[str], counter: TokenCounter, budget: int) -> list[str]:
    """Break single lines that alone exceed ``budget`` into word runs."""
    out: list[str] = []
    for line in lines:
        if counter.count(line) <= budget:
            out.append(line)
            continue
        words: list[str] = []
        for word in line.split(" "):
            if words and counter.count(" ".join([*words, word])) > budget:
                out.append(" ".join(words))
                words = []
            words.append(word)
        if words:
            out.append(" ".join(words))
    return out


def _pack(blocks: list[str], counter: TokenCounter, config: ChunkingConfig) -> list[str]:
    """Greedy packing of blocks into chunks, carrying trailing blocks over as overlap."""
    sizes = [counter.count(b) for b in blocks]
    chunks: list[list[int]] = []
    current: list[int] = []
    for i, size in enumerate(sizes):
        if current and _size(current, sizes) + size > config.max_tokens:
            carried = _overlap(current, sizes, config.overlap_tokens, size, config.max_tokens)
            if len(carried) < len(current):  # otherwise the next chunk contains this one
                chunks.append(current)
            current = carried
        current.append(i)
    if current:
        chunks.append(current)
    return ["\n\n".join(blocks[i] for i in chunk) for chunk in chunks]


def _size(idx: list[int], sizes: list[int]) -> int:
    # Approximation: the "\n\n" joints add no tokens for WordPiece or regex counters.
    return sum(sizes[i] for i in idx)


def _overlap(
    previous: list[int], sizes: list[int], overlap: int, incoming: int, max_tokens: int
) -> list[int]:
    carried: list[int] = []
    total = 0
    for i in reversed(previous):
        if total + sizes[i] > overlap or total + sizes[i] + incoming > max_tokens:
            break
        carried.insert(0, i)
        total += sizes[i]
    return carried
