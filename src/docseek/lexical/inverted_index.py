"""Field-aware inverted index: postings, document frequency and field lengths.

Each indexed document (a chunk) is split into fields (:func:`chunk_fields`). With
``fields=("all",)`` the index is a classic single-field index over the whole chunk.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from docseek.core.models import Chunk
from docseek.lexical.tokenizer import Tokenizer, TokenizerConfig

FIELDS: tuple[str, ...] = ("title", "section_path", "body", "code")
ALL_FIELD = "all"
_FENCE = re.compile(r"^(`{3,}|~{3,})[^\n]*\n(.*?)^\1[ \t]*$", re.MULTILINE | re.DOTALL)

IntArray = npt.NDArray[np.int32]
F32Array = npt.NDArray[np.float32]


def chunk_fields(chunk: Chunk) -> dict[str, str]:
    """Split a chunk into BM25F fields.

    - ``title``: the section title.
    - ``section_path``: ancestor headings (the path without the title itself).
    - ``body``: prose, i.e. the text with fenced code blocks removed (inline code stays).
    - ``code``: the contents of fenced code blocks.
    - ``all``: path + text, for single-field indexes.
    """
    meta = chunk.metadata
    title = str(meta.get("title") or "")
    path = [p.strip() for p in str(meta.get("section_path") or "").split(">") if p.strip()]
    ancestors = path[:-1] if path and path[-1] == title else path
    code = "\n".join(m.group(2) for m in _FENCE.finditer(chunk.text))
    body = _FENCE.sub("\n", chunk.text)
    return {
        "title": title,
        "section_path": " ".join(ancestors),
        "body": body,
        "code": code,
        ALL_FIELD: " ".join([*path, chunk.text]) if path else chunk.text,
    }


@dataclass
class Postings:
    docs: IntArray
    tfs: F32Array


@dataclass
class InvertedIndex:
    """Postings per field; ``df`` counts documents containing a term in any field."""

    doc_ids: list[str]
    fields: tuple[str, ...]
    tokenizer: TokenizerConfig
    postings: dict[str, dict[str, Postings]]
    lengths: dict[str, F32Array]
    df: dict[str, int]
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def avg_length(self, f: str) -> float:
        lengths = self.lengths[f]
        return float(lengths.mean()) if lengths.size else 0.0

    @classmethod
    def build(
        cls,
        doc_ids: Sequence[str],
        docs: Sequence[Mapping[str, str]],
        *,
        fields: Sequence[str],
        tokenizer: TokenizerConfig,
    ) -> InvertedIndex:
        """Index ``docs`` (field name → text); ``doc_ids[i]`` names ``docs[i]``."""
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length")
        tok = Tokenizer(tokenizer)
        raw: dict[str, dict[str, tuple[list[int], list[float]]]] = {f: {} for f in fields}
        lengths = {f: np.zeros(len(docs), dtype=np.float32) for f in fields}
        df: Counter[str] = Counter()
        for i, doc in enumerate(docs):
            seen: set[str] = set()
            for f in fields:
                terms = tok.tokenize(doc.get(f, ""))
                lengths[f][i] = len(terms)
                for term, tf in Counter(terms).items():
                    d, t = raw[f].setdefault(term, ([], []))
                    d.append(i)
                    t.append(float(tf))
                    seen.add(term)
            df.update(seen)
        postings = {
            f: {
                term: Postings(np.asarray(d, dtype=np.int32), np.asarray(t, dtype=np.float32))
                for term, (d, t) in by_term.items()
            }
            for f, by_term in raw.items()
        }
        return cls(list(doc_ids), tuple(fields), tokenizer, postings, lengths, dict(df))

    @classmethod
    def from_chunks(
        cls, chunks: Iterable[Chunk], *, fields: Sequence[str], tokenizer: TokenizerConfig
    ) -> InvertedIndex:
        chunk_list = list(chunks)
        return cls.build(
            [c.chunk_id for c in chunk_list],
            [chunk_fields(c) for c in chunk_list],
            fields=fields,
            tokenizer=tokenizer,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> InvertedIndex:
        with path.open("rb") as fh:
            index = pickle.load(fh)  # our own cache file, never user-supplied
        if not isinstance(index, InvertedIndex):
            raise TypeError(f"{path} does not contain an InvertedIndex")
        return index


def index_key(chunks: Sequence[Chunk], fields: Sequence[str], tokenizer: TokenizerConfig) -> str:
    """Content hash of everything an index depends on; the cache file name."""
    h = hashlib.sha256()
    h.update(json.dumps({"fields": list(fields), "tok": tokenizer.to_dict(), "v": 1}).encode())
    for c in chunks:
        h.update(c.model_dump_json(include={"chunk_id", "text", "metadata"}).encode())
    return h.hexdigest()[:16]


def load_or_build(
    chunks: Sequence[Chunk],
    *,
    fields: Sequence[str],
    tokenizer: TokenizerConfig,
    cache_dir: Path | None,
) -> InvertedIndex:
    """Build the index, reusing ``cache_dir/<hash>.pkl`` when inputs are unchanged."""
    if cache_dir is None:
        return InvertedIndex.from_chunks(chunks, fields=fields, tokenizer=tokenizer)
    path = cache_dir / f"{index_key(chunks, fields, tokenizer)}.pkl"
    if path.is_file():
        return InvertedIndex.load(path)
    index = InvertedIndex.from_chunks(chunks, fields=fields, tokenizer=tokenizer)
    index.save(path)
    return index
