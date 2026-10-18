"""Build lexical retrievers from experiment-config options.

Options (all optional)::

    k1: 1.2            b: 0.75
    fields: {all: 1.0}                    # or {title: 2, section_path: 1, body: 1, code: 0.5}
    tokenizer: {lowercase: true, stopwords: true, stem: true, code_aware: true}
    index_dir: data/index/bm25            # null disables the on-disk cache
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from docseek.core.acl import AccessPolicy
from docseek.core.models import Chunk
from docseek.lexical.bm25 import BM25Scorer
from docseek.lexical.inverted_index import ALL_FIELD, chunk_fields, load_or_build
from docseek.lexical.retriever import Bm25sScorer, LexicalRetriever
from docseek.lexical.tfidf import TfidfScorer
from docseek.lexical.tokenizer import Tokenizer, TokenizerConfig

DEFAULT_INDEX_DIR = Path("data/index/bm25")


def _tokenizer(opts: Mapping[str, Any]) -> TokenizerConfig:
    return TokenizerConfig(**dict(opts.get("tokenizer") or {}))


def _index_dir(opts: Mapping[str, Any]) -> Path | None:
    value = opts.get("index_dir", DEFAULT_INDEX_DIR)
    return Path(value) if value else None


def _tag(tok: TokenizerConfig) -> str:
    off = [name for name, on in tok.to_dict().items() if not on]
    return "" if not off else "-no_" + "_".join(off)


def build_bm25(
    opts: Mapping[str, Any], chunks: Sequence[Chunk], policy: AccessPolicy
) -> LexicalRetriever:
    k1, b = float(opts.get("k1", 1.2)), float(opts.get("b", 0.75))
    weights = {str(f): float(w) for f, w in dict(opts.get("fields") or {ALL_FIELD: 1.0}).items()}
    tok = _tokenizer(opts)
    index = load_or_build(
        chunks, fields=tuple(sorted(weights)), tokenizer=tok, cache_dir=_index_dir(opts)
    )
    scorer = BM25Scorer(index, k1=k1, b=b, field_weights=weights)
    kind = "bm25" if list(weights) == [ALL_FIELD] else "bm25f"
    return LexicalRetriever(f"{kind}-k1={k1}-b={b}{_tag(tok)}", chunks, scorer, tok, policy)


def build_tfidf(
    opts: Mapping[str, Any], chunks: Sequence[Chunk], policy: AccessPolicy
) -> LexicalRetriever:
    tok = _tokenizer(opts)
    index = load_or_build(chunks, fields=(ALL_FIELD,), tokenizer=tok, cache_dir=_index_dir(opts))
    return LexicalRetriever(f"tfidf{_tag(tok)}", chunks, TfidfScorer(index), tok, policy)


def build_bm25s(
    opts: Mapping[str, Any], chunks: Sequence[Chunk], policy: AccessPolicy
) -> LexicalRetriever:
    k1, b = float(opts.get("k1", 1.2)), float(opts.get("b", 0.75))
    tok = _tokenizer(opts)
    tokenizer = Tokenizer(tok)
    terms = [tokenizer.tokenize(chunk_fields(c)[ALL_FIELD]) for c in chunks]
    scorer = Bm25sScorer(terms, k1=k1, b=b)
    return LexicalRetriever(f"bm25s-k1={k1}-b={b}{_tag(tok)}", chunks, scorer, tok, policy)
