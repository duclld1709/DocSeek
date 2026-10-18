"""Configurable tokenizer for lexical retrieval (PROJECT_SPEC §6 M2).

Two modes:

- **plain**: split on anything that is not a letter or digit, so ``response_model`` becomes
  ``response``, ``model`` and ``BackgroundTasks`` becomes ``backgroundtasks``.
- **code-aware**: identifiers such as ``@app.get``, ``response_model`` and ``BackgroundTasks``
  are kept whole (lower-cased) *and* split into their parts, so a query can match either the
  exact symbol or the words it is made of: ``BackgroundTasks`` → ``backgroundtasks``,
  ``background``, ``tasks``.

Stop words and stemming only ever apply to plain words and identifier parts, never to a whole
identifier: ``response_model`` stays ``response_model`` even with stemming on.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from py_rust_stemmers import SnowballStemmer

# Classic English function words. Deliberately excludes words that carry meaning in API docs
# (get, put, post, return, yield, async, await, none, true, false).
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its
    itself just me more most my myself no nor not of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    """.split()  # noqa: SIM905
)

# An identifier, optionally prefixed by @ and dotted (``@app.get``, ``fastapi.Depends``), or a
# number such as ``404`` or ``3.12``. A trailing sentence period is not part of the match
# because a dot must be followed by a letter or underscore.
_CODE_TOKEN = re.compile(r"@?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\d+(?:\.\d+)*")
_PLAIN_TOKEN = re.compile(r"[A-Za-z0-9]+")
# Boundaries inside an identifier: separators, lower→Upper, and the end of an acronym
# (``HTTPException`` → ``HTTP``, ``Exception``).
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


@dataclass(frozen=True)
class TokenizerConfig:
    lowercase: bool = True
    stopwords: bool = True
    stem: bool = True
    code_aware: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def _stemmer() -> SnowballStemmer:
    return SnowballStemmer("english")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return str(_stemmer().stem_word(word))


def split_identifier(token: str) -> list[str]:
    """Parts of an identifier: ``@app.get`` → ``app, get``; ``HTTPException`` → ``HTTP,
    Exception``; ``response_model`` → ``response, model``."""
    parts: list[str] = []
    for piece in re.split(r"[@._]+", token):
        if piece:
            parts.extend(_CAMEL.findall(piece))
    return parts


def _is_compound(token: str) -> bool:
    """True when the token is more than one plain word (has separators or inner capitals)."""
    return len(split_identifier(token)) > 1 or any(ch in token for ch in "@._")


class Tokenizer:
    """Turns text into index terms according to a :class:`TokenizerConfig`."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()

    def _word(self, word: str) -> str | None:
        """Normalise one plain word or identifier part; ``None`` if it is dropped."""
        cfg = self.config
        if cfg.lowercase:
            word = word.lower()
        if cfg.stopwords and word.lower() in STOPWORDS:
            return None
        if cfg.stem and word.isalpha():
            word = _stem(word.lower()) if cfg.lowercase else _stem(word)
        return word

    def tokenize(self, text: str) -> list[str]:
        """Index terms of ``text`` in order; identifiers emit the whole token then its parts."""
        cfg = self.config
        out: list[str] = []
        if not cfg.code_aware:
            for raw in _PLAIN_TOKEN.findall(text):
                term = self._word(raw)
                if term:
                    out.append(term)
            return out
        for raw in _CODE_TOKEN.findall(text):
            if raw[0].isdigit() or not _is_compound(raw):
                term = self._word(raw)
                if term:
                    out.append(term)
                continue
            whole = raw.lower() if cfg.lowercase else raw
            out.append(whole)
            for part in split_identifier(raw):
                term = self._word(part)
                if term and term != whole:
                    out.append(term)
        return out
