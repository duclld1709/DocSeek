"""Token counting for chunk size limits.

The default counter uses the WordPiece tokenizer of the default embedding model, because
the 512-token limit in PROJECT_SPEC §6 (M1) is that model's input limit. ``RegexTokenCounter``
is a dependency-free approximation for tests and offline runs.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_REGEX_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@runtime_checkable
class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int:
        """Number of tokens in ``text`` (no special tokens)."""
        ...


class RegexTokenCounter:
    """Words and punctuation marks; close to, but below, WordPiece counts."""

    name = "regex"

    def count(self, text: str) -> int:
        return len(_REGEX_TOKEN_RE.findall(text))


class HFTokenCounter:
    """Counts tokens with a Hugging Face ``tokenizers`` tokenizer (downloaded once, cached).

    Args:
        model_name: Hub id whose ``tokenizer.json`` to load, e.g. the default embedder.
    """

    def __init__(self, model_name: str) -> None:
        from tokenizers import Tokenizer  # imported lazily: only real ingest runs need it

        self.name = f"hf:{model_name}"
        self._tokenizer = Tokenizer.from_pretrained(model_name)
        self._tokenizer.no_truncation()

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
