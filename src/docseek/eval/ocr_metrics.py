"""OCR quality metrics: CER, WER and table cell accuracy.

Both texts are normalised first: Unicode NFKC (the PDF text layer keeps typographic
ligatures such as "ﬂ", which OCR correctly reads as "fl"), then whitespace runs collapse to
one space (line breaks and column layout are not part of the score; reading order still is,
because the edit distance sees word order).

- ``CER = Levenshtein(chars(ref), chars(hyp)) / len(ref)``
- ``WER = Levenshtein(words(ref), words(hyp)) / len(words(ref))``

Both can exceed 1 when the hypothesis is much longer than the reference.

Table cell accuracy (best effort, layout-agnostic): for every reference row, the hypothesis
line that contains the most of that row's cells is taken as its match; a cell counts as
correct when its normalised text occurs in that line (markdown pipes are ignored).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rapidfuzz.distance import Levenshtein


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("CER is undefined for an empty reference")
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        raise ValueError("WER is undefined for an empty reference")
    return Levenshtein.distance(ref, hyp) / len(ref)


def _cell(text: str) -> str:
    return normalize(re.sub(r"[`*|]", " ", text)).lower()


@dataclass(frozen=True)
class CellScore:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else float("nan")


def table_cell_accuracy(tables: Sequence[dict[str, Any]], hypothesis: str) -> CellScore:
    lines = [_cell(line) for line in hypothesis.splitlines() if line.strip()]
    correct = total = 0
    for table in tables:
        for row in [table["header"], *table["rows"]]:
            cells = [_cell(c) for c in row if _cell(c)]
            total += len(cells)
            best = 0
            for line in lines:
                best = max(best, sum(1 for c in cells if c in line))
            correct += best
    return CellScore(correct, total)
