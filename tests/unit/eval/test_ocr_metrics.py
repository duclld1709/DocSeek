"""CER/WER and table cell accuracy against hand-computed values."""

import pytest

from docseek.eval.ocr_metrics import cer, table_cell_accuracy, wer


def test_cer_hand_computed() -> None:
    # "kitten" -> "sitting": 3 edits (k->s, e->i, +g) over 6 reference chars.
    assert cer("kitten", "sitting") == pytest.approx(3 / 6)
    assert cer("abc", "abc") == 0.0
    # Whitespace layout does not count: line breaks and runs of spaces collapse.
    assert cer("a  b\nc", "a b c") == 0.0
    # Hypothesis much longer than the reference: CER > 1.
    assert cer("ab", "abcdef") == pytest.approx(4 / 2)
    with pytest.raises(ValueError):
        cer("   ", "x")


def test_wer_hand_computed() -> None:
    # ref 4 words; hyp substitutes "sat" and drops "down": 2 edits / 4.
    assert wer("the cat sat down", "the cat sit") == pytest.approx(2 / 4)
    # Word order matters (two-column reading order errors are penalised).
    assert wer("a b c d", "c d a b") == pytest.approx(4 / 4)
    assert wer("x y", "x y") == 0.0


def test_table_cell_accuracy() -> None:
    tables = [
        {
            "header": ["Service", "SLA"],
            "rows": [["Payments API", "99.95%"], ["Payouts API", "99.9%"]],
        }
    ]
    plain = "Service SLA\nPayments API 99.95%\nPayouts AP1 99.9%\n"
    s = table_cell_accuracy(tables, plain)
    assert (s.correct, s.total) == (5, 6)  # "Payouts API" misread as "Payouts AP1"
    md = "| Service | SLA |\n|---|---|\n| Payments API | 99.95% |\n| Payouts API | 99.9% |"
    assert table_cell_accuracy(tables, md).accuracy == 1.0
    assert table_cell_accuracy(tables, "").correct == 0
