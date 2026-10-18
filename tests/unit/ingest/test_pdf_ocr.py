"""Text-layer detection, OCR fallback and RapidOCR reading order (no real OCR engine)."""

from pathlib import Path

import pymupdf
import pytest

from docseek.ingest.ocr import reading_order
from docseek.ingest.pdf_parser import page_has_text_layer, parse_pdf


class FakeOcr:
    name = "fake-ocr"

    def __init__(self) -> None:
        self.calls = 0

    def ocr(self, png: bytes) -> str:
        assert png.startswith(b"\x89PNG")
        self.calls += 1
        return "ocr text"


def _pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "This page has a real text layer with words.")
    scan = doc.new_page()  # image-only page: just a drawn rectangle, no text
    scan.draw_rect(pymupdf.Rect(50, 50, 200, 200), fill=(0.5, 0.5, 0.5))
    path = tmp_path / "mixed.pdf"
    doc.save(str(path))
    return path


def test_text_layer_threshold() -> None:
    assert not page_has_text_layer("  \n 12 \n")
    assert page_has_text_layer("x" * 20)


def test_parse_pdf_uses_text_layer_then_ocr(tmp_path: Path) -> None:
    ocr = FakeOcr()
    pages = parse_pdf(_pdf(tmp_path), ocr)
    assert [p.method for p in pages] == ["text-layer", "fake-ocr"]
    assert "real text layer" in pages[0].text
    assert pages[1].text == "ocr text"
    assert ocr.calls == 1


def test_parse_pdf_without_ocr_refuses_scans(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no text layer"):
        parse_pdf(_pdf(tmp_path), None)


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_reading_order_single_column_joins_lines() -> None:
    boxes = [
        (_box(10, 50, 60, 60), "world"),
        (_box(70, 51, 120, 61), "again"),
        (_box(10, 10, 60, 20), "hello"),
    ]
    assert reading_order(boxes, page_width=200) == "hello\nworld again"


def test_reading_order_two_columns_left_first() -> None:
    boxes = [
        (_box(10, 10, 90, 20), "L1"),
        (_box(110, 10, 190, 20), "R1"),
        (_box(10, 30, 90, 40), "L2"),
        (_box(110, 30, 190, 40), "R2"),
    ]
    assert reading_order(boxes, page_width=200).split("\n") == ["L1", "L2", "R1", "R2"]
