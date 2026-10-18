"""PDF pages to text: use the text layer when a page has one, otherwise OCR the page image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docseek.ingest.ocr import OcrEngine

MIN_TEXT_CHARS = 20  # fewer extractable characters than this = treat the page as a scan


@dataclass(frozen=True)
class ParsedPage:
    page: int  # 1-based
    text: str
    method: str  # "text-layer" or the OCR engine name


def page_has_text_layer(text: str) -> bool:
    return len(text.strip()) >= MIN_TEXT_CHARS


def parse_pdf(path: Path, ocr: OcrEngine | None, *, dpi: int = 150) -> list[ParsedPage]:
    import pymupdf

    out: list[ParsedPage] = []
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if page_has_text_layer(text):
                out.append(ParsedPage(i, text, "text-layer"))
                continue
            if ocr is None:
                raise ValueError(f"{path} page {i} has no text layer and no OCR engine was given")
            png = page.get_pixmap(dpi=dpi).tobytes("png")
            out.append(ParsedPage(i, ocr.ocr(png), ocr.name))
    return out


def page_png(path: Path, page: int, *, dpi: int = 150) -> bytes:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return bytes(doc[page - 1].get_pixmap(dpi=dpi).tobytes("png"))
