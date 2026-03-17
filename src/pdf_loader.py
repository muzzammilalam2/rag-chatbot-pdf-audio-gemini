from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .utils import normalize_text


@dataclass(frozen=True)
class PdfPage:
    pdf_path: str
    page_number: int
    text: str


def extract_pdf_pages(pdf_path: Path) -> list[PdfPage]:
    reader = PdfReader(str(pdf_path))
    pages: list[PdfPage] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = normalize_text(raw)
        if text:
            pages.append(PdfPage(pdf_path=str(pdf_path), page_number=i + 1, text=text))
    return pages

