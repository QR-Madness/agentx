"""Extract plain text from an uploaded document.

v1 formats: PDF (via ``pypdf``) + text/markdown/code (utf-8). Dispatch is by file
extension (the content-type header is unreliable across clients).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class UnsupportedDocumentError(Exception):
    """Raised when a file's type can't be parsed to text."""


def extension_of(filename: str) -> str:
    """Lower-case extension without the dot (``"report.PDF"`` → ``"pdf"``)."""
    return Path(filename).suffix.lstrip(".").lower()


def is_supported(filename: str, allowed: list[str]) -> bool:
    """Whether ``filename``'s extension is in the allow-list."""
    return extension_of(filename) in {e.lower() for e in allowed}


def parse_to_text(raw: bytes, filename: str) -> str:
    """Decode ``raw`` document bytes to plain text by file type.

    Raises :class:`UnsupportedDocumentError` for types we can't handle and on a
    hard parse failure, so the caller can mark the document ``failed`` with a reason.
    """
    ext = extension_of(filename)
    if ext == "pdf":
        return _clean(_parse_pdf(raw))
    # Everything else in the allow-list is treated as utf-8 text/code.
    try:
        return _clean(raw.decode("utf-8", errors="replace"))
    except Exception as e:  # pragma: no cover - decode with errors="replace" rarely raises
        raise UnsupportedDocumentError(f"could not decode {filename} as text: {e}") from e


def _clean(text: str) -> str:
    """Strip NUL bytes — Postgres ``TEXT`` rejects ``0x00``, and PDF extraction
    routinely emits them. Other control chars are valid in PG text, so leave them."""
    return text.replace("\x00", "")


def _parse_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise UnsupportedDocumentError(f"pypdf not available: {e}") from e
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as e:
        raise UnsupportedDocumentError(f"failed to read PDF: {e}") from e
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text.strip():
        raise UnsupportedDocumentError(
            "no extractable text (the PDF may be scanned images — OCR is out of v1 scope)"
        )
    return text
