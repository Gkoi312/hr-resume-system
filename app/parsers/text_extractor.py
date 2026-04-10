# app/parsers/text_extractor.py
"""Extract plain text from DOCX or TXT. PDFs are handled by Paddle / vision (not pypdf)."""

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_file(
    file_bytes: bytes,
    filename: str,
) -> str:
    """
    Extract plain text from DOCX or TXT.

    For ``.pdf`` (and images), returns empty string — use ``resume_parse_pipeline``
    with Paddle or vision LLM instead.
    """
    suffix = (Path(filename).suffix or "").lower()
    if suffix == ".txt" or suffix == ".md":
        return _extract_txt(file_bytes)
    if suffix == ".pdf":
        logger.debug("extract_text_from_file: PDF skipped (use Paddle / vision pipeline)")
        return ""
    if suffix in (".docx", ".doc"):
        return _extract_docx(file_bytes)
    logger.warning("Unsupported file type for text extraction: %s", filename)
    return ""


def _extract_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="replace").strip()
    except Exception as e:
        logger.warning("TXT decode error: %s", e)
        return data.decode("latin-1", errors="replace").strip()


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError:
        logger.warning("python-docx not installed; pip install python-docx")
        return ""
    try:
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(parts).strip()
    except Exception as e:
        logger.warning("DOCX extract error: %s", e)
        return ""
