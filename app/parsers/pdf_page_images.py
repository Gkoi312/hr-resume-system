# app/parsers/pdf_page_images.py
"""Rasterize PDF or image files to PNG (base64) for vision LLM."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# PyMuPDF ``filetype`` for common raster uploads
_FITZ_IMAGE_TYPES = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".webp": "webp",
    ".bmp": "bmp",
    ".tif": "tiff",
    ".tiff": "tiff",
}


def pdf_bytes_to_png_base64_pages(
    pdf_bytes: bytes,
    *,
    max_pages: int = 2,
    dpi: float = 150.0,
) -> List[str]:
    """
    Render each PDF page to PNG, return base64 strings (no data-URL prefix).

    Empty list if PyMuPDF unavailable or render fails.
    """
    if not pdf_bytes or max_pages < 1:
        return []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF (pymupdf) not installed; vision PDF path disabled")
        return []

    out: List[str] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            n = min(doc.page_count, max_pages)
            scale = dpi / 72.0
            mat = fitz.Matrix(scale, scale)
            for i in range(n):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                png = pix.tobytes("png")
                out.append(base64.b64encode(png).decode("ascii"))
        finally:
            doc.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("PDF rasterize failed: %s", e)
        return []

    return out


def image_bytes_to_png_base64_pages(file_bytes: bytes, filename: str) -> List[str]:
    """
    Load a single-page image file and return one PNG base64 string per page.

    Falls back to raw base64 of the file when it is already PNG and PyMuPDF fails.
    """
    if not file_bytes:
        return []
    suf = (Path(filename).suffix or "").lower()
    if suf == ".png":
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=file_bytes, filetype="png")
            try:
                page = doc.load_page(0)
                pix = page.get_pixmap(alpha=False)
                png = pix.tobytes("png")
                return [base64.b64encode(png).decode("ascii")]
            finally:
                doc.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("PNG via PyMuPDF failed, using raw bytes: %s", e)
            return [base64.b64encode(file_bytes).decode("ascii")]

    ftype = _FITZ_IMAGE_TYPES.get(suf)
    if not ftype:
        logger.warning("Unsupported image type for vision: %s", filename)
        return []
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=file_bytes, filetype=ftype)
        try:
            page = doc.load_page(0)
            pix = page.get_pixmap(alpha=False)
            png = pix.tobytes("png")
            return [base64.b64encode(png).decode("ascii")]
        finally:
            doc.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("Image rasterize failed: %s", e)
        return []


def vision_png_base64_pages_from_file(
    file_bytes: bytes,
    filename: str,
    *,
    max_pages: int = 2,
    dpi: float = 150.0,
) -> List[str]:
    """PDF -> multi-page PNG; image -> single-page PNG (for ``generate_json_with_images``)."""
    suf = (Path(filename).suffix or "").lower()
    if suf == ".pdf":
        return pdf_bytes_to_png_base64_pages(file_bytes, max_pages=max_pages, dpi=dpi)
    return image_bytes_to_png_base64_pages(file_bytes, filename)
