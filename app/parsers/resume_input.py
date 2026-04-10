# app/parsers/resume_input.py
"""Classify uploaded resume files for routing (text / office / PDF / image)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class ResumeInputKind(str, Enum):
    """How the parser should read the file before Layer1 mapping."""

    TEXT_PLAIN = "text_plain"  # .txt, .md
    OFFICE = "office"  # .doc, .docx
    PDF = "pdf"
    IMAGE = "image"  # raster resume scan / photo


_TEXT_EXT = {".txt", ".md"}
_OFFICE_EXT = {".doc", ".docx"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def resume_input_kind(filename: str) -> ResumeInputKind:
    suf = (Path(filename or "").suffix or "").lower()
    if suf in _TEXT_EXT:
        return ResumeInputKind.TEXT_PLAIN
    if suf in _OFFICE_EXT:
        return ResumeInputKind.OFFICE
    if suf == ".pdf":
        return ResumeInputKind.PDF
    if suf in _IMAGE_EXT:
        return ResumeInputKind.IMAGE
    return ResumeInputKind.PDF


def is_visual_document(filename: str) -> bool:
    """True when file should use Paddle or vision LLM (not plain text extraction)."""
    k = resume_input_kind(filename)
    return k in (ResumeInputKind.PDF, ResumeInputKind.IMAGE)


__all__ = ["ResumeInputKind", "is_visual_document", "resume_input_kind"]
