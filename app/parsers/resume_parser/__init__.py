# app/parsers/resume_parser/__init__.py
"""Resume text -> Layer 1 frame persisted on Resume; Layer2 derived when binding Candidate."""

from __future__ import annotations

from typing import Any, Dict
from app.parsers.resume_parser.candidate_profile_builder import (
    get_candidate_bind_for_resume,
)
from app.parsers.resume_parser.resume_rule_layer1.assemble import build_layer1_document


def merge_parsed_layer1_document(doc: Dict[str, Any], full_text: str = "") -> Dict[str, Any]:
    """
    Shape stored on ``Resume.parsed``: metadata + extracted content only.

    ``full_text`` is kept for call-site compatibility (e.g. pipeline) but not persisted
    here; use ``document_meta.raw_text_preview`` for short context. Layer2 is built
    on demand via ``get_candidate_bind_for_resume`` when syncing the Candidate.
    """
    _ = full_text  # optional at call sites; not written into resume.parsed
    return {
        "parser_version": doc["parser_version"],
        "document_meta": doc["document_meta"],
        "layer_1_extracted": doc["layer_1_extracted"],
    }


def parse_resume_text(
    text: str,
    *,
    document_id: str = "",
    candidate_id: str = "",
    file_name: str = "resume.txt",
    text_extraction_method: str = "text",
    ocr_used: bool = False,
    total_pages: int | None = None,
) -> Dict[str, Any]:
    """
    Build persisted Layer 1 payload: ``parser_version``, ``document_meta``,
    ``layer_1_extracted`` (no ``layer_2`` on resume).
    """
    raw = (text or "").strip()
    if not raw:
        doc = build_layer1_document(
            "",
            document_id=document_id,
            candidate_id=candidate_id,
            file_name=file_name,
            text_extraction_method=text_extraction_method,
            ocr_used=ocr_used,
            total_pages=total_pages,
        )
        return merge_parsed_layer1_document(doc, "")

    doc = build_layer1_document(
        text,
        document_id=document_id,
        candidate_id=candidate_id,
        file_name=file_name,
        text_extraction_method=text_extraction_method,
        ocr_used=ocr_used,
        total_pages=total_pages,
    )
    return merge_parsed_layer1_document(doc, text)


__all__ = [
    "build_layer1_document",
    "get_candidate_bind_for_resume",
    "merge_parsed_layer1_document",
    "parse_resume_text",
]
