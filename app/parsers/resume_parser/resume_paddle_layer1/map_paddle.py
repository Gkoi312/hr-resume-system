# app/parsers/resume_parser/resume_paddle_layer1/map_paddle.py
"""Paddle PP-Structure page JSON -> merged parsed resume (Layer1 + legacy flat)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.parsers.resume_paddle.flatten import flatten_ppstructure_pages
from app.parsers.resume_parser import parse_resume_text


def map_paddle_page_dicts_to_parsed(
    page_dicts: List[Dict[str, Any]],
    *,
    document_id: str = "",
    candidate_id: str = "",
    file_name: str = "resume.pdf",
    total_pages: Optional[int] = None,
) -> Tuple[Dict[str, Any], str, List[str]]:
    """
    Flatten Paddle JSON to text, then rule-based Layer1 (second mapping path vs LLM simple JSON).

    Returns (merged parsed dict, plain_text_used, warnings).
    """
    text, flat_warn = flatten_ppstructure_pages(page_dicts)
    warnings = list(flat_warn)
    n_pages = total_pages if total_pages is not None else len(page_dicts)
    doc = parse_resume_text(
        text,
        document_id=document_id,
        candidate_id=candidate_id,
        file_name=file_name,
        text_extraction_method="paddle_ppstructure_v3",
        ocr_used=True,
        total_pages=n_pages if n_pages > 0 else None,
    )
    dm = doc.get("document_meta") or {}
    merged_warn = list(dm.get("warnings") or [])
    for w in warnings:
        if w and w not in merged_warn:
            merged_warn.append(w)
    dm["warnings"] = merged_warn
    dm["parse_status"] = "partial" if merged_warn else "success"
    doc["document_meta"] = dm
    return doc, text, warnings
