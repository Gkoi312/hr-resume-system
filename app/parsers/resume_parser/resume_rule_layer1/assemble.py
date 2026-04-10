# app/parsers/resume_parser/resume_rule_layer1/assemble.py
"""Rule-based build of full Layer 1 document (segment + extract + align)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.parsers.resume_parser.resume_rule_layer1.aligned_from_rules import (
    additional_from_rule_extract,
    basic_from_rule_extract,
    education_from_rule_extract,
    projects_from_rule_extract,
    skills_from_rule_extract,
    work_from_rule_extract,
)
from app.parsers.resume_parser.skill_evidence import normalize_skill_list
from app.parsers.resume_parser.resume_rule_layer1.extract_additional import extract_additional_info
from app.parsers.resume_parser.resume_rule_layer1.extract_basic import extract_basic_info
from app.parsers.resume_parser.resume_rule_layer1.extract_education import extract_education
from app.parsers.resume_parser.resume_rule_layer1.extract_projects import extract_projects
from app.parsers.resume_parser.resume_rule_layer1.extract_work import extract_work_experience
from app.parsers.resume_parser.resume_rule_layer1.segmenter import SegmentKind, segment_resume
from app.schemas.resume_frame import ResumeLayer1Document
from app.parsers.resume_parser.layer1_normalizer import normalize_layer1_blocks


def _detect_language(text: str) -> str:
    if not text:
        return "zh"
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if cn > max(8, len(text) * 0.04) else "en"


def _file_type_from_name(file_name: str) -> str:
    suf = Path(file_name or "").suffix.lower().lstrip(".")
    return suf if suf in ("pdf", "docx", "doc", "txt", "md") else "txt"


def build_layer1_document(
    raw_text: str,
    *,
    document_id: str = "",
    candidate_id: str = "",
    file_name: str = "resume.txt",
    text_extraction_method: str = "text",
    ocr_used: bool = False,
    total_pages: Optional[int] = None,
) -> Dict[str, Any]:
    text = raw_text or ""
    base = ResumeLayer1Document().model_dump(mode="json")
    sec = segment_resume(text)

    warnings = list(sec.warnings)
    bspan = sec.segments[SegmentKind.BASIC]
    layer1: Dict[str, Any] = {
        "basic": basic_from_rule_extract(extract_basic_info(bspan.raw_block, bspan.start)),
    }
    layer1["basic"]["raw_block"] = bspan.raw_block

    espan = sec.segments[SegmentKind.EDUCATION]
    layer1["education"] = education_from_rule_extract(extract_education(espan.raw_block, espan.start))

    wspan = sec.segments[SegmentKind.WORK]
    layer1["work_experience"] = work_from_rule_extract(
        extract_work_experience(wspan.raw_block, wspan.start)
    )

    pspan = sec.segments[SegmentKind.PROJECTS]
    layer1["projects"] = projects_from_rule_extract(
        extract_projects(pspan.raw_block, pspan.start)
    )

    aspan = sec.segments[SegmentKind.ADDITIONAL]
    add_extra = extract_additional_info(aspan.raw_block, sec.subsection_markers)
    layer1["skills"] = normalize_skill_list(skills_from_rule_extract(add_extra)[:50])
    layer1["additional"] = additional_from_rule_extract(
        add_extra, aspan.raw_block, list(sec.subsection_markers)
    )

    if not sec.subsection_markers and aspan.raw_block.strip():
        warnings.append("additional_no_subsection_markers")

    preview = text[:800] if text else ""
    parse_status = "partial" if warnings else "success"

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    base["document_meta"].update(
        {
            "document_id": document_id,
            "candidate_id": candidate_id,
            "file_name": file_name or "",
            "file_type": _file_type_from_name(file_name),
            "language": _detect_language(text),
            "parse_time": now,
            "text_extraction_method": text_extraction_method,
            "ocr_used": ocr_used,
            "total_pages": total_pages,
            "parse_status": parse_status,
            "warnings": warnings,
            "raw_text_preview": preview,
        }
    )
    base["layer_1_extracted"] = layer1
    base["layer_1_extracted"] = normalize_layer1_blocks(layer1)
    return ResumeLayer1Document.model_validate(base).model_dump(mode="json")
