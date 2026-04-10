# app/parsers/resume_parser/resume_llm_layer1/pipeline.py
"""Resume file -> Layer1: vision LLM (optional), Paddle PP-Structure, or text + rules / text LLM."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.llm.chat_client import LLMClientError
from app.parsers.resume_input import ResumeInputKind, resume_input_kind
from app.parsers.resume_paddle.ppstructure_client import try_predict_file_bytes
from app.parsers.resume_parser.resume_paddle_layer1.map_paddle import (
    map_paddle_page_dicts_to_parsed,
)
from app.parsers.resume_parser.resume_rule_layer1.assemble import _detect_language
from app.parsers.resume_parser.resume_llm_layer1.extract import (
    extract_resume_simple_json,
    extract_resume_simple_json_vision,
    resume_llm_enabled,
    resume_llm_vision_enabled,
)
from app.parsers.resume_parser.resume_llm_layer1.map_simple_to_layer1 import map_simple_to_layer1
from app.parsers.resume_parser import merge_parsed_layer1_document, parse_resume_text
from app.parsers.text_extractor import extract_text_from_file
from app.schemas.resume_frame import ResumeLayer1Document

logger = logging.getLogger(__name__)


def _file_type_from_name(file_name: str) -> str:
    suf = Path(file_name or "").suffix.lower().lstrip(".")
    return suf if suf in ("pdf", "docx", "doc", "txt", "md") else "txt"


def build_document_from_layer1_mapped(
    layer1: Dict[str, Any],
    *,
    document_id: str,
    candidate_id: str,
    file_name: str,
    text_extraction_method: str,
    ocr_used: bool,
    total_pages: Optional[int],
    source_text: str,
    extra_warnings: List[str],
) -> Dict[str, Any]:
    """Wrap mapped layer_1_extracted with parser_version + document_meta."""
    warnings = list(extra_warnings or [])
    parse_status = "partial" if warnings else "success"
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    base = {
        "parser_version": "resume_v1",
        "document_meta": {
            "document_id": document_id,
            "candidate_id": candidate_id,
            "file_name": file_name or "",
            "file_type": _file_type_from_name(file_name),
            "language": _detect_language(source_text),
            "parse_time": now,
            "text_extraction_method": text_extraction_method,
            "ocr_used": ocr_used,
            "total_pages": total_pages,
            "parse_status": parse_status,
            "warnings": warnings,
            "raw_text_preview": (source_text or "")[:800],
        },
        "layer_1_extracted": layer1,
    }
    return ResumeLayer1Document.model_validate(base).model_dump(mode="json")


def _preview_from_simple(simple: Dict[str, Any]) -> str:
    basic = simple.get("basic")
    if isinstance(basic, dict):
        parts = [str(basic.get(k) or "") for k in ("name", "email", "phone", "target_role")]
        t = " ".join(p for p in parts if p).strip()
        if t:
            return t[:800]
    try:
        return json.dumps(simple, ensure_ascii=False)[:800]
    except (TypeError, ValueError):
        return ""


def _collect_rec_texts_from_obj(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "rec_texts" and isinstance(v, list):
                for x in v:
                    out.append(str(x))
            else:
                out.extend(_collect_rec_texts_from_obj(v))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_collect_rec_texts_from_obj(it))
    return out


def _build_paddle_rec_texts_source(page_dicts: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for pd in page_dicts:
        lines.extend(_collect_rec_texts_from_obj(pd))
    return "\n".join(lines)


async def parse_resume_document(
    file_bytes: bytes,
    file_name: str,
    *,
    document_id: str = "",
    candidate_id: str = "",
) -> Dict[str, Any]:
    """
    Single entry for uploaded resume bytes.

    - TXT / DOCX: ``text_extractor`` then text LLM (if enabled) or rule Layer1.
    - PDF / image:
      1) vision LLM (if ``RESUME_LLM_*`` + ``RESUME_LLM_VISION``), else
      2) Paddle PP-Structure -> ``rec_texts`` -> text LLM (if enabled), else
      3) Paddle flatten -> rule Layer1 (``map_paddle_page_dicts_to_parsed``).
    """
    fn = file_name or "resume.bin"
    kind = resume_input_kind(fn)
    llm_on = resume_llm_enabled()
    meta_kw = dict(
        document_id=document_id,
        candidate_id=candidate_id,
        file_name=fn,
        total_pages=None,
    )

    if kind in (ResumeInputKind.TEXT_PLAIN, ResumeInputKind.OFFICE):
        raw = extract_text_from_file(file_bytes, fn).strip()
        tex_method = "docx_parser" if kind == ResumeInputKind.OFFICE else "text"
        if not llm_on:
            return (
                parse_resume_text(raw, text_extraction_method=tex_method, ocr_used=False, **meta_kw)
                if raw
                else parse_resume_text("", text_extraction_method=tex_method, ocr_used=False, **meta_kw)
            )
        try:
            simple = await extract_resume_simple_json(raw)
            layer1 = map_simple_to_layer1(simple)
            warn = [str(w) for w in (simple.get("warnings") or []) if str(w).strip()]
            warn.append("parsed_via_resume_llm")
            doc = build_document_from_layer1_mapped(
                layer1,
                extra_warnings=warn,
                source_text=raw,
                text_extraction_method=tex_method,
                ocr_used=False,
                file_name=fn,
                document_id=document_id,
                candidate_id=candidate_id,
                total_pages=None,
            )
            return merge_parsed_layer1_document(doc, full_text=raw)
        except LLMClientError as e:
            logger.warning("Resume text LLM failed, using rules: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.exception("Resume text LLM unexpected error, using rules: %s", e)
        return (
            parse_resume_text(raw, text_extraction_method=tex_method, ocr_used=False, **meta_kw)
            if raw
            else parse_resume_text("", text_extraction_method=tex_method, ocr_used=False, **meta_kw)
        )

    # PDF or image
    if resume_llm_vision_enabled():
        try:
            simple = await extract_resume_simple_json_vision(
                "",
                file_bytes=file_bytes,
                file_name=fn,
            )
            layer1 = map_simple_to_layer1(simple)
            warn = [str(w) for w in (simple.get("warnings") or []) if str(w).strip()]
            warn.append("parsed_via_resume_llm_vision")
            preview = _preview_from_simple(simple)
            doc = build_document_from_layer1_mapped(
                layer1,
                extra_warnings=warn,
                source_text=preview,
                text_extraction_method="resume_llm_vision",
                ocr_used=True,
                file_name=fn,
                document_id=document_id,
                candidate_id=candidate_id,
                total_pages=None,
            )
            return merge_parsed_layer1_document(doc, full_text=preview)
        except LLMClientError as e:
            logger.warning("Resume vision LLM failed, falling back to Paddle: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.exception("Resume vision LLM unexpected error, falling back to Paddle: %s", e)

    pages, perr = try_predict_file_bytes(file_bytes, fn)
    if not pages:
        extra = ["paddle_no_output"]
        if perr:
            extra.append(f"paddle_error:{perr[:200]}")
        out = parse_resume_text(
            "",
            text_extraction_method="paddle_ppstructure_v3",
            ocr_used=True,
            **meta_kw,
        )
        dm = out.get("document_meta") or {}
        w = list(dm.get("warnings") or [])
        for x in extra:
            if x and x not in w:
                w.append(x)
        dm["warnings"] = w
        dm["parse_status"] = "partial"
        out["document_meta"] = dm
        return out

    if llm_on:
        rec_text_source = _build_paddle_rec_texts_source(pages)
        if rec_text_source:
            try:
                simple = await extract_resume_simple_json(rec_text_source)
                layer1 = map_simple_to_layer1(simple)
                warn = [str(w) for w in (simple.get("warnings") or []) if str(w).strip()]
                warn.append("parsed_via_resume_llm_from_paddle_rec_texts")
                doc = build_document_from_layer1_mapped(
                    layer1,
                    extra_warnings=warn,
                    source_text=rec_text_source,
                    text_extraction_method="paddle_rec_texts_llm",
                    ocr_used=True,
                    file_name=fn,
                    document_id=document_id,
                    candidate_id=candidate_id,
                    total_pages=len(pages),
                )
                return merge_parsed_layer1_document(doc, full_text=rec_text_source)
            except LLMClientError as e:
                logger.warning("Resume text LLM on paddle rec_texts failed, using rules: %s", e)
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "Resume text LLM on paddle rec_texts unexpected error, using rules: %s", e
                )

    parsed, _, _ = map_paddle_page_dicts_to_parsed(
        pages,
        document_id=document_id,
        candidate_id=candidate_id,
        file_name=fn,
        total_pages=len(pages),
    )
    return parsed


