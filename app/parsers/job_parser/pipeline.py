"""Job upload parsing pipeline.

pdf/image -> PaddleOCR + LLM
txt/doc/docx -> Python text extractor + LLM
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.llm.chat_client import LLMClientError
from app.parsers.job_parser.extract import extract_job_structured_from_text
from app.parsers.resume_paddle.ppstructure_client import try_predict_file_bytes
from app.parsers.text_extractor import extract_text_from_file
from app.schemas.job import JobStructured


class JobParseError(RuntimeError):
    """Base error for upload parsing failures."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class UnsupportedFileTypeError(JobParseError):
    def __init__(self, file_name: str):
        suffix = (Path(file_name or "").suffix or "").lower() or "(none)"
        super().__init__(
            "UNSUPPORTED_FILE_TYPE",
            f"Unsupported file type: {suffix}. Supported: .pdf/.png/.jpg/.jpeg/.webp/.bmp/.txt/.md/.doc/.docx",
        )


class OCRParseError(JobParseError):
    def __init__(self, message: str):
        super().__init__("OCR_FAILED", message)


class LLMParseError(JobParseError):
    def __init__(self, message: str):
        super().__init__("LLM_FAILED", message)


def _collect_rec_texts_from_obj(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "rec_texts" and isinstance(v, list):
                out.extend(str(x) for x in v)
            else:
                out.extend(_collect_rec_texts_from_obj(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_collect_rec_texts_from_obj(item))
    return out


def _text_from_paddle_pages(pages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for page in pages:
        lines.extend(_collect_rec_texts_from_obj(page))
    return "\n".join(x.strip() for x in lines if str(x).strip())


def _is_pdf_or_image(file_name: str) -> bool:
    suffix = (Path(file_name or "").suffix or "").lower()
    return suffix in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _is_text_extractable(file_name: str) -> bool:
    suffix = (Path(file_name or "").suffix or "").lower()
    return suffix in {".txt", ".md", ".doc", ".docx"}


async def parse_job_document(file_bytes: bytes, file_name: str) -> Dict[str, Any]:
    if not file_bytes:
        raise ValueError("empty job file")
    if _is_pdf_or_image(file_name):
        pages, err = try_predict_file_bytes(file_bytes, file_name)
        if not pages:
            raise OCRParseError(f"PaddleOCR failed: {err or 'no output'}")
        raw_text = _text_from_paddle_pages(pages).strip()
        if not raw_text:
            raise OCRParseError("PaddleOCR produced empty text")
        try:
            structured: JobStructured = await extract_job_structured_from_text(raw_text)
        except LLMClientError as exc:
            raise LLMParseError(str(exc)) from exc
        return {
            "raw_jd_text": raw_text,
            "structured": structured.model_dump(exclude_none=True),
            "text_extraction_method": "paddle_ocr",
        }

    if not _is_text_extractable(file_name):
        raise UnsupportedFileTypeError(file_name)

    raw_text = extract_text_from_file(file_bytes, file_name).strip()
    if not raw_text:
        raise RuntimeError("Python text extraction returned empty content")
    try:
        structured = await extract_job_structured_from_text(raw_text)
    except LLMClientError as exc:
        raise LLMParseError(str(exc)) from exc
    return {
        "raw_jd_text": raw_text,
        "structured": structured.model_dump(exclude_none=True),
        "text_extraction_method": "python_text_extractor",
    }

