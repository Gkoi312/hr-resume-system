# app/schemas/resume_frame.py
"""Pydantic shapes for resume frame: layer_1_extracted aligns with layer1_simple_v1 blocks."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str = ""
    candidate_id: str = ""
    file_name: str = ""
    file_type: str = "pdf"
    language: str = "zh"
    parse_time: str = ""
    text_extraction_method: str = "pdf_parser"
    ocr_used: bool = False
    total_pages: Optional[int] = None
    parse_status: str = "success"
    warnings: List[str] = Field(default_factory=list)
    raw_text_preview: str = ""


class L1BasicBlock(BaseModel):
    """Same keys as layer1_simple_v1.basic."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    phone: str = ""
    email: str = ""
    location: str = ""
    birth_text: str = ""
    target_role: str = ""
    links: List[str] = Field(default_factory=list)
    raw_block: str = ""


class L1EducationBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    school: str = ""
    degree: str = ""
    major: str = ""
    start: str = ""
    end: str = ""
    school_tier: str = "other"
    raw_block: str = ""


class L1WorkBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company: str = ""
    job_role: str = ""
    job_type: str = "unknown"
    start: str = ""
    end: str = ""
    descriptions: List[str] = Field(default_factory=list)
    raw_block: str = ""


class L1ProjectBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project_name: str = ""
    role: str = ""
    start: str = ""
    end: str = ""
    descriptions: List[str] = Field(default_factory=list)
    raw_block: str = ""


class L1AdditionalBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    languages: List[str] = Field(default_factory=list)
    certificates: List[str] = Field(default_factory=list)
    awards: List[str] = Field(default_factory=list)
    self_evaluation: str = ""
    raw_block: str = ""


class Layer1Extracted(BaseModel):
    """Blocks align with layer1_simple_v1; skills are top-level (not inside additional)."""

    model_config = ConfigDict(extra="ignore")

    basic: L1BasicBlock = Field(default_factory=L1BasicBlock)
    education: List[L1EducationBlock] = Field(default_factory=list)
    work_experience: List[L1WorkBlock] = Field(default_factory=list)
    projects: List[L1ProjectBlock] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    additional: L1AdditionalBlock = Field(default_factory=L1AdditionalBlock)


class ResumeLayer1Document(BaseModel):
    """parser_version + document_meta + layer_1_extracted (no layer_2)."""

    model_config = ConfigDict(extra="ignore")

    parser_version: str = "resume_v1"
    document_meta: DocumentMeta = Field(default_factory=DocumentMeta)
    layer_1_extracted: Layer1Extracted = Field(default_factory=Layer1Extracted)


def empty_layer1_document(parser_version: str = "resume_v1") -> dict[str, Any]:
    """JSON-serializable empty Layer 1 document for DB/API."""
    doc = ResumeLayer1Document(parser_version=parser_version)
    return doc.model_dump(mode="json")


def validate_layer1_document(data: dict[str, Any]) -> ResumeLayer1Document:
    return ResumeLayer1Document.model_validate(data)
