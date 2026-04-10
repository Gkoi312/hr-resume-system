# app/schemas/resume.py
"""Pydantic schemas for Resume."""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app import statuses


class ResumeBase(BaseModel):
    """Base resume schema."""

    file_path: Optional[str] = None
    file_name: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    status: str = statuses.RESUME_STATUS_UPLOADED


class ResumeCreate(ResumeBase):
    """Schema for creating a resume."""

    candidate_id: UUID


class ResumeUpdate(BaseModel):
    """Schema for partial resume update."""

    file_path: Optional[str] = None
    file_name: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    status: Optional[str] = None


class ResumeInDB(ResumeBase):
    """Schema for resume as stored."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    candidate_id: UUID
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


class ResumeResponse(ResumeInDB):
    """Schema for resume in API response."""

    pass
