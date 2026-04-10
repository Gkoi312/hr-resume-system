# app/schemas/job.py
"""Pydantic schemas for Job."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app import statuses


class JobStructured(BaseModel):
    """Structured JD fields (optional nested schema)."""

    job_title: Optional[str] = None
    required_skills: Optional[list[str]] = None
    preferred_skills: Optional[list[str]] = None
    responsibilities: Optional[list[str]] = None
    min_years: Optional[int] = None
    education_requirement: Optional[str] = None
    industry_preference: Optional[list[str]] = None
    keywords: Optional[list[str]] = None
    job_summary: Optional[str] = None


class JobBase(BaseModel):
    """Base job schema."""

    title: str
    raw_jd_text: Optional[str] = None
    structured: Optional[JobStructured] = None
    status: str = statuses.JOB_STATUS_DRAFT


class JobCreate(JobBase):
    """Schema for creating a job."""

    created_by_id: Optional[UUID] = None


class JobUpdate(BaseModel):
    """Schema for partial job update."""

    title: Optional[str] = None
    raw_jd_text: Optional[str] = None
    structured: Optional[JobStructured] = None
    status: Optional[str] = None


class JobInDB(JobBase):
    """Schema for job as stored."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_by_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


class JobResponse(JobInDB):
    """Schema for job in API response."""

    pass
