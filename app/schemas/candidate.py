# app/schemas/candidate.py
"""Pydantic schemas for Candidate."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class CandidateBase(BaseModel):
    """Base candidate schema."""

    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    education: Optional[List[Dict[str, Any]]] = None
    work_experience: Optional[List[Dict[str, Any]]] = None
    skills: Optional[List[str]] = None
    projects: Optional[List[Dict[str, Any]]] = None
    years_of_experience: Optional[int] = None
    summary: Optional[str] = None
    direction_tags: Optional[List[str]] = None


class CandidateCreate(CandidateBase):
    """Schema for creating a candidate."""

    pass


class CandidateUpdate(BaseModel):
    """Schema for partial candidate update."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    education: Optional[List[Dict[str, Any]]] = None
    work_experience: Optional[List[Dict[str, Any]]] = None
    skills: Optional[List[str]] = None
    projects: Optional[List[Dict[str, Any]]] = None
    years_of_experience: Optional[int] = None
    summary: Optional[str] = None
    direction_tags: Optional[List[str]] = None


class CandidateInDB(CandidateBase):
    """Schema for candidate as stored."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime


class CandidateResponse(CandidateInDB):
    """Schema for candidate in API response."""

    pass
