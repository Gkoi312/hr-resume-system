"""Pydantic schemas for background Task."""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TaskBase(BaseModel):
    """Base task schema."""

    task_type: str
    resource_type: Optional[str] = None
    resource_id: Optional[UUID] = None
    status: str
    error_message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    result_summary: Optional[Dict[str, Any]] = None


class TaskCreate(BaseModel):
    """Schema for creating a task."""

    task_type: str
    resource_type: Optional[str] = None
    resource_id: Optional[UUID] = None
    payload: Optional[Dict[str, Any]] = None


class TaskInDB(TaskBase):
    """Schema for task as stored."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime


class TaskResponse(TaskInDB):
    """Schema for task in API response."""

    pass

