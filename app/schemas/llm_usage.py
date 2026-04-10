# app/schemas/llm_usage.py
"""Pydantic schemas for LLM Usage Log."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class LLMUsageLogBase(BaseModel):
    """Base LLM usage log schema."""

    request_id: str
    module_name: str
    user_id: Optional[str] = None
    conversation_id: Optional[UUID] = None
    model_name: Optional[str] = None
    tool_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    duration_ms: Optional[int] = None


class LLMUsageLogCreate(LLMUsageLogBase):
    """Schema for creating an LLM usage log."""

    pass


class LLMUsageLogInDB(LLMUsageLogBase):
    """Schema for LLM usage log as stored."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime


class LLMUsageLogResponse(LLMUsageLogInDB):
    """Schema for LLM usage log in API response."""

    pass
