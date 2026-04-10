# app/schemas/evaluation.py
"""Pydantic schemas for RAG Evaluation."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class EvaluationBase(BaseModel):
    """Base evaluation schema."""

    query: str
    rewritten_query: Optional[str] = None
    context: str
    response: str


class EvaluationCreate(EvaluationBase):
    """Schema for creating an evaluation record."""

    message_id: Optional[UUID] = None
    conversation_id: Optional[UUID] = None
    user_id: Optional[str] = None


class EvaluationUpdateResults(BaseModel):
    """Schema for updating evaluation results."""

    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    evaluation_status: str = "completed"
    error_message: Optional[str] = None
    evaluation_duration_ms: Optional[int] = None


class EvaluationInDB(EvaluationBase):
    """Schema for evaluation as stored."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    message_id: Optional[UUID] = None
    conversation_id: Optional[UUID] = None
    user_id: Optional[str] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    evaluation_status: str
    error_message: Optional[str] = None
    evaluation_duration_ms: Optional[int] = None
    created_at: datetime
    evaluated_at: Optional[datetime] = None


class EvaluationResponse(EvaluationInDB):
    """Schema for evaluation in API response."""

    pass
