# app/database/models.py
"""
SQLAlchemy ORM models for HR Resume Screening System.
Defines: User, Job, Resume, Candidate, CandidateJobMatch;
reusable: RAGEvaluation, LLMUsageLog (no FK to messages in phase 1).
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app import statuses

_VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "db").strip().lower()
_DATABASE_URL = os.getenv("DATABASE_URL", "").strip().lower()
_IS_SQLITE = "sqlite" in _DATABASE_URL
_USE_PGVECTOR = _VECTOR_STORE_BACKEND == "pgvector" and not _IS_SQLITE
_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

if _USE_PGVECTOR:
    try:
        from pgvector.sqlalchemy import Vector as PgVector  # type: ignore[import-not-found]
        from pgvector.vector import Vector as _PgVectorCore  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "VECTOR_STORE_BACKEND=pgvector but python package `pgvector` is not available. "
            "Please `pip install pgvector` and ensure PostgreSQL has the `vector` extension."
        ) from exc

    def _pgvector_from_db(cls, value):  # type: ignore[no-untyped-def]
        """asyncpg 常把 ``vector`` 列解码成 list；pgvector 默认 ``_from_db`` 只按字符串 ``from_text`` 解析。"""
        import numpy as np

        if value is None or isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float32)
        return cls.from_text(value).to_numpy().astype(np.float32)

    _PgVectorCore._from_db = classmethod(_pgvector_from_db)
    _VectorColumnType = PgVector(_EMBEDDING_DIM)
else:
    _VectorColumnType = JSON


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ==================== User (reusable from CookHero) ====================

class UserModel(Base):
    """ORM model for application users."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    occupation: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    profile: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_instruction: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (Index("ix_users_username", "username", unique=True),)

    jobs: Mapped[List["JobModel"]] = relationship(
        "JobModel", back_populates="created_by", foreign_keys="JobModel.created_by_id"
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "username": self.username,
            "occupation": self.occupation,
            "bio": self.bio,
            "profile": self.profile,
            "user_instruction": self.user_instruction,
            "created_at": self.created_at.isoformat(),
        }


# ==================== Job ====================

class JobModel(Base):
    """ORM model for job positions (JD)."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    raw_jd_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    structured: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    min_years: Mapped[Optional[int]] = mapped_column(nullable=True)
    education_requirement: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default=statuses.JOB_STATUS_DRAFT, nullable=False
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    created_by: Mapped[Optional["UserModel"]] = relationship(
        "UserModel", back_populates="jobs", foreign_keys=[created_by_id]
    )
    matches: Mapped[List["CandidateJobMatchModel"]] = relationship(
        "CandidateJobMatchModel", back_populates="job", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_jobs_created_by_updated", "created_by_id", "updated_at"),)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "created_by_id": str(self.created_by_id) if self.created_by_id else None,
            "title": self.title,
            "raw_jd_text": self.raw_jd_text,
            "structured": self.structured,
            "min_years": self.min_years,
            "education_requirement": self.education_requirement,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ==================== Candidate ====================

class CandidateModel(Base):
    """ORM model for candidates (one per person; multiple resumes possible)."""

    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    phone: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    education: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    work_experience: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    skills: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    projects: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    years_of_experience: Mapped[Optional[int]] = mapped_column(nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    direction_tags: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    resumes: Mapped[List["ResumeModel"]] = relationship(
        "ResumeModel", back_populates="candidate", cascade="all, delete-orphan"
    )
    matches: Mapped[List["CandidateJobMatchModel"]] = relationship(
        "CandidateJobMatchModel", back_populates="candidate", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "education": self.education,
            "work_experience": self.work_experience,
            "skills": self.skills,
            "projects": self.projects,
            "years_of_experience": self.years_of_experience,
            "summary": self.summary,
            "direction_tags": self.direction_tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ==================== Resume ====================

class ResumeModel(Base):
    """ORM model for resume files and parsed content (many per candidate)."""

    __tablename__ = "resumes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    file_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    parsed: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default=statuses.RESUME_STATUS_UPLOADED, nullable=False
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    candidate: Mapped["CandidateModel"] = relationship(
        "CandidateModel", back_populates="resumes"
    )

    __table_args__ = (Index("ix_resumes_candidate_updated", "candidate_id", "updated_at"),)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "candidate_id": str(self.candidate_id),
            "file_path": self.file_path,
            "file_name": self.file_name,
            "parsed": self.parsed,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ==================== CandidateJobMatch ====================

class CandidateJobMatchModel(Base):
    """ORM model for job-candidate match result (scores and explanation)."""

    __tablename__ = "candidate_job_matches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    overall_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    skill_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    experience_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    education_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    semantic_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    industry_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    pros: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    cons: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    recommendation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    explanation: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default=statuses.MATCH_STATUS_COMPLETED, nullable=False
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    job: Mapped["JobModel"] = relationship("JobModel", back_populates="matches")
    candidate: Mapped["CandidateModel"] = relationship(
        "CandidateModel", back_populates="matches"
    )

    __table_args__ = (
        UniqueConstraint("job_id", "candidate_id", name="uq_candidate_job_match_pair"),
        Index("ix_match_job_created", "job_id", "created_at"),
        Index("ix_match_candidate_created", "candidate_id", "created_at"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "job_id": str(self.job_id),
            "candidate_id": str(self.candidate_id),
            "overall_score": self.overall_score,
            "skill_score": self.skill_score,
            "experience_score": self.experience_score,
            "education_score": self.education_score,
            "semantic_score": self.semantic_score,
            "industry_score": self.industry_score,
            "pros": self.pros,
            "cons": self.cons,
            "recommendation": self.recommendation,
            "explanation": self.explanation,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
        }


# ==================== Task (background jobs) ====================


class TaskModel(Base):
    """ORM model for background tasks (async operations)."""

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    resource_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True
    )
    resource_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(
        String(32), default=statuses.TASK_STATUS_PENDING, nullable=False, index=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_tasks_type_created", "task_type", "created_at"),
        Index("ix_tasks_resource", "resource_type", "resource_id"),
        Index("ix_tasks_status_created", "status", "created_at"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "task_type": self.task_type,
            "resource_type": self.resource_type,
            "resource_id": str(self.resource_id) if self.resource_id else None,
            "status": self.status,
            "error_message": self.error_message,
            "payload": self.payload,
            "result_summary": self.result_summary,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ==================== RAG Vector Index (V1.2) ====================


class VectorProfileModel(Base):
    """Persisted vector profile record for Job/Candidate.

    Notes:
    - By default we store vector as JSON list[float] for portability.
    - When VECTOR_STORE_BACKEND=pgvector (and not sqlite), we store vector in pgvector column type.
    - Unique per (entity_type, entity_id, profile_type) and always keep latest.
    """

    __tablename__ = "vector_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    profile_type: Mapped[str] = mapped_column(String(32), nullable=False, default="general")

    vector: Mapped[List[float]] = mapped_column(_VectorColumnType, nullable=False)
    meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    version: Mapped[int] = mapped_column(default=1, nullable=False)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default="available", nullable=False, index=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "entity_type",
            "entity_id",
            "profile_type",
            name="uq_vector_profile_entity",
        ),
        Index("ix_vector_profiles_entity", "entity_type", "entity_id"),
        Index("ix_vector_profiles_status_updated", "status", "updated_at"),
    )


# ==================== RAG Evaluation (reusable) ====================

class RAGEvaluationModel(Base):
    """
    ORM model for RAG evaluation results.
    message_id/conversation_id nullable when messages table not present.
    """

    __tablename__ = "rag_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)

    query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)

    context_precision: Mapped[Optional[float]] = mapped_column(nullable=True)
    context_recall: Mapped[Optional[float]] = mapped_column(nullable=True)
    faithfulness: Mapped[Optional[float]] = mapped_column(nullable=True)
    answer_relevancy: Mapped[Optional[float]] = mapped_column(nullable=True)

    evaluation_status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evaluation_duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_rag_evaluations_conv_created", "conversation_id", "created_at"),
        Index("ix_rag_evaluations_status", "evaluation_status"),
        Index("ix_rag_evaluations_user_created", "user_id", "created_at"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "message_id": str(self.message_id) if self.message_id else None,
            "conversation_id": str(self.conversation_id) if self.conversation_id else None,
            "user_id": self.user_id,
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "context": self.context,
            "response": self.response,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "evaluation_status": self.evaluation_status,
            "error_message": self.error_message,
            "evaluation_duration_ms": self.evaluation_duration_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
        }


# ==================== LLM Usage Log (reusable) ====================

class LLMUsageLogModel(Base):
    """ORM model for LLM usage statistics (no content stored)."""

    __tablename__ = "llm_usage_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    module_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    tool_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    __table_args__ = (
        Index("ix_llm_usage_user_created", "user_id", "created_at"),
        Index("ix_llm_usage_module_created", "module_name", "created_at"),
        Index("ix_llm_usage_conversation", "conversation_id"),
        Index("ix_llm_usage_model_created", "model_name", "created_at"),
        Index("ix_llm_usage_tool_created", "tool_name", "created_at"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "request_id": self.request_id,
            "module_name": self.module_name,
            "user_id": self.user_id,
            "conversation_id": str(self.conversation_id) if self.conversation_id else None,
            "model_name": self.model_name,
            "tool_name": self.tool_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
