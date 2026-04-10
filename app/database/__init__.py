# app/database/__init__.py
"""
Database module for HR Resume Screening System.
Exposes session, models, and repositories.
"""

from app.database.session import (
    async_session_factory,
    get_async_session,
    get_session_context,
    init_db,
    close_db,
)
from app.database.models import (
    Base,
    UserModel,
    JobModel,
    ResumeModel,
    CandidateModel,
    CandidateJobMatchModel,
    RAGEvaluationModel,
    LLMUsageLogModel,
)

from app.database.repository.job_repository import JobRepository, job_repository
from app.database.repository.resume_repository import ResumeRepository, resume_repository
from app.database.repository.candidate_repository import (
    CandidateRepository,
    candidate_repository,
)
from app.database.repository.match_repository import MatchRepository, match_repository
from app.database.repository.evaluation_repository import (
    EvaluationRepository,
    evaluation_repository,
)
from app.database.repository.llm_usage_repository import (
    LLMUsageRepository,
    llm_usage_repository,
)

__all__ = [
    "async_session_factory",
    "get_async_session",
    "get_session_context",
    "init_db",
    "close_db",
    "Base",
    "UserModel",
    "JobModel",
    "ResumeModel",
    "CandidateModel",
    "CandidateJobMatchModel",
    "RAGEvaluationModel",
    "LLMUsageLogModel",
    "JobRepository",
    "job_repository",
    "ResumeRepository",
    "resume_repository",
    "CandidateRepository",
    "candidate_repository",
    "MatchRepository",
    "match_repository",
    "EvaluationRepository",
    "evaluation_repository",
    "LLMUsageRepository",
    "llm_usage_repository",
]
