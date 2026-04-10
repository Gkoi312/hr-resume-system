# app/database/repository/__init__.py

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
