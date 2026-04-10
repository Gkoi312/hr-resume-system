# app/database/repository/match_repository.py
"""Repository for CandidateJobMatch CRUD and queries."""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app import statuses
from app.database.models import CandidateJobMatchModel
from app.database.session import get_session_context


class MatchRepository:
    """Data access for candidate-job matches."""

    async def create(
        self,
        job_id: uuid.UUID,
        candidate_id: uuid.UUID,
        overall_score: Optional[float] = None,
        skill_score: Optional[float] = None,
        experience_score: Optional[float] = None,
        llm_quality_score: Optional[float] = None,
        education_score: Optional[float] = None,
        semantic_score: Optional[float] = None,
        industry_score: Optional[float] = None,
        pros: Optional[List[str]] = None,
        cons: Optional[List[str]] = None,
        recommendation: Optional[str] = None,
        explanation: Optional[Dict[str, Any]] = None,
        status: str = statuses.MATCH_STATUS_COMPLETED,
    ) -> CandidateJobMatchModel:
        async with get_session_context() as session:
            match = CandidateJobMatchModel(
                job_id=job_id,
                candidate_id=candidate_id,
                overall_score=overall_score,
                skill_score=skill_score,
                experience_score=experience_score,
                llm_quality_score=llm_quality_score,
                education_score=education_score,
                semantic_score=semantic_score,
                industry_score=industry_score,
                pros=pros,
                cons=cons,
                recommendation=recommendation,
                explanation=explanation,
                status=status,
            )
            session.add(match)
            await session.flush()
            await session.refresh(match)
            return match

    async def get_by_id(self, match_id: uuid.UUID) -> Optional[CandidateJobMatchModel]:
        async with get_session_context() as session:
            stmt = select(CandidateJobMatchModel).where(
                CandidateJobMatchModel.id == match_id
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_by_job_and_candidate(
        self, job_id: uuid.UUID, candidate_id: uuid.UUID
    ) -> Optional[CandidateJobMatchModel]:
        async with get_session_context() as session:
            stmt = select(CandidateJobMatchModel).where(
                CandidateJobMatchModel.job_id == job_id,
                CandidateJobMatchModel.candidate_id == candidate_id,
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def list_by_job(
        self,
        job_id: uuid.UUID,
        limit: int = 200,
        offset: int = 0,
        order_by_score: bool = True,
    ) -> List[CandidateJobMatchModel]:
        async with get_session_context() as session:
            stmt = (
                select(CandidateJobMatchModel)
                .where(CandidateJobMatchModel.job_id == job_id)
                .options(selectinload(CandidateJobMatchModel.candidate))
            )
            if order_by_score:
                stmt = stmt.order_by(
                    CandidateJobMatchModel.overall_score.desc().nullslast(),
                    CandidateJobMatchModel.created_at.desc(),
                )
            else:
                stmt = stmt.order_by(CandidateJobMatchModel.created_at.desc())
            stmt = stmt.limit(limit).offset(offset)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def list_by_candidate(
        self,
        candidate_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CandidateJobMatchModel]:
        async with get_session_context() as session:
            stmt = (
                select(CandidateJobMatchModel)
                .where(CandidateJobMatchModel.candidate_id == candidate_id)
                .options(selectinload(CandidateJobMatchModel.job))
                .order_by(CandidateJobMatchModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update(
        self,
        match_id: uuid.UUID,
        *,
        overall_score: Optional[float] = None,
        skill_score: Optional[float] = None,
        experience_score: Optional[float] = None,
        llm_quality_score: Optional[float] = None,
        education_score: Optional[float] = None,
        semantic_score: Optional[float] = None,
        industry_score: Optional[float] = None,
        pros: Optional[List[str]] = None,
        cons: Optional[List[str]] = None,
        recommendation: Optional[str] = None,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[CandidateJobMatchModel]:
        async with get_session_context() as session:
            stmt = select(CandidateJobMatchModel).where(
                CandidateJobMatchModel.id == match_id
            )
            result = await session.execute(stmt)
            match = result.scalar_one_or_none()
            if not match:
                return None
            if overall_score is not None:
                match.overall_score = overall_score
            if skill_score is not None:
                match.skill_score = skill_score
            if experience_score is not None:
                match.experience_score = experience_score
            if llm_quality_score is not None:
                match.llm_quality_score = llm_quality_score
            if education_score is not None:
                match.education_score = education_score
            if semantic_score is not None:
                match.semantic_score = semantic_score
            if industry_score is not None:
                match.industry_score = industry_score
            if pros is not None:
                match.pros = pros
            if cons is not None:
                match.cons = cons
            if recommendation is not None:
                match.recommendation = recommendation
            if status is not None:
                match.status = status
            if error_message is not None:
                match.error_message = error_message
            await session.flush()
            await session.refresh(match)
            return match

    async def delete(self, match_id: uuid.UUID) -> bool:
        async with get_session_context() as session:
            stmt = select(CandidateJobMatchModel).where(
                CandidateJobMatchModel.id == match_id
            )
            result = await session.execute(stmt)
            match = result.scalar_one_or_none()
            if not match:
                return False
            await session.delete(match)
            return True

    async def delete_by_job_and_candidate(
        self,
        job_id: uuid.UUID,
        candidate_id: uuid.UUID,
    ) -> int:
        """Delete matches for a specific (job, candidate) pair. Returns count deleted."""
        async with get_session_context() as session:
            stmt = select(CandidateJobMatchModel).where(
                CandidateJobMatchModel.job_id == job_id,
                CandidateJobMatchModel.candidate_id == candidate_id,
            )
            result = await session.execute(stmt)
            matches = result.scalars().all()
            for m in matches:
                await session.delete(m)
            return len(matches)
    async def delete_by_job(self, job_id: uuid.UUID) -> int:
        """Delete all matches for a job. Returns count deleted."""
        async with get_session_context() as session:
            stmt = select(CandidateJobMatchModel).where(
                CandidateJobMatchModel.job_id == job_id
            )
            result = await session.execute(stmt)
            matches = result.scalars().all()
            for m in matches:
                await session.delete(m)
            return len(matches)


match_repository = MatchRepository()
