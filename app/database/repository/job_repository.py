# app/database/repository/job_repository.py
"""Repository for Job CRUD and queries."""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app import statuses
from app.database.models import JobModel
from app.database.session import get_session_context


class JobRepository:
    """Data access for jobs."""

    async def create(
        self,
        title: str,
        created_by_id: Optional[uuid.UUID] = None,
        raw_jd_text: Optional[str] = None,
        structured: Optional[Dict[str, Any]] = None,
        status: str = statuses.JOB_STATUS_DRAFT,
        error_message: Optional[str] = None,
    ) -> JobModel:
        async with get_session_context() as session:
            job = JobModel(
                title=title,
                created_by_id=created_by_id,
                raw_jd_text=raw_jd_text,
                structured=structured,
                status=status,
                error_message=error_message,
            )
            session.add(job)
            await session.flush()
            await session.refresh(job)
            return job

    async def get_by_id(self, job_id: uuid.UUID) -> Optional[JobModel]:
        async with get_session_context() as session:
            stmt = select(JobModel).where(JobModel.id == job_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_by_id_with_matches(self, job_id: uuid.UUID) -> Optional[JobModel]:
        async with get_session_context() as session:
            stmt = (
                select(JobModel)
                .options(selectinload(JobModel.matches))
                .where(JobModel.id == job_id)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def list(
        self,
        created_by_id: Optional[uuid.UUID] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[JobModel]:
        async with get_session_context() as session:
            stmt = select(JobModel).order_by(JobModel.updated_at.desc())
            if created_by_id is not None:
                stmt = stmt.where(JobModel.created_by_id == created_by_id)
            if status is not None:
                stmt = stmt.where(JobModel.status == status)
            stmt = stmt.limit(limit).offset(offset)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update(
        self,
        job_id: uuid.UUID,
        patch: Dict[str, Any],
    ) -> Optional[JobModel]:
        allowed = frozenset(
            {"title", "raw_jd_text", "structured", "status", "error_message"}
        )
        data = {k: v for k, v in patch.items() if k in allowed}
        async with get_session_context() as session:
            stmt = select(JobModel).where(JobModel.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if not job:
                return None
            for key, value in data.items():
                setattr(job, key, value)
            await session.flush()
            await session.refresh(job)
            return job

    async def delete(self, job_id: uuid.UUID) -> bool:
        async with get_session_context() as session:
            stmt = select(JobModel).where(JobModel.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if not job:
                return False
            await session.delete(job)
            return True


job_repository = JobRepository()
