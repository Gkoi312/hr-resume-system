# app/database/repository/resume_repository.py
"""Repository for Resume CRUD and queries."""

import uuid
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select

from app import statuses
from app.database.models import ResumeModel
from app.database.session import get_session_context


class ResumeRepository:
    """Data access for resumes."""

    _UNSET = object()

    async def create(
        self,
        candidate_id: uuid.UUID,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        parsed: Optional[Dict[str, Any]] = None,
        status: str = statuses.RESUME_STATUS_UPLOADED,
    ) -> ResumeModel:
        async with get_session_context() as session:
            resume = ResumeModel(
                candidate_id=candidate_id,
                file_path=file_path,
                file_name=file_name,
                parsed=parsed,
                status=status,
            )
            session.add(resume)
            await session.flush()
            await session.refresh(resume)
            return resume

    async def get_by_id(self, resume_id: uuid.UUID) -> Optional[ResumeModel]:
        async with get_session_context() as session:
            stmt = select(ResumeModel).where(ResumeModel.id == resume_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def list_by_candidate(
        self,
        candidate_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ResumeModel]:
        async with get_session_context() as session:
            stmt = (
                select(ResumeModel)
                .where(ResumeModel.candidate_id == candidate_id)
                .order_by(ResumeModel.updated_at.desc()) # 倒序排序，最近更新的在前
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def list_by_candidate_ids(
        self,
        candidate_ids: Sequence[uuid.UUID],
    ) -> List[ResumeModel]:
        """All resume rows for the given candidates (any status). Caller filters parsed/status."""
        ids = list(candidate_ids)
        if not ids:
            return []
        async with get_session_context() as session:
            stmt = select(ResumeModel).where(ResumeModel.candidate_id.in_(ids))
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update(
        self,
        resume_id: uuid.UUID,
        *,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        parsed: object = _UNSET,
        status: Optional[str] = None,
        error_message: object = _UNSET,
    ) -> Optional[ResumeModel]:
        async with get_session_context() as session:
            stmt = select(ResumeModel).where(ResumeModel.id == resume_id)
            result = await session.execute(stmt)
            resume = result.scalar_one_or_none()
            if not resume:
                return None
            if file_path is not None:
                resume.file_path = file_path
            if file_name is not None:
                resume.file_name = file_name
            if parsed is not self._UNSET:
                resume.parsed = parsed  # type: ignore[assignment]
            if status is not None:
                resume.status = status
            if error_message is not self._UNSET:
                resume.error_message = error_message  # type: ignore[assignment]
            await session.flush()
            await session.refresh(resume)
            return resume

    async def delete(self, resume_id: uuid.UUID) -> bool:
        async with get_session_context() as session:
            stmt = select(ResumeModel).where(ResumeModel.id == resume_id)
            result = await session.execute(stmt)
            resume = result.scalar_one_or_none()
            if not resume:
                return False
            await session.delete(resume)
            return True

    async def find_latest_reusable_by_file_name(
        self,
        file_name: str,
    ) -> Optional[ResumeModel]:
        """
        Latest resume row with this exact ``file_name`` that finished parse + candidate bind.
        Used by batch demos to skip re-parsing unchanged files (match by name only).
        """
        async with get_session_context() as session:
            stmt = (
                select(ResumeModel)
                .where(ResumeModel.file_name == file_name)
                .where(ResumeModel.status == statuses.RESUME_STATUS_CANDIDATE_BOUND)
                .order_by(ResumeModel.updated_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()


resume_repository = ResumeRepository()
