# app/database/repository/candidate_repository.py
"""Repository for Candidate CRUD and queries."""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import Select, String, and_, func, or_, select
from sqlalchemy.orm import selectinload

from app.database.models import CandidateModel
from app.database.session import get_session_context


def _normalize_email(email: Optional[str]) -> Optional[str]:
    """Single canonical form for upsert / lookup (trim + lower)."""
    if email is None:
        return None
    s = str(email).strip()
    return s.lower() if s else None


class CandidateRepository:
    """Data access for candidates."""

    async def create(
        self,
        name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        education: Optional[List[Dict[str, Any]]] = None,
        work_experience: Optional[List[Dict[str, Any]]] = None,
        skills: Optional[List[str]] = None,
        projects: Optional[List[Dict[str, Any]]] = None,
        years_of_experience: Optional[int] = None,
        summary: Optional[str] = None,
        direction_tags: Optional[List[str]] = None,
    ) -> CandidateModel:
        async with get_session_context() as session:
            candidate = CandidateModel(
                name=name,
                email=_normalize_email(email),
                phone=phone,
                education=education,
                work_experience=work_experience,
                skills=skills,
                projects=projects,
                years_of_experience=years_of_experience,
                summary=summary,
                direction_tags=direction_tags,
            )
            session.add(candidate)
            await session.flush()
            await session.refresh(candidate)
            return candidate

    async def get_by_id(self, candidate_id: uuid.UUID) -> Optional[CandidateModel]:
        async with get_session_context() as session:
            stmt = select(CandidateModel).where(CandidateModel.id == candidate_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    # 给「候选人详情页要带多份简历」这类场景用；纯 get_by_id 不会自动加载 resumes
    async def get_by_id_with_resumes(
        self, candidate_id: uuid.UUID
    ) -> Optional[CandidateModel]:
        async with get_session_context() as session:
            stmt = (
                select(CandidateModel)
                .options(selectinload(CandidateModel.resumes))
                .where(CandidateModel.id == candidate_id)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[CandidateModel]:
        norm = _normalize_email(email)
        if not norm:
            return None
        async with get_session_context() as session:
            stmt = (
                select(CandidateModel)
                .where(
                    func.lower(func.trim(CandidateModel.email)) == norm,
                )
                .order_by(CandidateModel.updated_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalars().first()

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        keyword: Optional[str] = None,
        skill: Optional[str] = None,
        industry: Optional[str] = None,
        education: Optional[str] = None,
        min_years: Optional[int] = None,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> List[CandidateModel]:
        async with get_session_context() as session:
            stmt: Select[tuple[CandidateModel]] = select(CandidateModel)

            conditions = []
            if keyword:
                pattern = f"%{keyword}%"
                conditions.append(
                    or_(
                        CandidateModel.name.ilike(pattern),
                        CandidateModel.email.ilike(pattern),
                        CandidateModel.summary.ilike(pattern),
                    )
                )
            if min_years is not None:
                conditions.append(CandidateModel.years_of_experience >= min_years)
            # For JSON fields we start with simple text search to keep implementation small.
            if skill:
                pattern = f"%{skill.lower()}%"
                conditions.append(
                    func.lower(func.cast(CandidateModel.skills, String)).like(pattern)
                )
            if education:
                pattern = f"%{education.lower()}%"
                conditions.append(
                    func.lower(func.cast(CandidateModel.education, String)).like(
                        pattern
                    )
                )
            if industry:
                pattern = f"%{industry.lower()}%"
                conditions.append(
                    func.lower(func.cast(CandidateModel.work_experience, String)).like(
                        pattern
                    )
                )

            if conditions:
                stmt = stmt.where(and_(*conditions))

            if sort_by == "years_of_experience":
                order_col = CandidateModel.years_of_experience
            elif sort_by == "name":
                order_col = CandidateModel.name
            else:
                order_col = CandidateModel.updated_at

            if sort_order.lower() == "asc":
                stmt = stmt.order_by(order_col.asc())
            else:
                stmt = stmt.order_by(order_col.desc())

            stmt = stmt.limit(limit).offset(offset)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update(
        self,
        candidate_id: uuid.UUID,
        *,
        name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        education: Optional[List[Dict[str, Any]]] = None,
        work_experience: Optional[List[Dict[str, Any]]] = None,
        skills: Optional[List[str]] = None,
        projects: Optional[List[Dict[str, Any]]] = None,
        years_of_experience: Optional[int] = None,
        summary: Optional[str] = None,
        direction_tags: Optional[List[str]] = None,
    ) -> Optional[CandidateModel]:
        async with get_session_context() as session:
            stmt = select(CandidateModel).where(CandidateModel.id == candidate_id)
            result = await session.execute(stmt)
            candidate = result.scalar_one_or_none()
            if not candidate:
                return None
            if name is not None:
                candidate.name = name
            if email is not None:
                candidate.email = _normalize_email(email)
            if phone is not None:
                candidate.phone = phone
            if education is not None:
                candidate.education = education
            if work_experience is not None:
                candidate.work_experience = work_experience
            if skills is not None:
                candidate.skills = skills
            if projects is not None:
                candidate.projects = projects
            if years_of_experience is not None:
                candidate.years_of_experience = years_of_experience
            if summary is not None:
                candidate.summary = summary
            if direction_tags is not None:
                candidate.direction_tags = direction_tags
            await session.flush()
            await session.refresh(candidate)
            return candidate

    async def delete(self, candidate_id: uuid.UUID) -> bool:
        async with get_session_context() as session:
            stmt = select(CandidateModel).where(CandidateModel.id == candidate_id)
            result = await session.execute(stmt)
            candidate = result.scalar_one_or_none()
            if not candidate:
                return False
            await session.delete(candidate)
            return True


candidate_repository = CandidateRepository()
