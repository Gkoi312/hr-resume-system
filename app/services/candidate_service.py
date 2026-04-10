# app/services/candidate_service.py
"""Minimal candidate service for API: get, list."""

import uuid
from typing import List, Optional

from app.database.repository.candidate_repository import candidate_repository
from app.schemas.candidate import CandidateCreate, CandidateResponse, CandidateUpdate
from app.schemas.task import TaskCreate
from app.services.task_service import task_service

class CandidateService:
    """Service for candidate operations."""

    async def create_candidate(
        self,
        payload: CandidateCreate,
    ) -> Optional[CandidateResponse]:
        """Create a candidate from structured data."""
        candidate = await candidate_repository.create(
            name=payload.name,
            email=payload.email,
            phone=payload.phone,
            education=payload.education,
            work_experience=payload.work_experience,
            skills=payload.skills,
            projects=payload.projects,
            years_of_experience=payload.years_of_experience,
            summary=payload.summary,
            direction_tags=payload.direction_tags,
        )
        # V1.2: enqueue index build (best-effort)
        if candidate:
            await task_service.create_task(
                TaskCreate(
                    task_type="candidate_profile_index_build",
                    resource_type="candidate",
                    resource_id=candidate.id,
                    payload={"candidate_id": str(candidate.id)},
                )
            )
        return CandidateResponse.model_validate(candidate) if candidate else None

    async def get_candidate(self, candidate_id: uuid.UUID) -> Optional[CandidateResponse]:
        """Get a candidate by id."""
        candidate = await candidate_repository.get_by_id(candidate_id)
        return CandidateResponse.model_validate(candidate) if candidate else None

    async def list_candidates(
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
    ) -> List[CandidateResponse]:
        """List candidates with optional search filters."""
        candidates = await candidate_repository.list(
            limit=limit,
            offset=offset,
            keyword=keyword,
            skill=skill,
            industry=industry,
            education=education,
            min_years=min_years,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return [CandidateResponse.model_validate(c) for c in candidates]

    async def update_candidate(
        self,
        candidate_id: uuid.UUID,
        update: CandidateUpdate,
    ) -> Optional[CandidateResponse]:
        """Partially update a candidate."""
        candidate = await candidate_repository.update(
            candidate_id,
            name=update.name,
            email=update.email,
            phone=update.phone,
            education=update.education,
            work_experience=update.work_experience,
            skills=update.skills,
            projects=update.projects,
            years_of_experience=update.years_of_experience,
            summary=update.summary,
            direction_tags=update.direction_tags,
        )
        # V1.2: enqueue index build (best-effort)
        if candidate:
            await task_service.create_task(
                TaskCreate(
                    task_type="candidate_profile_index_build",
                    resource_type="candidate",
                    resource_id=candidate.id,
                    payload={"candidate_id": str(candidate.id)},
                )
            )
        return CandidateResponse.model_validate(candidate) if candidate else None


candidate_service = CandidateService()
