# app/api/v1/endpoints/matching.py
"""
Matching API:
  POST   /api/v1/matching/run                  - run matching for a job
  POST   /api/v1/matching/education-filter      - candidates passing resume education gate only
  GET    /api/v1/matching/job/{id}             - get matches by job
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.deps import get_current_user_optional
from app.api.job_access import ensure_job_access
from app.database.models import UserModel
from app.database.repository.match_repository import match_repository
from app.schemas.match import (
    EducationFilterResponse,
    MatchResponse,
    MatchWithCandidate,
)
from app.services.job_service import job_service
from app.services.matching_service import matching_service

router = APIRouter(prefix="/matching", tags=["matching"])


class MatchingRunBody(BaseModel):
    """Body for run: job_id, optional candidate_ids."""

    job_id: uuid.UUID
    candidate_ids: Optional[List[uuid.UUID]] = None


class MatchingRetryBody(BaseModel):
    """Body for retrying matching for a single candidate on a job."""

    job_id: uuid.UUID
    candidate_id: uuid.UUID


@router.post("/run", response_model=list[MatchResponse])
async def run_matching(
    body: MatchingRunBody,
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """
    Run matching for a job. Optionally pass candidate_ids to limit
    candidates; if omitted, all candidates in DB are matched.
    """
    job = await job_service.get_job(body.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    ensure_job_access(job, current_user)
    results = await matching_service.run_matching(
        job_id=body.job_id,
        candidate_ids=body.candidate_ids,
        delete_old=True,
    )
    return results


@router.post("/education-filter", response_model=EducationFilterResponse)
async def education_filter_candidates(
    body: MatchingRunBody,
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """
    List candidates who pass the resume structured education gate for this job.
    Does not run semantic matching or write match rows.
    """
    job = await job_service.get_job(body.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    ensure_job_access(job, current_user)
    return await matching_service.list_candidates_passing_education_gate(
        job_id=body.job_id,
        candidate_ids=body.candidate_ids,
    )


@router.post("/retry-for-candidate", response_model=list[MatchResponse])
async def retry_for_candidate(
    body: MatchingRetryBody,
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """
    Re-run matching for a single (job, candidate) pair.
    Intended for 'edit candidate then recompute this job' flows.
    """
    job = await job_service.get_job(body.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    ensure_job_access(job, current_user)
    # 删除旧的 match 记录（仅这一对 job+candidate）
    await match_repository.delete_by_job_and_candidate(
        job_id=body.job_id,
        candidate_id=body.candidate_id,
    )
    # 只针对该 candidate 运行一次匹配；不清空其他候选人的匹配结果
    results = await matching_service.run_matching(
        job_id=body.job_id,
        candidate_ids=[body.candidate_id],
        delete_old=False,
    )
    return results


@router.get("/job/{job_id}", response_model=list[MatchWithCandidate])
async def get_matches_by_job(
    job_id: uuid.UUID,
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Get match list for a job (ordered by overall_score desc)."""
    job = await job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    ensure_job_access(job, current_user)
    return await matching_service.get_matches_by_job(
        job_id=job_id,
        limit=limit,
        offset=offset,
    )
