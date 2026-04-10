# app/api/v1/endpoints/candidates.py
"""
Candidates API:
  GET    /api/v1/candidates/{id}   - get candidate
  GET    /api/v1/candidates        - list candidates

  候选人资源的 HTTP 接口层，提供 创建 / 查询单个 / 分页搜索 / 部分更新 四个标准 REST 动作
  参数校验和响应结构由 schemas.candidate 里的 Pydantic 模型定义
  实际的数据读写逻辑在 candidate_service 内部。
"""

import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Query

from app.schemas.candidate import CandidateCreate, CandidateResponse, CandidateUpdate
from app.services.candidate_service import candidate_service

router = APIRouter(prefix="/candidates", tags=["candidates"])


@router.post("", response_model=CandidateResponse)
async def create_candidate(payload: CandidateCreate):
    """Create a candidate from structured data (used by tools/scripts)."""
    candidate = await candidate_service.create_candidate(payload)
    if not candidate:
        raise HTTPException(status_code=500, detail="Failed to create candidate")
    return candidate


@router.get("/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(candidate_id: uuid.UUID):
    """Get a candidate by id."""
    candidate = await candidate_service.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


# 在前端列表页做「搜索人才库、按技能/行业筛选、按更新时间排序」的核心入口。
@router.get("", response_model=List[CandidateResponse])
async def list_candidates(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    keyword: str | None = Query(None, description="Search in name/email/summary"),
    skill: str | None = Query(None, description="Filter by skill"),
    industry: str | None = Query(None, description="Filter by industry text"),
    education: str | None = Query(None, description="Filter by education text"),
    min_years: int | None = Query(None, ge=0, description="Minimum years of experience"),
    sort_by: str = Query(
        "updated_at",
        pattern="^(updated_at|years_of_experience|name)$",
        description="Sort field",
    ),
    sort_order: str = Query(
        "desc", pattern="^(asc|desc)$", description="Sort order (asc|desc)"
    ),
):
    """List candidates with optional search filters."""
    return await candidate_service.list_candidates(
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


@router.patch("/{candidate_id}", response_model=CandidateResponse)
async def update_candidate(candidate_id: uuid.UUID, update: CandidateUpdate):
    """Partially update a candidate."""
    candidate = await candidate_service.update_candidate(candidate_id, update)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate
