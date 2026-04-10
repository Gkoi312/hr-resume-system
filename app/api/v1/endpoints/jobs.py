# app/api/v1/endpoints/jobs.py
"""
Jobs API:
  POST   /api/v1/jobs          - create job (structured-first)
  POST   /api/v1/jobs/upload   - upload JD file -> extract raw_jd_text + structured
  GET    /api/v1/jobs/{id}     - get job
  GET    /api/v1/jobs          - list jobs
  PATCH  /api/v1/jobs/{id}     - update job
"""

import uuid
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app import statuses
from app.api.deps import get_current_user_optional
from app.api.job_access import ensure_job_access
from app.database.models import UserModel
from app.parsers.job_parser.pipeline import (
    LLMParseError,
    OCRParseError,
    UnsupportedFileTypeError,
    parse_job_document,
)
from app.schemas.job import JobCreate, JobResponse, JobUpdate
from app.services.job_service import job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse)
async def create_job(
    payload: JobCreate,
    auto_analyze: bool = Query(False, description="Deprecated; parser path removed"),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Create a job (JD) from structured payload."""
    if current_user is not None:
        payload.created_by_id = current_user.id
    job = await job_service.create_job(payload, auto_analyze=auto_analyze)
    if not job:
        raise HTTPException(status_code=500, detail="Failed to create job")
    return job


@router.post("/upload", response_model=JobResponse)
async def upload_job(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    status: str = Form(statuses.JOB_STATUS_ACTIVE),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Upload JD file then parse to canonical structured JSON."""
    content = await file.read()
    file_name = file.filename or "job.txt"
    try:
        parsed = await parse_job_document(content, file_name)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=415,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except OCRParseError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except LLMParseError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    # Keep a temp copy for debugging traceability.
    upload_dir = Path(tempfile.gettempdir()) / "hr_job_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_name = f"{uuid.uuid4().hex}_{Path(file_name).name}"
    tmp_path = upload_dir / tmp_name
    tmp_path.write_bytes(content)

    payload = JobCreate(
        title=(title or parsed["structured"].get("job_title") or Path(file_name).stem),
        raw_jd_text=parsed["raw_jd_text"],
        structured=parsed["structured"],
        status=status,
        created_by_id=current_user.id if current_user is not None else None,
    )
    job = await job_service.create_job(payload, auto_analyze=False)
    if not job:
        raise HTTPException(status_code=500, detail="Failed to create uploaded job")
    return job


@router.get("", response_model=list[JobResponse])
async def list_jobs(
    created_by_id: Optional[uuid.UUID] = Query(
        None,
        description="Filter by creator (ignored when Bearer token is present)",
    ),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """List jobs with optional filters. Authenticated users only see their own jobs."""
    scope_id = current_user.id if current_user is not None else created_by_id
    return await job_service.list_jobs(
        created_by_id=scope_id,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: uuid.UUID,
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Get a job by id."""
    job = await job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    ensure_job_access(job, current_user)
    return job


@router.patch("/{job_id}", response_model=JobResponse)
async def update_job(
    job_id: uuid.UUID,
    payload: JobUpdate,
    auto_analyze: bool = Query(False, description="Deprecated; parser path removed"),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Update a job (partial)."""
    existing = await job_service.get_job(job_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Job not found")
    ensure_job_access(existing, current_user)
    job = await job_service.update_job(job_id, payload, auto_analyze=auto_analyze)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/{job_id}/retry-analyze", response_model=JobResponse)
async def retry_analyze_job(
    job_id: uuid.UUID,
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Deprecated no-op endpoint kept for compatibility."""
    existing = await job_service.get_job(job_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Job not found")
    ensure_job_access(existing, current_user)
    job = await job_service.retry_analyze_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job