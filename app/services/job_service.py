# app/services/job_service.py
"""
Job service: create, get, list, update jobs.
Depends: JobRepository, schemas (JobCreate, JobUpdate, JobResponse).
"""

import uuid
from typing import Any, Dict, List, Optional

from app import statuses
from app.database.repository.job_repository import job_repository
from app.schemas.job import JobCreate, JobResponse, JobUpdate
from app.schemas.task import TaskCreate
from app.services.task_service import task_service


class JobService:
    """Service for job (JD) operations."""

    async def create_job(
        self,
        payload: JobCreate,
        *,
        created_by_id: Optional[uuid.UUID] = None,
        auto_analyze: bool = False,
    ) -> Optional[JobResponse]:
        """Create a job. Uses payload.created_by_id if set, else created_by_id arg."""
        _ = auto_analyze  # parser path removed; keep API compatibility
        structured = payload.structured.model_dump(exclude_none=True) if payload.structured else None
        job_status = payload.status
        error_message: Optional[str] = None

        effective_owner = payload.created_by_id or created_by_id

        job = await job_repository.create(
            title=payload.title,
            created_by_id=effective_owner,
            raw_jd_text=payload.raw_jd_text,
            structured=structured,
            status=job_status,
            error_message=error_message,
        )
        if job and (job.structured or job.raw_jd_text):
            await task_service.create_task(
                TaskCreate(
                    task_type="job_profile_index_build",
                    resource_type="job",
                    resource_id=job.id,
                    payload={"job_id": str(job.id)},
                )
            )
        return JobResponse.model_validate(job) if job else None

    async def get_job(self, job_id: uuid.UUID) -> Optional[JobResponse]:
        """Get a job by id."""
        job = await job_repository.get_by_id(job_id)
        return JobResponse.model_validate(job) if job else None

    async def list_jobs(
        self,
        created_by_id: Optional[uuid.UUID] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[JobResponse]:
        """List jobs with optional filters."""
        jobs = await job_repository.list(
            created_by_id=created_by_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        return [JobResponse.model_validate(j) for j in jobs]

    async def update_job(
        self,
        job_id: uuid.UUID,
        payload: JobUpdate,
        *,
        auto_analyze: bool = False,
    ) -> Optional[JobResponse]:
        """Update a job by id. Only fields set in the request body are updated."""
        _ = auto_analyze  # parser path removed; keep API compatibility
        patch: Dict[str, Any] = payload.model_dump(exclude_unset=True)
        if "structured" in patch and patch["structured"] is not None:
            structured_val = patch["structured"]
            if hasattr(structured_val, "model_dump"):
                patch["structured"] = structured_val.model_dump(exclude_none=True)
            elif isinstance(structured_val, dict):
                patch["structured"] = {
                    k: v for k, v in structured_val.items() if v is not None
                }

        job = await job_repository.update(job_id, patch)
        if job and (job.structured or job.raw_jd_text):
            await task_service.create_task(
                TaskCreate(
                    task_type="job_profile_index_build",
                    resource_type="job",
                    resource_id=job.id,
                    payload={"job_id": str(job.id)},
                )
            )
        return JobResponse.model_validate(job) if job else None

    async def mark_failed(
        self,
        job_id: uuid.UUID,
        error_message: str,
    ) -> Optional[JobResponse]:
        """Mark job as failed with an error message."""
        job = await job_repository.update(
            job_id,
            {
                "status": statuses.JOB_STATUS_FAILED,
                "error_message": error_message,
            },
        )
        return JobResponse.model_validate(job) if job else None

    async def retry_analyze_job(self, job_id: uuid.UUID) -> Optional[JobResponse]:
        """Parser flow removed. Endpoint kept for backward compatibility."""
        job = await job_repository.get_by_id(job_id)
        return JobResponse.model_validate(job) if job else None


job_service = JobService()
