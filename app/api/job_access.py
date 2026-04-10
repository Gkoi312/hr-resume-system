# app/api/job_access.py
"""Enforce job ownership for authenticated HR users."""

from typing import Optional

from fastapi import HTTPException, status

from app.database.models import UserModel
from app.schemas.job import JobResponse


def ensure_job_access(job: JobResponse, user: Optional[UserModel]) -> None:
    """If user is logged in, allow only jobs they created (legacy rows with no owner still allowed)."""
    if user is None:
        return
    if job.created_by_id is not None and job.created_by_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not allowed to access this job",
        )
