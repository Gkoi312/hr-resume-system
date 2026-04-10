"""Tasks API:
  POST   /api/v1/tasks/resume-upload   - create async resume upload/parse/bind task
  POST   /api/v1/tasks/matching-run    - create async matching task
  GET    /api/v1/tasks/{id}            - get task by id
  GET    /api/v1/tasks                 - list recent tasks
"""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app import statuses
from app.api.deps import get_current_user_optional
from app.api.job_access import ensure_job_access
from app.database.models import UserModel
from app.schemas.task import TaskCreate, TaskResponse
from app.services.job_service import job_service
from app.services.matching_service import matching_service
from app.services.resume_service import resume_service
from app.services.task_service import task_service

router = APIRouter(prefix="/tasks", tags=["tasks"])


class MatchingTaskBody(BaseModel):
    """Body for async matching task."""

    job_id: uuid.UUID
    candidate_ids: Optional[List[uuid.UUID]] = None


@router.post("/resume-upload", response_model=TaskResponse)
async def create_resume_upload_task(
    file: UploadFile = File(...),
    candidate_id: Optional[uuid.UUID] = Form(None),
) -> TaskResponse:
    """
    Create an async task to upload/parse/bind a resume.

    返回 task 信息，调用方可轮询 /tasks/{id} 获取进度和最终结果（resume_id）。
    """
    # 将文件保存到本地路径，由后台 worker 从磁盘读取并处理。
    import os
    from pathlib import Path

    content = await file.read()
    orig_name = file.filename or "resume.pdf"
    upload_dir = os.getenv("RESUME_UPLOAD_DIR", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    suffix = Path(orig_name).suffix or ".pdf"
    stored_name = f"{uuid.uuid4()}{suffix}"
    full_path = os.path.join(upload_dir, stored_name)
    with open(full_path, "wb") as f:
        f.write(content)

    task = await task_service.create_task(
        TaskCreate(
            task_type="resume_upload",
            resource_type="resume",
            resource_id=None,
            payload={
                "file_path": full_path,
                "original_name": orig_name,
                "candidate_id": str(candidate_id) if candidate_id else None,
            },
        )
    )
    # 实际执行由独立 worker 进程轮询 tasks 表并处理，不在 API 进程内直接运行。
    return task


@router.post("/matching-run", response_model=TaskResponse)
async def create_matching_task(
    body: MatchingTaskBody,
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
) -> TaskResponse:
    """
    Create an async task to run matching for a job.

    返回 task 信息，调用方可轮询 /tasks/{id} 获取进度和 match_count。
    """
    job = await job_service.get_job(body.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    ensure_job_access(job, current_user)
    task = await task_service.create_task(
        TaskCreate(
            task_type="matching_run",
            resource_type="job",
            resource_id=body.job_id,
            payload={
                "job_id": str(body.job_id),
                "candidate_ids": [str(cid) for cid in body.candidate_ids]
                if body.candidate_ids
                else None,
            },
        )
    )
    # 实际执行由独立 worker 进程轮询 tasks 表并处理，不在 API 进程内直接运行。
    return task


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: uuid.UUID) -> TaskResponse:
    """Get a task by id."""
    task = await task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("", response_model=List[TaskResponse])
async def list_tasks(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[TaskResponse]:
    """List recent tasks."""
    return await task_service.list_recent(limit=limit, offset=offset)

