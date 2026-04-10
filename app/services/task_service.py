"""Service for background Task operations."""

import uuid
from typing import Any, Dict, List, Optional

from app import statuses
from app.database.repository.task_repository import task_repository
from app.schemas.task import TaskCreate, TaskResponse


class TaskService:
    """Service wrapper over TaskRepository with Pydantic responses."""

    async def create_task(self, payload: TaskCreate) -> TaskResponse:
        task = await task_repository.create(
            task_type=payload.task_type,
            resource_type=payload.resource_type,
            resource_id=payload.resource_id,
            payload=payload.payload,
            status=statuses.TASK_STATUS_PENDING,
        )
        return TaskResponse.model_validate(task)

    async def get_task(self, task_id: uuid.UUID) -> Optional[TaskResponse]:
        task = await task_repository.get_by_id(task_id)
        return TaskResponse.model_validate(task) if task else None

    async def list_recent(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TaskResponse]:
        tasks = await task_repository.list_recent(limit=limit, offset=offset)
        return [TaskResponse.model_validate(t) for t in tasks]

    async def mark_running(self, task_id: uuid.UUID) -> Optional[TaskResponse]:
        task = await task_repository.update(
            task_id,
            status=statuses.TASK_STATUS_RUNNING,
            error_message=None,
        )
        return TaskResponse.model_validate(task) if task else None

    async def mark_completed(
        self,
        task_id: uuid.UUID,
        result_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[TaskResponse]:
        task = await task_repository.update(
            task_id,
            status=statuses.TASK_STATUS_COMPLETED,
            result_summary=result_summary,
        )
        return TaskResponse.model_validate(task) if task else None

    async def mark_failed(
        self,
        task_id: uuid.UUID,
        error_message: str,
    ) -> Optional[TaskResponse]:
        task = await task_repository.update(
            task_id,
            status=statuses.TASK_STATUS_FAILED,
            error_message=error_message,
        )
        return TaskResponse.model_validate(task) if task else None


task_service = TaskService()

