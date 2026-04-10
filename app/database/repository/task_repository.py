"""Repository for background Task CRUD and queries."""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app import statuses
from app.database.models import TaskModel
from app.database.session import get_session_context


class TaskRepository:
    """Data access for background tasks."""

    async def create(
        self,
        task_type: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[uuid.UUID] = None,
        payload: Optional[Dict[str, Any]] = None,
        status: str = statuses.TASK_STATUS_PENDING,
    ) -> TaskModel:
        async with get_session_context() as session:
            task = TaskModel(
                task_type=task_type,
                resource_type=resource_type,
                resource_id=resource_id,
                status=status,
                payload=payload,
            )
            session.add(task)
            await session.flush()
            await session.refresh(task)
            return task

    async def get_by_id(self, task_id: uuid.UUID) -> Optional[TaskModel]:
        async with get_session_context() as session:
            stmt = select(TaskModel).where(TaskModel.id == task_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def update(
        self,
        task_id: uuid.UUID,
        *,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
        result_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[TaskModel]:
        async with get_session_context() as session:
            stmt = select(TaskModel).where(TaskModel.id == task_id)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            if not task:
                return None
            if status is not None:
                task.status = status
            if error_message is not None:
                task.error_message = error_message
            if result_summary is not None:
                task.result_summary = result_summary
            await session.flush()
            await session.refresh(task)
            return task

    async def list_recent(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TaskModel]:
        async with get_session_context() as session:
            stmt = (
                select(TaskModel)
                .order_by(TaskModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def acquire_next_pending(
        self,
        task_types: Optional[List[str]] = None,
    ) -> Optional[TaskModel]:
        """
        Atomically fetch and mark the next pending task as running.

        Uses SELECT ... FOR UPDATE SKIP LOCKED to ensure that multiple workers
        do not pick the same task concurrently.
        """
        async with get_session_context() as session:
            stmt = (
                select(TaskModel)
                .where(TaskModel.status == statuses.TASK_STATUS_PENDING)
                .order_by(TaskModel.created_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if task_types:
                stmt = stmt.where(TaskModel.task_type.in_(task_types))
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            if not task:
                return None
            task.status = statuses.TASK_STATUS_RUNNING
            task.error_message = None
            await session.flush()
            await session.refresh(task)
            return task


task_repository = TaskRepository()

