"""Background worker that polls tasks table and executes pending tasks.

Usage (example):

    python -m app.workers.task_worker

This worker is safe to run with multiple instances concurrently thanks to
`SELECT ... FOR UPDATE SKIP LOCKED` in TaskRepository.acquire_next_pending.
"""

import asyncio
import logging
import os
import uuid
from typing import List

from app import statuses
from app.database.repository.task_repository import task_repository
from app.database.session import close_db, init_db
from app.services.task_service import task_service
from app.tasks.match_task import handle_matching_run
from app.tasks.rag_index_task import (
    handle_candidate_profile_index_build,
    handle_job_profile_index_build,
)
from app.tasks.resume_task import handle_resume_upload

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = float(os.getenv("TASK_WORKER_POLL_INTERVAL", "2.0"))
MAX_CONCURRENCY = int(os.getenv("TASK_WORKER_MAX_CONCURRENCY", "4"))
WORKER_ID = os.getenv("TASK_WORKER_ID", str(uuid.uuid4()))


async def dispatch_task(task) -> None:
    """Dispatch task to concrete handler based on task_type."""
    if task.task_type == "matching_run":
        await handle_matching_run(task)
    elif task.task_type == "resume_upload":
        await handle_resume_upload(task)
    elif task.task_type == "job_profile_index_build":
        await handle_job_profile_index_build(task)
    elif task.task_type == "candidate_profile_index_build":
        await handle_candidate_profile_index_build(task)
    else:
        raise ValueError(f"Unknown task_type: {task.task_type}")


async def worker_loop() -> None:
    """Main worker loop: poll for pending tasks and execute them."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Task worker %s starting...", WORKER_ID)

    await init_db()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    try:
        while True:
            task = await task_repository.acquire_next_pending(
                task_types=[
                    "matching_run",
                    "resume_upload",
                    "job_profile_index_build",
                    "candidate_profile_index_build",
                ],
            )
            if not task:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            logger.info("Worker %s picked task %s (%s)", WORKER_ID, task.id, task.task_type)

            async def _run(t):
                async with sem:
                    try:
                        await dispatch_task(t)
                        logger.info("Task %s completed", t.id)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Task %s failed: %s", t.id, exc)
                        await task_service.mark_failed(t.id, str(exc))

            asyncio.create_task(_run(task))
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(worker_loop())

