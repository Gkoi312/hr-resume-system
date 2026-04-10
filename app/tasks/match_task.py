"""Task handler for matching_run tasks."""

import uuid
from typing import Any, Dict, List, Optional

from app.services.matching_service import matching_service
from app.services.task_service import task_service


async def handle_matching_run(task) -> None:
    """
    Execute a matching_run task.

    Expects payload:
      {
        "job_id": "<uuid>",
        "candidate_ids": ["<uuid>", ...] | null
      }
    """
    payload = task.payload or {}
    job_id_str: Optional[str] = payload.get("job_id")
    if not job_id_str:
        raise ValueError("matching_run task missing job_id in payload")
    job_id = uuid.UUID(job_id_str)

    candidate_ids_raw = payload.get("candidate_ids") or None
    candidate_ids: Optional[List[uuid.UUID]] = None
    if candidate_ids_raw:
        candidate_ids = [uuid.UUID(cid) for cid in candidate_ids_raw]

    results = await matching_service.run_matching(
        job_id=job_id,
        candidate_ids=candidate_ids,
    )
    await task_service.mark_completed(
        task.id,
        result_summary={
            "job_id": job_id_str,
            "match_count": len(results),
        },
    )

