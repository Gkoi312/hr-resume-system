"""
Import job JSONs from `jobs/` and verify job vectorization chunks.

Main behaviors:
1) Load each `jobs/*.json`
2) Validate/normalize `structured` into `app.schemas.job.JobStructured`
3) Insert jobs via `app.services.job_service.job_service.create_job`
   (enqueue `job_profile_index_build` tasks automatically)
4) If `--wait`, wait for the latest indexing task completion
5) Verify `vector_profiles` contains expected job chunk profile_types:
   - `skill`
   - `role`
   - `resp` or `resp_*`

Usage:
  python scripts/import_jobs_and_verify_vectors.py --jobs-dir jobs
  python scripts/import_jobs_and_verify_vectors.py --jobs-dir jobs --wait
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_pgvector_env() -> None:
    """
    If user env requests VECTOR_STORE_BACKEND=pgvector but python package `pgvector`
    is missing, `app.database.models` will crash at import time.

    In that case, we fall back to JSON vector storage (`db`) for local verification.
    """
    if (os.getenv("VECTOR_STORE_BACKEND") or "").strip().lower() != "pgvector":
        return
    try:
        import pgvector  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        os.environ["VECTOR_STORE_BACKEND"] = "db"


_ensure_pgvector_env()

from app import statuses  # noqa: E402
from app.database.models import TaskModel, VectorProfileModel  # noqa: E402
from app.database.session import get_session_context, init_db  # noqa: E402
from app.schemas.job import JobCreate, JobStructured  # noqa: E402
from app.services.job_service import job_service  # noqa: E402


def _iter_job_json_files(jobs_dir: Path) -> Iterable[Path]:
    for p in sorted(jobs_dir.glob("*.json")):
        if p.is_file():
            yield p


def _normalize_job_payload(data: Dict[str, Any]) -> JobCreate:
    if "title" not in data:
        raise ValueError("job json missing `title`")

    raw = data.get("raw_jd_text")
    structured_in = data.get("structured") or None

    structured: Optional[JobStructured] = None
    if structured_in is not None:
        structured = JobStructured.model_validate(structured_in)

    status = data.get("status") or statuses.JOB_STATUS_ACTIVE
    return JobCreate(
        title=str(data["title"]),
        raw_jd_text=raw,
        structured=structured,
        status=status,
    )


async def _get_latest_index_task_id(job_id: uuid.UUID) -> Optional[uuid.UUID]:
    async with get_session_context() as session:
        stmt = (
            select(TaskModel)
            .where(
                TaskModel.task_type == "job_profile_index_build",
                TaskModel.resource_id == job_id,
            )
            .order_by(TaskModel.created_at.desc())
            .limit(1)
        )
        res = await session.execute(stmt)
        task = res.scalar_one_or_none()
        return task.id if task else None


async def _wait_task_completed(task_id: uuid.UUID, *, timeout_s: float) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        async with get_session_context() as session:
            stmt = select(TaskModel).where(TaskModel.id == task_id)
            res = await session.execute(stmt)
            task = res.scalar_one_or_none()
            if not task:
                await asyncio.sleep(0.5)
                continue
            if task.status == statuses.TASK_STATUS_COMPLETED:
                return True
            if task.status == statuses.TASK_STATUS_FAILED:
                return False
        await asyncio.sleep(1.0)
    return False


async def _fetch_job_vector_profile_types(job_id: uuid.UUID) -> List[str]:
    async with get_session_context() as session:
        stmt = (
            select(VectorProfileModel.profile_type)
            .where(
                VectorProfileModel.entity_type == "job",
                VectorProfileModel.entity_id == job_id,
                VectorProfileModel.status == "available",
            )
            .order_by(VectorProfileModel.profile_type.asc())
        )
        res = await session.execute(stmt)
        rows = list(res.scalars().all())
        return [str(x) for x in rows]


def _validate_expected_profile_types(profile_types: List[str]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if "skill" not in profile_types:
        missing.append("skill")
    if "role" not in profile_types:
        missing.append("role")
    has_resp = any(pt == "resp" or pt.startswith("resp_") for pt in profile_types)
    if not has_resp:
        missing.append("resp (resp or resp_*)")
    return (len(missing) == 0), missing


async def import_and_verify(jobs_dir: Path, *, wait: bool) -> None:
    await init_db()

    for jf in _iter_job_json_files(jobs_dir):
        job_data = json.loads(jf.read_text(encoding="utf-8"))
        payload = _normalize_job_payload(job_data)

        print(f"==> Import job: {jf.name} title={payload.title!r}")
        job = await job_service.create_job(payload, auto_analyze=False)
        if not job:
            print(f"    FAILED: create_job returned None for title={payload.title!r}")
            continue

        task_id = await _get_latest_index_task_id(job.id)
        if not task_id:
            print(f"    WARN: no job_profile_index_build task found for job_id={job.id}")
            continue

        if not wait:
            print(f"    job_id={job.id} task_id={task_id}")
            print("    SKIP vector verification (run with --wait after starting worker).")
            continue

        ok = await _wait_task_completed(task_id, timeout_s=180.0)
        if not ok:
            print(f"    job_id={job.id} task_id={task_id} ERROR: task not completed")
            continue

        profile_types = await _fetch_job_vector_profile_types(job.id)
        passed, missing = _validate_expected_profile_types(profile_types)
        print(f"    job_id={job.id}")
        print(f"    vector profile_types(available)={profile_types}")
        if passed:
            print("    PASS: expected embedding chunks exist")
        else:
            print(f"    FAIL: missing {missing}")
            print("    Hint: ensure `python -m app.workers.task_worker` is running and embedding provider works.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", type=str, default="jobs", help="Directory containing job JSON files")
    parser.add_argument("--wait", action="store_true", help="Wait for indexing tasks to complete")
    args = parser.parse_args()

    jobs_dir = Path(args.jobs_dir).resolve()
    if not jobs_dir.exists():
        raise SystemExit(f"jobs dir not found: {jobs_dir}")

    asyncio.run(import_and_verify(jobs_dir, wait=args.wait))


if __name__ == "__main__":
    main()

