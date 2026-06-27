"""
Import candidate JSONs from `testOCR/output/*_candidate_profile.json` and
verify candidate vectorization chunks exist in `vector_profiles`.

What it does:
1) Load each candidate json
2) Upsert candidate via `candidate_service` (create or update)
3) This enqueues `candidate_profile_index_build` tasks
4) If `--wait` is provided, wait for the latest indexing task to complete
5) Verify vector profiles for the candidate:
   - Always: `cand_role`
   - If skills present: `skill`
   - If projects present: `proj_{i}` (by original list index)
   - If work_experience present: `work_{i}` (by original list index)

Usage:
  python scripts/import_candidates_and_verify_vectors.py --cand-dir testOCR/output
  python scripts/import_candidates_and_verify_vectors.py --cand-dir testOCR/output --wait
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
    is missing, app.database.models will crash on import.

    For local verification, we can fall back to JSON-backed vectors (`db`).
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
from app.schemas.candidate import CandidateCreate, CandidateUpdate  # noqa: E402
from app.services.candidate_service import candidate_service  # noqa: E402
from app.database.repository.candidate_repository import candidate_repository  # noqa: E402


def _iter_candidate_json_files(cand_dir: Path) -> Iterable[Path]:
    for p in sorted(cand_dir.glob("*_candidate_profile.json")):
        if p.is_file():
            yield p


def _project_text(item: Dict[str, Any]) -> str:
    name = str(item.get("name") or "").strip()
    role = str(item.get("role") or "").strip()
    desc = str(item.get("description") or "").strip()
    parts: List[str] = []
    if name:
        parts.append(f"项目: {name}")
    if role:
        parts.append(f"角色: {role}")
    if desc:
        parts.append(desc)
    return "\n".join(parts)


def _work_text(item: Dict[str, Any]) -> str:
    company = str(item.get("company") or "").strip()
    title = str(item.get("title") or item.get("position") or "").strip()
    desc = str(item.get("description") or "").strip()
    parts: List[str] = []
    if company:
        parts.append(f"公司: {company}")
    if title:
        parts.append(f"职位: {title}")
    if desc:
        parts.append(desc)
    return "\n".join(parts)


def _normalize_candidate_payload(data: Dict[str, Any]) -> CandidateCreate:
    if "name" not in data:
        raise ValueError("candidate json missing `name`")

    return CandidateCreate(
        name=str(data["name"]),
        email=data.get("email"),
        phone=data.get("phone"),
        education=data.get("education"),
        work_experience=data.get("work_experience"),
        skills=data.get("skills"),
        projects=data.get("projects"),
        years_of_experience=data.get("years_of_experience"),
        summary=data.get("summary"),
        direction_tags=data.get("direction_tags"),
    )


def _expected_profile_types(data: Dict[str, Any]) -> List[str]:
    expected: List[str] = []
    expected.append("cand_role")

    skills = data.get("skills") or []
    if isinstance(skills, list) and any(str(x).strip() for x in skills):
        expected.append("skill")

    projects = data.get("projects") or []
    if isinstance(projects, list):
        for i, p in enumerate(projects):
            if not isinstance(p, dict):
                continue
            t = _project_text(p)
            if t.strip():
                expected.append(f"proj_{i}")

    work = data.get("work_experience") or []
    if isinstance(work, list):
        for i, w in enumerate(work):
            if not isinstance(w, dict):
                continue
            t = _work_text(w)
            if t.strip():
                expected.append(f"work_{i}")

    return expected


async def _get_latest_index_task_id(candidate_id: uuid.UUID) -> Optional[uuid.UUID]:
    async with get_session_context() as session:
        stmt = (
            select(TaskModel)
            .where(
                TaskModel.task_type == "candidate_profile_index_build",
                TaskModel.resource_id == candidate_id,
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


async def _fetch_candidate_vector_profile_types(candidate_id: uuid.UUID) -> List[str]:
    async with get_session_context() as session:
        stmt = (
            select(VectorProfileModel.profile_type)
            .where(
                VectorProfileModel.entity_type == "candidate",
                VectorProfileModel.entity_id == candidate_id,
                VectorProfileModel.status == "available",
            )
            .order_by(VectorProfileModel.profile_type.asc())
        )
        res = await session.execute(stmt)
        rows = list(res.scalars().all())
        return [str(x) for x in rows]


async def import_and_verify(cand_dir: Path, *, wait: bool) -> None:
    await init_db()

    for jf in _iter_candidate_json_files(cand_dir):
        data = json.loads(jf.read_text(encoding="utf-8"))
        expected = _expected_profile_types(data)
        payload = _normalize_candidate_payload(data)

        print(f"==> Import candidate: {jf.name} name={payload.name!r}")

        # Upsert by email if available; otherwise just create.
        candidate = None
        if payload.email:
            candidate = await candidate_repository.get_by_email(payload.email)

        if candidate:
            update = CandidateUpdate(
                name=payload.name,
                email=payload.email,
                phone=payload.phone,
                education=payload.education,
                work_experience=payload.work_experience,
                skills=payload.skills,
                projects=payload.projects,
                years_of_experience=payload.years_of_experience,
                summary=payload.summary,
                direction_tags=payload.direction_tags,
            )
            candidate = await candidate_service.update_candidate(candidate.id, update)
        else:
            candidate = await candidate_service.create_candidate(payload)

        if not candidate:
            print("    FAILED: candidate_service returned None")
            continue

        task_id = await _get_latest_index_task_id(candidate.id)
        if not task_id:
            print(f"    WARN: no candidate_profile_index_build task found for candidate_id={candidate.id}")
            continue

        if not wait:
            print(f"    candidate_id={candidate.id} task_id={task_id}")
            print("    SKIP vector verification (run with --wait after starting worker).")
            continue

        ok = await _wait_task_completed(task_id, timeout_s=180.0)
        if not ok:
            print(f"    candidate_id={candidate.id} task_id={task_id} ERROR: task not completed")
            continue

        profile_types = await _fetch_candidate_vector_profile_types(candidate.id)
        missing = [pt for pt in expected if pt not in profile_types]
        passed = len(missing) == 0

        print(f"    candidate_id={candidate.id}")
        print(f"    vector profile_types(available)={profile_types}")
        if passed:
            print("    PASS: expected embedding chunks exist")
        else:
            print(f"    FAIL: missing {missing}")
            print("    Hint: ensure `python -m app.workers.task_worker` is running and embedding works.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cand-dir",
        type=str,
        default="testOCR/output",
        help="Directory containing *_candidate_profile.json files",
    )
    parser.add_argument("--wait", action="store_true", help="Wait for indexing tasks to complete")
    args = parser.parse_args()

    cand_dir = Path(args.cand_dir).resolve()
    if not cand_dir.exists():
        raise SystemExit(f"cand dir not found: {cand_dir}")

    asyncio.run(import_and_verify(cand_dir, wait=args.wait))


if __name__ == "__main__":
    main()

