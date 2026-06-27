"""
Run matching directly from existing DB data.

Use this when jobs/candidates/resumes/vector indexes are already in the database,
and you only want to pick an existing job plus existing candidates and execute
the matching service.

Examples:

    conda activate hr_resume
    python scripts/run_existing_db_match.py --job-id a828236c-1850-471a-945c-9f0e4dd18b42 --candidate-ids 88642181-87c9-46b1-9d98-99943634fac5,b41797cd-fefa-4f51-b45f-4b87a8c6b243,839ef030-b24a-42d8-8894-d7e66fbd392d,e5cc63b9-10d5-45d4-bdac-a8b9d60bfd96

    python scripts/run_existing_db_match.py --job-title RAG --candidate-limit 4

    python scripts/run_existing_db_match.py --job-title 后端 --candidate-keyword 张 --candidate-limit 4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.database.repository.candidate_repository import candidate_repository  # noqa: E402
from app.database.repository.job_repository import job_repository  # noqa: E402
from app.database.session import close_db, init_db  # noqa: E402
from app.services.matching_service import matching_service  # noqa: E402


def _parse_uuid_list(raw: str) -> List[uuid.UUID]:
    out: List[uuid.UUID] = []
    for part in (raw or "").split(","):
        s = part.strip()
        if not s:
            continue
        out.append(uuid.UUID(s))
    return out


async def _choose_job(
    *,
    job_id: Optional[uuid.UUID],
    job_title: Optional[str],
):
    if job_id is not None:
        job = await job_repository.get_by_id(job_id)
        if job is None:
            raise RuntimeError(f"未找到岗位: {job_id}")
        return job

    jobs = await job_repository.list(limit=200, offset=0)
    if not jobs:
        raise RuntimeError("数据库中没有可用岗位")
    if job_title:
        key = job_title.strip().lower()
        matched = [j for j in jobs if key in (j.title or "").lower()]
        if not matched:
            raise RuntimeError(f"未找到标题包含 {job_title!r} 的岗位")
        return matched[0]
    return jobs[0]


async def _load_candidates_by_ids(candidate_ids: Iterable[uuid.UUID]):
    out = []
    for cid in candidate_ids:
        candidate = await candidate_repository.get_by_id(cid)
        if candidate is not None:
            out.append(candidate)
    return out


async def _choose_candidates(
    *,
    candidate_ids: Optional[List[uuid.UUID]],
    candidate_keyword: Optional[str],
    candidate_limit: int,
):
    if candidate_ids:
        candidates = await _load_candidates_by_ids(candidate_ids)
        if not candidates:
            raise RuntimeError("candidate_ids 指定的候选人都不存在")
        return candidates

    candidates = await candidate_repository.list(
        limit=max(1, candidate_limit),
        offset=0,
        keyword=candidate_keyword,
    )
    if not candidates:
        kw = f"（keyword={candidate_keyword}）" if candidate_keyword else ""
        raise RuntimeError(f"数据库中没有可用候选人{kw}")
    return candidates


def _print_selected(job, candidates) -> None:
    print("========== 使用现有数据库数据跑匹配 ==========")
    print(f"岗位: {job.title}")
    print(f"job_id: {job.id}")
    print()
    print("候选人列表:")
    for idx, c in enumerate(candidates, start=1):
        skill_count = len(c.skills or [])
        print(
            f"{idx:2}. {c.name or '-'}  id={c.id}  "
            f"email={c.email or '-'}  skills={skill_count}"
        )
    print()


def _print_ranked_results(ranked) -> None:
    print("========== 匹配结果（按 overall_score 降序）==========")
    for idx, m in enumerate(ranked, start=1):
        name = m.candidate_name or "-"
        email = m.candidate_email or "-"
        overall = m.overall_score if m.overall_score is not None else 0.0
        skill = m.skill_score if m.skill_score is not None else 0.0
        semantic = m.semantic_score if m.semantic_score is not None else 0.0
        llm_q = m.llm_quality_score if m.llm_quality_score is not None else 0.0
        rec = m.recommendation or "-"
        print(
            f"{idx:2}. overall={overall:5.1f}  skill={skill:5.1f}  "
            f"semantic={semantic:5.1f}  llm={llm_q:5.1f}  "
            f"{name} <{email}>  — {rec}"
        )
        if m.explanation and m.explanation.summary_for_hr:
            print(f"    summary: {m.explanation.summary_for_hr}")
    print()


async def _async_main(args) -> None:
    await init_db()
    try:
        job = await _choose_job(
            job_id=args.job_id,
            job_title=args.job_title,
        )
        candidates = await _choose_candidates(
            candidate_ids=args.candidate_ids,
            candidate_keyword=args.candidate_keyword,
            candidate_limit=args.candidate_limit,
        )
        _print_selected(job, candidates)

        candidate_ids = [c.id for c in candidates]
        await matching_service.run_matching(
            job_id=job.id,
            candidate_ids=candidate_ids,
            delete_old=True,
        )
        ranked = await matching_service.get_matches_by_job(
            job.id,
            limit=max(20, len(candidate_ids) + 5),
            offset=0,
        )
        ranked = [m for m in ranked if m.candidate_id in set(candidate_ids)]
        _print_ranked_results(ranked)
    finally:
        await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-id",
        type=uuid.UUID,
        default=None,
        help="Existing job UUID. If omitted, choose by --job-title or latest updated job.",
    )
    parser.add_argument(
        "--job-title",
        type=str,
        default=None,
        help="Substring match against existing job title when --job-id is not given.",
    )
    parser.add_argument(
        "--candidate-ids",
        type=_parse_uuid_list,
        default=None,
        help="Comma-separated candidate UUIDs. If omitted, select from DB by keyword/latest.",
    )
    parser.add_argument(
        "--candidate-keyword",
        type=str,
        default=None,
        help="Candidate keyword used when --candidate-ids is not given.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=4,
        help="How many candidates to take from DB when --candidate-ids is not given.",
    )
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
