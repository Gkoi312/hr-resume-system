"""
Demo: create job「RAG开发实习生」, upload resumes from a folder, drain vector index
tasks (no separate worker), run matching, print ranking.

Usage (from repo root, with .env / DB / Ollama embedding ready):

    conda activate hr_resume
    python scripts/run_rag_intern_match_demo.py

    python scripts/run_rag_intern_match_demo.py --resumes-dir path/to/dir
    python scripts/run_rag_intern_match_demo.py --no-drain   # only enqueue tasks; run worker yourself
    python scripts/run_rag_intern_match_demo.py --reuse-parsed   # 同文件名已 candidate_bound 则跳过解析，只处理新文件
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app import statuses  # noqa: E402
from app.database.repository.resume_repository import resume_repository  # noqa: E402
from app.database.repository.task_repository import task_repository  # noqa: E402
from app.database.session import close_db, init_db  # noqa: E402
from app.parsers.resume_parser.resume_llm_layer1.pipeline import parse_resume_document  # noqa: E402
from app.schemas.job import JobCreate, JobStructured  # noqa: E402
from app.services.job_service import job_service  # noqa: E402
from app.services.matching_service import matching_service  # noqa: E402
from app.services.resume_service import resume_service  # noqa: E402
from app.services.task_service import task_service  # noqa: E402
from app.workers.task_worker import dispatch_task  # noqa: E402

logger = logging.getLogger(__name__)

_INDEX_TYPES = ["job_profile_index_build", "candidate_profile_index_build"]
_RESUME_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".txt"}


def _rag_intern_job_payload() -> JobCreate:
    """Structured JD: RAG 开发实习生（匹配 chunk / 向量 / 检索相关表述）。"""
    raw = """岗位：RAG开发实习生

我们希望你参与企业知识库问答、检索增强生成（RAG）相关功能开发与实验，与后端、算法同学协作，把文档切块、向量索引、重排序与评测链路跑通并落地到服务里。

实习地点与到岗时间以 HR 沟通为准；本岗位侧重工程实现与实验记录，适合对 LLM 应用与搜索技术感兴趣的同学。
"""
    structured = JobStructured(
        job_title="RAG开发实习生",
        required_skills=[
            "Python",
            "RAG / 检索增强生成",
            "向量数据库或向量检索",
            "文本切块与 Embedding",
            "HTTP API / 后端基础",
        ],
        preferred_skills=[
            "LangChain / LlamaIndex 等编排框架（任一）",
            "PostgreSQL / pgvector",
            "BM25 或混合检索（向量+关键词）",
            "Docker",
            "对 LLM Prompt 与评测有基本了解",
        ],
        responsibilities=[
            "参与文档解析、分块策略设计与实现",
            "维护或对接 Embedding 与向量入库、查询接口",
            "协助搭建/优化 RAG 流水线（检索、重排、上下文拼装）",
            "编写简单评测脚本与实验记录，对比不同配置效果",
            "阅读内外部文档，整理可复用的技术笔记",
        ],
        min_years=0,
        # 不写「本科」等关键字，避免简历未解析出学历时全部被硬过滤；需要严门槛时改为「本科及以上」等即可。
        education_requirement="学历专业与年级以简历与面试沟通为准",
        keywords=[
            "RAG",
            "embedding",
            "vector",
            "retrieval",
            "chunk",
            "rerank",
            "知识库",
        ],
        job_summary="RAG / 检索增强生成方向实习生：Python 工程 + 向量检索与流水线实验。",
    )
    return JobCreate(
        title="RAG开发实习生",
        raw_jd_text=raw,
        structured=structured,
        status=statuses.JOB_STATUS_ACTIVE,
    )


def _list_resume_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")
    files: List[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in _RESUME_SUFFIXES:
            files.append(p)
    return files


async def _upload_resume_file(file_path: Path, *, reuse_parsed: bool) -> Optional[uuid.UUID]:
    fn = file_path.name
    if reuse_parsed:
        existing = await resume_repository.find_latest_reusable_by_file_name(fn)
        if existing is not None:
            logger.info("Reuse candidate %s for %s (skip parse)", existing.candidate_id, fn)
            return existing.candidate_id

    content = file_path.read_bytes()
    upload_dir = Path(tempfile.gettempdir()) / "hr_resume_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_name = f"{uuid.uuid4().hex}_{Path(fn).name}"
    tmp_path_obj = upload_dir / tmp_name
    tmp_path_obj.write_bytes(content)
    tmp_path = str(tmp_path_obj)

    resume = await resume_service.create_resume_record(
        candidate_id=None,
        file_path=tmp_path,
        file_name=fn,
    )
    await resume_service.update_resume_status(resume.id, statuses.RESUME_STATUS_EXTRACTING)
    try:
        parsed = await parse_resume_document(
            content,
            fn,
            document_id=str(resume.id),
            candidate_id=str(resume.candidate_id),
        )
        await resume_service.save_parsed_resume(
            resume.id,
            parsed,
            status=statuses.RESUME_STATUS_PARSED,
        )
        await resume_service.bind_candidate_from_resume(resume.id)
        await resume_service.update_resume_status(
            resume.id,
            statuses.RESUME_STATUS_CANDIDATE_BOUND,
        )
        logger.info("Resume OK: %s -> candidate %s", fn, resume.candidate_id)
        return resume.candidate_id
    except Exception as exc:  # noqa: BLE001
        logger.exception("Resume failed: %s", fn)
        await resume_service.mark_failed(resume.id, str(exc))
        return None


async def _drain_index_tasks() -> None:
    while True:
        task = await task_repository.acquire_next_pending(task_types=_INDEX_TYPES)
        if task is None:
            break
        logger.info("Running task %s (%s)", task.id, task.task_type)
        try:
            await dispatch_task(task)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Task %s failed", task.id)
            await task_service.mark_failed(task.id, str(exc))


async def _async_main(resumes_dir: Path, drain: bool, reuse_parsed: bool) -> None:
    await init_db()

    job = await job_service.create_job(_rag_intern_job_payload(), auto_analyze=False)
    if not job:
        raise RuntimeError("create_job returned None")
    logger.info("Created job %s — %s", job.id, job.title)

    files = _list_resume_files(resumes_dir)
    if not files:
        raise FileNotFoundError(f"No resume files (*{', *'.join(sorted(_RESUME_SUFFIXES))}) in {resumes_dir}")

    candidate_ids: List[uuid.UUID] = []
    for fp in files:
        cid = await _upload_resume_file(fp, reuse_parsed=reuse_parsed)
        if cid is not None:
            candidate_ids.append(cid)

    if not candidate_ids:
        raise RuntimeError("No resume uploaded successfully; fix parse/OCR/LLM env and retry.")

    if drain:
        await _drain_index_tasks()
    else:
        logger.warning("Skipped draining index tasks; start worker: python -m app.workers.task_worker")

    matches = await matching_service.run_matching(
        job_id=job.id,
        candidate_ids=candidate_ids,
        delete_old=True,
    )
    ranked = await matching_service.get_matches_by_job(job.id, limit=100, offset=0)

    print()
    print("========== 匹配排名（按 overall_score 降序）==========")
    print(f"Job: {job.title}  id={job.id}")
    print()
    for i, m in enumerate(ranked, start=1):
        name = m.candidate_name or "(未解析姓名)"
        email = m.candidate_email or "-"
        score = m.overall_score if m.overall_score is not None else 0.0
        rec = m.recommendation or "-"
        sem = getattr(m.explanation, "semantic_status", None) if m.explanation else None
        extra = f"  [{sem}]" if sem else ""
        print(f"{i:2}. {score:5.1f}  {name}  <{email}>  — {rec}{extra}")
    print()
    print(f"Raw match count: {len(matches)}  (list_with_candidate: {len(ranked)})")
    await close_db()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resumes-dir",
        type=Path,
        default=ROOT / "resumes",
        help="Folder containing resume files (default: ./resumes)",
    )
    parser.add_argument(
        "--no-drain",
        action="store_true",
        help="Do not run index tasks in-process; enqueue only (use task worker separately)",
    )
    parser.add_argument(
        "--reuse-parsed",
        action="store_true",
        help=(
            "If DB already has a resume with the same file name in candidate_bound status, "
            "reuse that candidate and skip parse (new files in the folder are still parsed)"
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        _async_main(
            args.resumes_dir.resolve(),
            drain=not args.no_drain,
            reuse_parsed=args.reuse_parsed,
        )
    )


if __name__ == "__main__":
    main()
