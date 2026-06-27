"""
按与 matching 相同的逻辑，打印某岗位下各候选人的语义分明细（技能 / 经历 delivery / 角色轴）。

用法（仓库根目录，需 .env 与数据库可用）：

    python scripts/print_match_semantic_detail.py
    python scripts/print_match_semantic_detail.py --job-id a828236c-1850-471a-945c-9f0e4dd18b42
    python scripts/print_match_semantic_detail.py --job-title "RAG开发实习生"
    python scripts/print_match_semantic_detail.py --verbose   # 附带 delivery 对齐表等长字段

说明：库里 Match 只存总分与精简 explanation（状态、证据片段、对齐表）；各轴明细需本脚本重算。

环境变量（与 matching 一致）：SEMANTIC_WEIGHT_SKILL / DELIVERY / ROLE；
SEMANTIC_DELIVERY_JOB_COVERAGE（0–1，职责覆盖在 delivery 子项中的权重）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.database.repository.job_repository import job_repository  # noqa: E402
from app.database.repository.match_repository import match_repository  # noqa: E402
from app.database.session import close_db, init_db  # noqa: E402
from app.rag.vector_store import vector_store  # noqa: E402
from app.services.semantic_chunk_matching import (  # noqa: E402
    compute_semantic_scores_for_candidates_bulk,
)


def _short_dd(dd: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
    if not isinstance(dd, dict):
        return {}
    out = dict(dd)
    if not verbose and "delivery_alignments" in out:
        aln = out.get("delivery_alignments")
        out["delivery_alignments"] = (
            f"<{len(aln)} rows>" if isinstance(aln, list) else aln
        )
    return out


def _print_one(
    name: str,
    email: str,
    db_semantic: Optional[float],
    score: float,
    detail: Dict[str, Any],
    verbose: bool,
) -> None:
    print(f"{'─' * 60}")
    print(f"候选人: {name or '(无姓名)'}  <{email or '-'}>")
    print(f"库中 Match.semantic_score: {db_semantic}")
    ws = detail.get("semantic_weight_skill")
    wd = detail.get("semantic_weight_delivery")
    wr = detail.get("semantic_weight_role")
    if ws is not None and wd is not None and wr is not None:
        wtxt = f"skill={ws:.3f} delivery={wd:.3f} role={wr:.3f}"
    else:
        wtxt = "skill=? delivery=? role=?"
    print(f"重算 overall: {round(score, 1)}   (权重 {wtxt})")
    print(f"  semantic_status:     {detail.get('semantic_status')}")
    print(
        f"  技能轴:  complete={detail.get('skill_axis_complete')}  "
        f"band={detail.get('skill_band')}  cosine={detail.get('skill_similarity')}"
    )
    print(
        f"  角色轴:  complete={detail.get('role_axis_complete')}  "
        f"band={detail.get('role_band')}  cosine={detail.get('role_similarity')}"
    )
    print(
        f"  经历轴:  complete={detail.get('delivery_axis_complete')}  "
        f"band={detail.get('delivery_band')}"
    )
    dd = detail.get("delivery_detail")
    if isinstance(dd, dict):
        sdd = _short_dd(dd, verbose)
        print(f"  delivery_detail:")
        print(json.dumps(sdd, ensure_ascii=False, indent=4))
    if verbose:
        ev = detail.get("evidence_snippets")
        if ev:
            print("  evidence_snippets:")
            print(json.dumps(ev, ensure_ascii=False, indent=2))


async def _async_main(
    job_id: Optional[uuid.UUID],
    job_title: str,
    verbose: bool,
) -> None:
    await init_db()
    if job_id is None:
        jobs = await job_repository.list(limit=200, offset=0)
        chosen = None
        for j in jobs:
            t = (j.title or "").strip()
            if t == job_title or job_title in t:
                chosen = j
                break
        if not chosen:
            print(f"未找到 title 为或包含 {job_title!r} 的岗位", file=sys.stderr)
            await close_db()
            sys.exit(1)
        job_id = chosen.id
        print(f"使用岗位: {chosen.title}  id={job_id}\n")
    else:
        job = await job_repository.get_by_id(job_id)
        if not job:
            print(f"岗位不存在: {job_id}", file=sys.stderr)
            await close_db()
            sys.exit(1)
        print(f"使用岗位: {job.title}  id={job_id}\n")

    matches = await match_repository.list_by_job(job_id, limit=500, offset=0)
    if not matches:
        print("该岗位下没有 Match 记录。", file=sys.stderr)
        await close_db()
        sys.exit(1)

    candidates: List[Any] = []
    meta: List[Tuple[str, str, Optional[float]]] = []
    for m in matches:
        c = m.candidate
        if c is None:
            continue
        candidates.append(c)
        meta.append(
            (
                (c.name or "").strip() or "(无姓名)",
                (c.email or "").strip() or "-",
                m.semantic_score,
            )
        )

    semantic_map = await compute_semantic_scores_for_candidates_bulk(
        job_id, candidates, vector_store
    )

    # 按库中 match 分数顺序打印（与 demo 排名一致）
    for c, (name, email, db_sem) in zip(candidates, meta):
        score, detail = semantic_map.get(c.id, (0.0, {}))
        if not isinstance(detail, dict):
            detail = {}
        _print_one(name, email, db_sem, float(score), detail, verbose)

    print(f"{'─' * 60}")
    await close_db()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-id", type=uuid.UUID, default=None, help="岗位 UUID")
    p.add_argument(
        "--job-title",
        default="RAG开发实习生",
        help="未指定 --job-id 时，按标题精确或子串匹配最近更新的岗位",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="打印完整 delivery_detail（含每条职责↔经历对齐）",
    )
    args = p.parse_args()
    asyncio.run(_async_main(args.job_id, args.job_title, args.verbose))


if __name__ == "__main__":
    main()
