"""
对指定候选人执行与 Worker 相同的向量入库逻辑，并打印 vector_profiles 结果。

用法（在项目根目录）:
  python scripts/test_candidate_vector_upsert.py --list
  python scripts/test_candidate_vector_upsert.py --candidate-id <uuid>

依赖: 与主应用相同（DATABASE_URL、.env；pgvector 时需 PostgreSQL 已安装 vector 扩展）。

Embedding:
  - EMBEDDING_PROVIDER=ollama: POST {OLLAMA_EMBED_BASE_URL}/api/embeddings ，默认 http://127.0.0.1:11434
  - EMBEDDING_PROVIDER=debug: 不下载模型；EMBEDDING_DIM 须与库表列一致。
  - EMBEDDING_PROVIDER=bge: Hugging Face 首次自动拉模型（需联网）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from sqlalchemy import select  # noqa: E402

from app.database.models import CandidateModel, VectorProfileModel  # noqa: E402
from app.database.session import close_db, get_session_context, init_db  # noqa: E402
from app.llm.embedding import embedding_client  # noqa: E402
from app.rag.vector_store import vector_store  # noqa: E402
from app.tasks.rag_index_task import _index_candidate  # noqa: E402


async def list_candidates(limit: int) -> None:
    async with get_session_context() as session:
        stmt = (
            select(CandidateModel.id, CandidateModel.name)
            .order_by(CandidateModel.id)
            .limit(limit)
        )
        res = await session.execute(stmt)
        rows = list(res.all())
    if not rows:
        print("candidates 表为空。")
        return
    print(f"最近 {len(rows)} 条候选人 id / name:")
    for cid, name in rows:
        async with get_session_context() as session:
            vstmt = select(VectorProfileModel.status).where(
                VectorProfileModel.entity_type == "candidate",
                VectorProfileModel.entity_id == cid,
                VectorProfileModel.profile_type == "general",
            )
            vres = await session.execute(vstmt)
            st = vres.scalar_one_or_none()
        extra = f"  vector: {st}" if st else "  vector: (无记录)"
        print(f"  {cid}  {name!r}  {extra}")


async def print_vector_row(candidate_id: uuid.UUID) -> None:
    async with get_session_context() as session:
        stmt = select(VectorProfileModel).where(
            VectorProfileModel.entity_type == "candidate",
            VectorProfileModel.entity_id == candidate_id,
            VectorProfileModel.profile_type == "general",
        )
        res = await session.execute(stmt)
        row = res.scalar_one_or_none()
    if not row:
        print("vector_profiles: 尚无 general 记录（入库未执行或失败）。")
        return
    vec = vector_store._coerce_vector(row.vector)
    print("--- vector_profiles 行 ---")
    print(f"  status:            {row.status}")
    print(f"  vector dim:        {len(vec)}")
    print(f"  content_hash:      {row.content_hash}")
    print(f"  embedding_model:   {row.embedding_model}")
    if row.error_message:
        print(f"  error_message:     {row.error_message[:500]}")
    meta = row.meta or {}
    text = meta.get("semantic_profile_text")
    if isinstance(text, str) and text.strip():
        preview = text.strip().replace("\n", " ")[:240]
        print(f"  semantic_text:     {preview!r}...")


async def main() -> int:
    p = argparse.ArgumentParser(description="测试候选人向量入库（_index_candidate）")
    p.add_argument("--list", action="store_true", help="列出部分候选人及是否已有 vector 记录")
    p.add_argument("--candidate-id", type=str, default=None, help="候选人 UUID")
    p.add_argument("--limit", type=int, default=20, help="--list 时条数上限")
    args = p.parse_args()

    prov = (os.getenv("EMBEDDING_PROVIDER") or "debug").lower()
    dim_cfg = os.getenv("EMBEDDING_DIM", "")
    vbackend = (os.getenv("VECTOR_STORE_BACKEND") or "db").lower()
    print(
        f"(env) VECTOR_STORE_BACKEND={vbackend!r}  "
        f"EMBEDDING_PROVIDER={prov!r}  "
        f"EMBEDDING_DIM={dim_cfg or '(默认见代码)'!r}  "
        f"client.model_name={getattr(embedding_client, 'model_name', None)!r}\n"
    )
    if vbackend == "pgvector" and prov == "debug" and not dim_cfg.strip():
        print(
            "提示: pgvector + debug 时请在 .env 设置 EMBEDDING_DIM，与 vector 列维度一致（常为 384）。\n",
            file=sys.stderr,
        )

    await init_db()

    try:
        if args.list:
            await list_candidates(args.limit)
            return 0

        if not args.candidate_id:
            print("请指定 --candidate-id <uuid> 或先用 --list 查看。", file=sys.stderr)
            return 2

        candidate_id = uuid.UUID(args.candidate_id.strip())

        async with get_session_context() as session:
            stmt = select(CandidateModel.id).where(CandidateModel.id == candidate_id)
            res = await session.execute(stmt)
            if res.scalar_one_or_none() is None:
                print(f"候选人不存在: {candidate_id}", file=sys.stderr)
                return 1

        print(f"执行 _index_candidate({candidate_id}) ...")
        summary = await _index_candidate(candidate_id)
        print("结果:", summary)
        await print_vector_row(candidate_id)

        if summary.get("skipped_profile"):
            print("\n(本次因 content_hash + model 未变而跳过重新 embedding，库中仍为原向量。)")

        st_row = None
        async with get_session_context() as session:
            stmt = select(VectorProfileModel.status).where(
                VectorProfileModel.entity_type == "candidate",
                VectorProfileModel.entity_id == candidate_id,
                VectorProfileModel.profile_type == "general",
            )
            res = await session.execute(stmt)
            st_row = res.scalar_one_or_none()
        if st_row == "available" and not summary.get("skipped_profile"):
            print("\n结论: 向量已写入且 status=available，入库流程正常。")
        elif st_row == "available" and summary.get("skipped_profile"):
            print("\n结论: 已有可用向量，本次未重写（跳过）。")
        elif st_row == "embedding_failed":
            print("\n结论: 嵌入失败，请查看 error_message 与 EMBEDDING_PROVIDER / 网络 / 维度配置。", file=sys.stderr)
            return 1
        return 0
    finally:
        await close_db()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
