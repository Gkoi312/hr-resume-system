"""Task handlers for RAG index build tasks (V1.2).

Task types:
- job_profile_index_build
- candidate_profile_index_build
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.database.models import VectorProfileModel
from app.database.repository.candidate_repository import candidate_repository
from app.database.repository.job_repository import job_repository
from app.database.session import get_session_context
from app.llm.embedding import embedding_client
from app.rag.chunk_profiles import build_candidate_chunks, build_job_chunks
from app.rag.vector_store import ChunkVectorRecord, vector_store
from app.services.task_service import task_service


def _failed_upsert_vector() -> List[float]:
    """pgvector 列固定维度且非空；失败记录不能用 []，用零向量占位（status≠available 不参与匹配）。"""
    dim = int(getattr(embedding_client, "dim", 0) or 0)
    if dim <= 0:
        dim = int(os.getenv("EMBEDDING_DIM", "384"))
    return [0.0] * dim


async def _load_entity_existing_chunks(
    *,
    entity_type: str,
    entity_id: uuid.UUID,
) -> Dict[str, VectorProfileModel]:
    async with get_session_context() as session:
        stmt = select(VectorProfileModel).where(
            VectorProfileModel.entity_type == entity_type,
            VectorProfileModel.entity_id == entity_id,
        )
        res = await session.execute(stmt)
        rows = list(res.scalars().all())
        return {r.profile_type: r for r in rows}


async def _embed_chunks_and_sync(
    *,
    entity_type: str,
    entity_id: uuid.UUID,
    chunks: List[Any],
) -> Dict[str, Any]:
    """
    Embed multiple chunk texts and sync into `vector_profiles`.
    chunks: list[ChunkSpec] from app.rag.chunk_profiles
    """
    embedding_model = getattr(embedding_client, "model_name", None)
    existing = await _load_entity_existing_chunks(entity_type=entity_type, entity_id=entity_id)

    # Per-chunk skip support using content_hash + embedding_model.
    need_embed: List[tuple[int, Any]] = []
    records: List[ChunkVectorRecord] = []

    for idx, c in enumerate(chunks):
        old = existing.get(c.profile_type)
        if (
            old
            and old.status == "available"
            and old.content_hash == c.content_hash
            and (not embedding_model or old.embedding_model == embedding_model)
        ):
            vec = vector_store._coerce_vector(old.vector)
            records.append(
                ChunkVectorRecord(
                    profile_type=c.profile_type,
                    vector=vec,
                    meta=c.meta,
                    content_hash=c.content_hash,
                    embedding_model=embedding_model,
                    status="available",
                    error_message=None,
                )
            )
        else:
            need_embed.append((idx, c))

    # Batch embed for all chunks that changed.
    embedded_ok = True
    embed_vectors: List[List[float]] = []
    embed_error: Optional[str] = None

    if need_embed:
        try:
            texts = [c.text for _, c in need_embed]
            embed_vectors = await embedding_client.embed_texts(texts)
            if len(embed_vectors) != len(need_embed):
                raise RuntimeError(
                    f"embed_texts returned {len(embed_vectors)} vectors, expected {len(need_embed)}"
                )
        except Exception as exc:  # noqa: BLE001
            embedded_ok = False
            embed_error = str(exc)
            embed_vectors = [_failed_upsert_vector() for _ in need_embed]

    # Build final ordered records list (sync_entity_vector_chunks deletes old ones not in desired).
    final_records_by_type: Dict[str, ChunkVectorRecord] = {}
    for c in chunks:
        final_records_by_type[c.profile_type] = None  # type: ignore[assignment]

    # First fill skipped ones (records already built).
    for r in records:
        final_records_by_type[r.profile_type] = r

    # Fill embedded/failed ones.
    for (idx, c), vec in zip(need_embed, embed_vectors):
        status = "available" if embedded_ok else "embedding_failed"
        final_records_by_type[c.profile_type] = ChunkVectorRecord(
            profile_type=c.profile_type,
            vector=vec,
            meta=c.meta,
            content_hash=c.content_hash,
            embedding_model=embedding_model,
            status=status,
            error_message=None if embedded_ok else embed_error,
        )

    final_records: List[ChunkVectorRecord] = []
    for c in chunks:
        rec = final_records_by_type.get(c.profile_type)
        if rec is None:
            # Should never happen; fallback to failed.
            rec = ChunkVectorRecord(
                profile_type=c.profile_type,
                vector=_failed_upsert_vector(),
                meta=c.meta,
                content_hash=c.content_hash,
                embedding_model=embedding_model,
                status="embedding_failed",
                error_message="chunk_record_missing_after_embed",
            )
        final_records.append(rec)

    await vector_store.sync_entity_vector_chunks(
        entity_type=entity_type,
        entity_id=entity_id,
        records=final_records,
        version=1,
    )

    return {
        "entity_type": entity_type,
        "entity_id": str(entity_id),
        "chunk_count": len(chunks),
        "skipped_chunks": len(chunks) - len(need_embed),
        "embedded_chunks": len(need_embed),
        "failed": (not embedded_ok),
        "error": embed_error,
    }


async def handle_job_profile_index_build(task) -> None:
    payload = task.payload or {}
    job_id_str: Optional[str] = payload.get("job_id") or (str(task.resource_id) if task.resource_id else None)
    if not job_id_str:
        raise ValueError("job_profile_index_build task missing job_id")
    job_id = uuid.UUID(job_id_str)

    await task_service.mark_running(task.id)
    job = await job_repository.get_by_id(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")

    chunks = build_job_chunks(job)
    summary = await _embed_chunks_and_sync(
        entity_type="job",
        entity_id=job.id,
        chunks=chunks,
    )

    await task_service.mark_completed(
        task.id,
        result_summary={"job_id": job_id_str, **summary},
    )


async def handle_candidate_profile_index_build(task) -> None:
    payload = task.payload or {}
    candidate_id_str: Optional[str] = payload.get("candidate_id") or (
        str(task.resource_id) if task.resource_id else None
    )
    if not candidate_id_str:
        raise ValueError("candidate_profile_index_build task missing candidate_id")
    candidate_id = uuid.UUID(candidate_id_str)

    await task_service.mark_running(task.id)
    candidate = await candidate_repository.get_by_id(candidate_id)
    if not candidate:
        raise ValueError(f"candidate not found: {candidate_id}")

    chunks = build_candidate_chunks(candidate)
    summary = await _embed_chunks_and_sync(
        entity_type="candidate",
        entity_id=candidate.id,
        chunks=chunks,
    )
    await task_service.mark_completed(
        task.id,
        result_summary={"candidate_id": str(candidate_id), **summary},
    )

