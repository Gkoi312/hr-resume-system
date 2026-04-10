from __future__ import annotations

"""
    job <-> candidate semantic matching only
"""

import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import delete, select

from app.database.models import VectorProfileModel
from app.database.session import get_session_context


@dataclass
class ChunkVectorRecord:
    profile_type: str
    vector: List[float]
    meta: Dict[str, Any]
    content_hash: str
    embedding_model: Optional[str]
    status: str
    error_message: Optional[str] = None


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class PgVectorStore:
    """
    Persistent vector store backed by Postgres vector columns.

    Chunk-based vector store for job/candidate semantic matching.

    - Each chunk is stored as a separate row in `vector_profiles` using
      `(entity_type, entity_id, profile_type)`.
    - Resume evidence chunks are intentionally not supported in MVP.
    """

    async def sync_entity_vector_chunks(
        self,
        *,
        entity_type: str,
        entity_id: uuid.UUID,
        records: List[ChunkVectorRecord],
        version: int = 1,
    ) -> None:
        desired = {r.profile_type for r in records}
        async with get_session_context() as session:
            q = select(VectorProfileModel).where(
                VectorProfileModel.entity_type == entity_type,
                VectorProfileModel.entity_id == entity_id,
            )
            if desired:
                await session.execute(
                    delete(VectorProfileModel).where(
                        VectorProfileModel.entity_type == entity_type,
                        VectorProfileModel.entity_id == entity_id,
                        VectorProfileModel.profile_type.notin_(list(desired)),
                    )
                )
            else:
                await session.execute(
                    delete(VectorProfileModel).where(
                        VectorProfileModel.entity_type == entity_type,
                        VectorProfileModel.entity_id == entity_id,
                    )
                )
            if not records:
                return
            res = await session.execute(q)
            existing_list = list(res.scalars().all())
            by_type = {r.profile_type: r for r in existing_list}
            for rec in records:
                row = by_type.get(rec.profile_type)
                if row:
                    row.vector = rec.vector
                    row.meta = rec.meta or {}
                    row.content_hash = rec.content_hash
                    row.version = version
                    row.embedding_model = rec.embedding_model
                    row.status = rec.status
                    row.error_message = rec.error_message
                else:
                    session.add(
                        VectorProfileModel(
                            entity_type=entity_type,
                            entity_id=entity_id,
                            profile_type=rec.profile_type,
                            vector=rec.vector,
                            meta=rec.meta or {},
                            content_hash=rec.content_hash,
                            version=version,
                            embedding_model=rec.embedding_model,
                            status=rec.status,
                            error_message=rec.error_message,
                        )
                    )

    async def get_entity_chunks(
        self,
        entity_type: str,
        entity_id: uuid.UUID,
        *,
        status: str = "available",
    ) -> Dict[str, Tuple[List[float], Dict[str, Any]]]:
        async with get_session_context() as session:
            stmt = select(VectorProfileModel).where(
                VectorProfileModel.entity_type == entity_type,
                VectorProfileModel.entity_id == entity_id,
                VectorProfileModel.status == status,
            )
            res = await session.execute(stmt)
            rows = list(res.scalars().all())
            return {
                r.profile_type: (self._coerce_vector(r.vector), (r.meta or {}))
                for r in rows
            }

    async def get_candidate_chunks_bulk(
        self,
        candidate_ids: List[uuid.UUID],
        *,
        status: str = "available",
    ) -> Dict[uuid.UUID, Dict[str, Tuple[List[float], Dict[str, Any]]]]:
        if not candidate_ids:
            return {}
        async with get_session_context() as session:
            stmt = select(VectorProfileModel).where(
                VectorProfileModel.entity_type == "candidate",
                VectorProfileModel.entity_id.in_(candidate_ids),
                VectorProfileModel.status == status,
            )
            res = await session.execute(stmt)
            rows = list(res.scalars().all())
        out: Dict[uuid.UUID, Dict[str, Tuple[List[float], Dict[str, Any]]]] = {}
        for r in rows:
            bucket = out.setdefault(r.entity_id, {})
            bucket[r.profile_type] = (self._coerce_vector(r.vector), (r.meta or {}))
        return out

    @staticmethod
    def _coerce_vector(v: Any) -> List[float]:
        if v is None:
            return []
        if isinstance(v, list):
            return [float(x) for x in v]
        # pgvector.sqlalchemy.Vector is iterable; treat it as sequence of floats.
        try:
            return [float(x) for x in v]
        except Exception:
            return []

def _build_vector_store() -> Any:
    # Keep a single backend to maintain readability.
    return PgVectorStore()


vector_store = _build_vector_store()

