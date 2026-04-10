# app/database/repository/llm_usage_repository.py
"""Repository for LLM usage statistics data access."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func, select

from app.database.models import LLMUsageLogModel
from app.database.session import get_session_context

logger = logging.getLogger(__name__)


class LLMUsageRepository:
    """Data access layer for LLM usage statistics."""

    async def create_log(
        self,
        request_id: str,
        module_name: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        model_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ) -> LLMUsageLogModel:
        async with get_session_context() as session:
            log = LLMUsageLogModel(
                id=uuid.uuid4(),
                request_id=request_id,
                module_name=module_name,
                user_id=user_id,
                conversation_id=uuid.UUID(conversation_id) if conversation_id else None,
                model_name=model_name,
                tool_name=tool_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                duration_ms=duration_ms,
            )
            session.add(log)
            await session.flush()
            await session.refresh(log)
            return log

    def _build_conditions(
        self,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List:
        conditions = []
        if user_id:
            conditions.append(LLMUsageLogModel.user_id == user_id)
        if conversation_id:
            conditions.append(
                LLMUsageLogModel.conversation_id == uuid.UUID(conversation_id)
            )
        if start_date:
            conditions.append(LLMUsageLogModel.created_at >= start_date)
        if end_date:
            conditions.append(LLMUsageLogModel.created_at <= end_date)
        return conditions

    @staticmethod
    def _log_to_dict(log: LLMUsageLogModel) -> Dict[str, Any]:
        return log.to_dict()

    async def get_summary(
        self,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        async with get_session_context() as session:
            conditions = self._build_conditions(
                user_id=user_id,
                conversation_id=conversation_id,
                start_date=start_date,
                end_date=end_date,
            )
            stmt = select(
                func.count(LLMUsageLogModel.id).label("total_calls"),
                func.sum(LLMUsageLogModel.input_tokens).label("total_input_tokens"),
                func.sum(LLMUsageLogModel.output_tokens).label("total_output_tokens"),
                func.sum(LLMUsageLogModel.total_tokens).label("total_tokens"),
                func.avg(LLMUsageLogModel.total_tokens).label("avg_tokens_per_call"),
                func.avg(LLMUsageLogModel.duration_ms).label("avg_duration_ms"),
                func.min(LLMUsageLogModel.created_at).label("first_call"),
                func.max(LLMUsageLogModel.created_at).label("last_call"),
            )
            if conditions:
                stmt = stmt.where(and_(*conditions))
            result = await session.execute(stmt)
            row = result.one()
            return {
                "total_calls": row.total_calls or 0,
                "total_input_tokens": row.total_input_tokens or 0,
                "total_output_tokens": row.total_output_tokens or 0,
                "total_tokens": row.total_tokens or 0,
                "avg_tokens_per_call": float(row.avg_tokens_per_call)
                if row.avg_tokens_per_call
                else 0,
                "avg_duration_ms": float(row.avg_duration_ms)
                if row.avg_duration_ms
                else 0,
                "period": {
                    "start": start_date.isoformat()
                    if start_date
                    else (row.first_call.isoformat() if row.first_call else None),
                    "end": end_date.isoformat()
                    if end_date
                    else (row.last_call.isoformat() if row.last_call else None),
                },
            }

    async def get_time_series(
        self,
        days: int = 7,
        granularity: str = "hour",
        user_id: Optional[str] = None,
        module_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session_context() as session:
            start_date = datetime.utcnow() - timedelta(days=days)
            conditions = [LLMUsageLogModel.created_at >= start_date]
            if user_id:
                conditions.append(LLMUsageLogModel.user_id == user_id)
            if module_name:
                conditions.append(LLMUsageLogModel.module_name == module_name)
            if model_name:
                conditions.append(LLMUsageLogModel.model_name == model_name)
            date_trunc = (
                func.date_trunc("day", LLMUsageLogModel.created_at)
                if granularity == "day"
                else func.date_trunc("hour", LLMUsageLogModel.created_at)
            )
            stmt = (
                select(
                    date_trunc.label("period"),
                    func.count(LLMUsageLogModel.id).label("call_count"),
                    func.sum(LLMUsageLogModel.input_tokens).label("input_tokens"),
                    func.sum(LLMUsageLogModel.output_tokens).label("output_tokens"),
                    func.sum(LLMUsageLogModel.total_tokens).label("total_tokens"),
                    func.avg(LLMUsageLogModel.duration_ms).label("avg_duration_ms"),
                )
                .where(and_(*conditions))
                .group_by(date_trunc)
                .order_by(date_trunc)
            )
            result = await session.execute(stmt)
            rows = result.all()
            return [
                {
                    "period": row.period.isoformat() if row.period else None,
                    "call_count": row.call_count,
                    "input_tokens": row.input_tokens or 0,
                    "output_tokens": row.output_tokens or 0,
                    "total_tokens": row.total_tokens or 0,
                    "avg_duration_ms": float(row.avg_duration_ms)
                    if row.avg_duration_ms
                    else 0,
                }
                for row in rows
            ]

    async def get_distribution_by_module(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session_context() as session:
            conditions = self._build_conditions(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
            stmt = select(
                LLMUsageLogModel.module_name,
                func.count(LLMUsageLogModel.id).label("call_count"),
                func.sum(LLMUsageLogModel.total_tokens).label("total_tokens"),
                func.avg(LLMUsageLogModel.total_tokens).label("avg_tokens"),
                func.avg(LLMUsageLogModel.duration_ms).label("avg_duration_ms"),
            )
            if conditions:
                stmt = stmt.where(and_(*conditions))
            stmt = stmt.group_by(LLMUsageLogModel.module_name).order_by(
                func.sum(LLMUsageLogModel.total_tokens).desc()
            )
            result = await session.execute(stmt)
            rows = result.all()
            return [
                {
                    "module_name": row.module_name,
                    "call_count": row.call_count,
                    "total_tokens": row.total_tokens or 0,
                    "avg_tokens": float(row.avg_tokens) if row.avg_tokens else 0,
                    "avg_duration_ms": float(row.avg_duration_ms)
                    if row.avg_duration_ms
                    else 0,
                }
                for row in rows
            ]

    async def get_distribution_by_model(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session_context() as session:
            conditions = self._build_conditions(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
            stmt = select(
                LLMUsageLogModel.model_name,
                func.count(LLMUsageLogModel.id).label("call_count"),
                func.sum(LLMUsageLogModel.total_tokens).label("total_tokens"),
                func.avg(LLMUsageLogModel.total_tokens).label("avg_tokens"),
                func.avg(LLMUsageLogModel.duration_ms).label("avg_duration_ms"),
            )
            if conditions:
                stmt = stmt.where(and_(*conditions))
            stmt = stmt.group_by(LLMUsageLogModel.model_name).order_by(
                func.sum(LLMUsageLogModel.total_tokens).desc()
            )
            result = await session.execute(stmt)
            rows = result.all()
            return [
                {
                    "model_name": row.model_name or "unknown",
                    "call_count": row.call_count,
                    "total_tokens": row.total_tokens or 0,
                    "avg_tokens": float(row.avg_tokens) if row.avg_tokens else 0,
                    "avg_duration_ms": float(row.avg_duration_ms)
                    if row.avg_duration_ms
                    else 0,
                }
                for row in rows
            ]

    async def get_by_conversation(
        self, conversation_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        async with get_session_context() as session:
            stmt = (
                select(LLMUsageLogModel)
                .where(
                    LLMUsageLogModel.conversation_id == uuid.UUID(conversation_id)
                )
                .order_by(LLMUsageLogModel.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [self._log_to_dict(log) for log in result.scalars().all()]

    async def get_distinct_modules(self) -> List[str]:
        async with get_session_context() as session:
            stmt = (
                select(LLMUsageLogModel.module_name)
                .distinct()
                .where(LLMUsageLogModel.module_name.isnot(None))
                .order_by(LLMUsageLogModel.module_name)
            )
            result = await session.execute(stmt)
            return [r[0] for r in result.all()]

    async def get_distinct_models(self) -> List[str]:
        async with get_session_context() as session:
            stmt = (
                select(LLMUsageLogModel.model_name)
                .distinct()
                .where(LLMUsageLogModel.model_name.isnot(None))
                .order_by(LLMUsageLogModel.model_name)
            )
            result = await session.execute(stmt)
            return [r[0] for r in result.all()]

    async def get_distinct_tools(self) -> List[str]:
        async with get_session_context() as session:
            stmt = (
                select(LLMUsageLogModel.tool_name)
                .distinct()
                .where(LLMUsageLogModel.tool_name.isnot(None))
                .order_by(LLMUsageLogModel.tool_name)
            )
            result = await session.execute(stmt)
            return [r[0] for r in result.all()]


llm_usage_repository = LLMUsageRepository()
