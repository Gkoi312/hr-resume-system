# app/database/repository/evaluation_repository.py
"""Repository for RAG evaluation data access."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func, or_, select

from app.database.models import RAGEvaluationModel
from app.database.session import get_session_context

logger = logging.getLogger(__name__)


class EvaluationRepository:
    """Data access layer for RAG evaluations."""

    async def create(
        self,
        query: str,
        context: str,
        response: str,
        message_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        rewritten_query: Optional[str] = None,
        user_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> RAGEvaluationModel:
        async with get_session_context() as session:
            evaluation = RAGEvaluationModel(
                id=uuid.uuid4(),
                message_id=uuid.UUID(message_id) if message_id else None,
                conversation_id=uuid.UUID(conversation_id) if conversation_id else None,
                user_id=user_id,
                query=query,
                rewritten_query=rewritten_query,
                context=context,
                response=response,
                evaluation_status="pending",
                created_at=created_at or datetime.utcnow(),
            )
            session.add(evaluation)
            await session.flush()
            await session.refresh(evaluation)
            return evaluation

    async def update_results(
        self,
        evaluation_id: str,
        results: Dict[str, float],
        duration_ms: int,
        status: str = "completed",
        error_message: Optional[str] = None,
        evaluated_at: Optional[datetime] = None,
    ) -> bool:
        async with get_session_context() as session:
            stmt = select(RAGEvaluationModel).where(
                RAGEvaluationModel.id == uuid.UUID(evaluation_id)
            )
            result = await session.execute(stmt)
            evaluation = result.scalar_one_or_none()
            if not evaluation:
                return False
            evaluation.faithfulness = results.get("faithfulness")
            evaluation.answer_relevancy = results.get("answer_relevancy")
            evaluation.evaluation_status = status
            evaluation.error_message = error_message
            evaluation.evaluation_duration_ms = duration_ms
            evaluation.evaluated_at = evaluated_at or datetime.utcnow()
            await session.flush()
            return True

    async def get_by_id(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        async with get_session_context() as session:
            stmt = select(RAGEvaluationModel).where(
                RAGEvaluationModel.id == uuid.UUID(evaluation_id)
            )
            result = await session.execute(stmt)
            evaluation = result.scalar_one_or_none()
            return evaluation.to_dict() if evaluation else None

    async def get_by_conversation(
        self, conversation_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        async with get_session_context() as session:
            stmt = (
                select(RAGEvaluationModel)
                .where(
                    RAGEvaluationModel.conversation_id == uuid.UUID(conversation_id)
                )
                .order_by(RAGEvaluationModel.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [e.to_dict() for e in result.scalars().all()]

    async def get_statistics(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        async with get_session_context() as session:
            conditions = [RAGEvaluationModel.evaluation_status == "completed"]
            if user_id:
                conditions.append(RAGEvaluationModel.user_id == user_id)
            if start_date:
                conditions.append(RAGEvaluationModel.created_at >= start_date)
            if end_date:
                conditions.append(RAGEvaluationModel.created_at <= end_date)

            stmt = select(
                func.count(RAGEvaluationModel.id).label("total"),
                func.avg(RAGEvaluationModel.faithfulness).label("avg_faithfulness"),
                func.avg(RAGEvaluationModel.answer_relevancy).label(
                    "avg_answer_relevancy"
                ),
                func.min(RAGEvaluationModel.faithfulness).label("min_faithfulness"),
                func.max(RAGEvaluationModel.faithfulness).label("max_faithfulness"),
                func.min(RAGEvaluationModel.answer_relevancy).label(
                    "min_answer_relevancy"
                ),
                func.max(RAGEvaluationModel.answer_relevancy).label(
                    "max_answer_relevancy"
                ),
                func.avg(RAGEvaluationModel.evaluation_duration_ms).label(
                    "avg_duration_ms"
                ),
            ).where(and_(*conditions))

            result = await session.execute(stmt)
            row = result.one()

            base_cond = conditions[1:] if len(conditions) > 1 else []
            pending_stmt = select(func.count(RAGEvaluationModel.id)).where(
                and_(RAGEvaluationModel.evaluation_status == "pending", *base_cond)
            )
            failed_stmt = select(func.count(RAGEvaluationModel.id)).where(
                and_(RAGEvaluationModel.evaluation_status == "failed", *base_cond)
            )
            pending_result = await session.execute(pending_stmt)
            failed_result = await session.execute(failed_stmt)

            return {
                "total_evaluations": row.total or 0,
                "pending_count": pending_result.scalar() or 0,
                "failed_count": failed_result.scalar() or 0,
                "period": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None,
                },
                "metrics": {
                    "faithfulness": {
                        "mean": float(row.avg_faithfulness)
                        if row.avg_faithfulness
                        else None,
                        "min": float(row.min_faithfulness)
                        if row.min_faithfulness
                        else None,
                        "max": float(row.max_faithfulness)
                        if row.max_faithfulness
                        else None,
                    },
                    "answer_relevancy": {
                        "mean": float(row.avg_answer_relevancy)
                        if row.avg_answer_relevancy
                        else None,
                        "min": float(row.min_answer_relevancy)
                        if row.min_answer_relevancy
                        else None,
                        "max": float(row.max_answer_relevancy)
                        if row.max_answer_relevancy
                        else None,
                    },
                },
                "avg_evaluation_duration_ms": float(row.avg_duration_ms)
                if row.avg_duration_ms
                else None,
            }

    async def get_trends(
        self,
        days: int = 7,
        granularity: str = "day",
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session_context() as session:
            start_date = datetime.utcnow() - timedelta(days=days)
            conditions = [
                RAGEvaluationModel.evaluation_status == "completed",
                RAGEvaluationModel.created_at >= start_date,
            ]
            if user_id:
                conditions.append(RAGEvaluationModel.user_id == user_id)

            date_trunc = (
                func.date_trunc("hour", RAGEvaluationModel.created_at)
                if granularity == "hour"
                else func.date_trunc("day", RAGEvaluationModel.created_at)
            )
            stmt = (
                select(
                    date_trunc.label("period"),
                    func.count(RAGEvaluationModel.id).label("count"),
                    func.avg(RAGEvaluationModel.faithfulness).label("avg_faithfulness"),
                    func.avg(RAGEvaluationModel.answer_relevancy).label(
                        "avg_answer_relevancy"
                    ),
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
                    "count": row.count,
                    "metrics": {
                        "faithfulness": float(row.avg_faithfulness)
                        if row.avg_faithfulness
                        else None,
                        "answer_relevancy": float(row.avg_answer_relevancy)
                        if row.avg_answer_relevancy
                        else None,
                    },
                }
                for row in rows
            ]

    async def get_alerts(
        self,
        thresholds: Dict[str, float],
        limit: int = 50,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session_context() as session:
            conditions = [RAGEvaluationModel.evaluation_status == "completed"]
            if user_id:
                conditions.append(RAGEvaluationModel.user_id == user_id)
            threshold_conditions = []
            if "faithfulness" in thresholds:
                threshold_conditions.append(
                    RAGEvaluationModel.faithfulness < thresholds["faithfulness"]
                )
            if "answer_relevancy" in thresholds:
                threshold_conditions.append(
                    RAGEvaluationModel.answer_relevancy
                    < thresholds["answer_relevancy"]
                )
            if threshold_conditions:
                conditions.append(or_(*threshold_conditions))
            stmt = (
                select(RAGEvaluationModel)
                .where(and_(*conditions))
                .order_by(RAGEvaluationModel.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            evaluations = result.scalars().all()
            alerts = []
            for e in evaluations:
                alert_data = e.to_dict()
                alert_data["violated_thresholds"] = []
                if (
                    e.faithfulness is not None
                    and "faithfulness" in thresholds
                    and e.faithfulness < thresholds["faithfulness"]
                ):
                    alert_data["violated_thresholds"].append("faithfulness")
                if (
                    e.answer_relevancy is not None
                    and "answer_relevancy" in thresholds
                    and e.answer_relevancy < thresholds["answer_relevancy"]
                ):
                    alert_data["violated_thresholds"].append("answer_relevancy")
                alerts.append(alert_data)
            return alerts


evaluation_repository = EvaluationRepository()
