# app/services/matching_service.py
"""
Matching service: run_matching(job_id), get_matches_by_job(job_id).

Pipeline:
1) Load candidate pool (explicit IDs or bounded list).
2) Hard-filter by education from ``Resume.parsed`` — survivors only.
3) Semantic scoring and ``CandidateJobMatch`` rows apply **only** to survivors;
   filtered-out candidates are never passed to the vector bulk query.

Depends: JobRepository, CandidateRepository, MatchRepository, resume-backed education gate.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from app import statuses
from app.database.models import CandidateModel
from app.database.repository.candidate_repository import candidate_repository
from app.database.repository.job_repository import job_repository
from app.database.repository.match_repository import match_repository
from app.rag.hybrid_retrieval import tokenize
from app.rag.vector_store import vector_store
from app.services.education_resume_gate import filter_candidates_by_resume_education
from app.services.semantic_chunk_matching import (
    compute_semantic_scores_for_candidates_bulk,
)
from app.schemas.match import (
    DeliveryAlignmentItem,
    EducationFilterCandidateItem,
    EducationFilterResponse,
    MatchExplanation,
    MatchResponse,
    MatchWithCandidate,
    SemanticSnippet,
)


def _pros_cons_recommendation(
    semantic_score: float,
    semantic_status: Optional[str],
) -> tuple[List[str], List[str], str]:
    pros: List[str] = []
    cons: List[str] = []
    if semantic_status == "not_indexed":
        cons.append("候选人简历未建立语义索引，无法计算语义匹配")
    else:
        if semantic_score >= 80:
            pros.append("简历与岗位语义相关度高")
        elif semantic_score >= 60:
            pros.append("简历与岗位存在一定语义相关性")
        else:
            cons.append("简历与岗位语义相关性偏低")
    if semantic_score >= 80:
        recommendation = "建议进入初筛"
    elif semantic_score >= 60:
        recommendation = "建议备选"
    elif semantic_score >= 40:
        recommendation = "可观望"
    else:
        recommendation = "暂不推荐"
    return pros, cons, recommendation


def _build_match_explanation(
    education_gate: Dict[str, Any],
    semantic_score: float,
    semantic_status: Optional[str],
) -> MatchExplanation:
    hard_met: List[str] = []
    hard_missing: List[str] = []
    strong: List[str] = []
    risks: List[str] = []
    focus_points: List[str] = []

    req_edu = education_gate.get("required_education")
    levels = education_gate.get("resume_degree_levels") or []
    meets_edu = education_gate.get("meets_requirement")
    gate_src = education_gate.get("education_gate_source")
    if req_edu and meets_edu and gate_src == "resume_parsed":
        hard_met.append(
            "学历硬性门槛已通过（简历结构化）：要求 {}；简历学历：{}".format(
                req_edu,
                " / ".join(levels) if levels else "已匹配",
            )
        )

    if semantic_status == "not_indexed":
        risks.append("语义检索不可用：需先完成候选人侧向量索引")
        focus_points.append("确认候选人简历已解析并已触发 profile 向量构建")

    if semantic_score >= 75:
        strong.append("语义向量匹配表现较好，可结合片段证据复核")

    if semantic_score >= 80:
        suggested = "recommend_interview"
    elif semantic_score >= 60:
        suggested = "further_screening"
    else:
        suggested = "not_recommended"

    summary_parts: List[str] = []
    summary_parts.append(f"语义匹配分：{semantic_score}。")
    if hard_met:
        summary_parts.append("已满足的核心条件包括：" + "；".join(hard_met) + "。")
    if hard_missing:
        summary_parts.append("尚未满足的硬性要求：" + "；".join(hard_missing) + "。")
    if strong:
        summary_parts.append("亮点：" + "；".join(strong) + "。")
    if risks:
        summary_parts.append("风险点：" + "；".join(risks) + "。")
    summary_text = "".join(summary_parts) if summary_parts else None

    return MatchExplanation(
        hard_requirements_met=hard_met,
        missing_requirements=hard_missing,
        strong_signals=strong,
        risk_signals=risks,
        summary_for_hr=summary_text,
        interview_focus_points=focus_points,
        suggested_action=suggested,
    )


def _shared_terms_for_alignment(job_snip: str, cand_snip: str, max_n: int = 12) -> List[str]:
    a = set(tokenize(job_snip))
    b = set(tokenize(cand_snip))
    return sorted(a & b)[:max_n]


def _enrich_delivery_alignments(raw: List[Any]) -> List[DeliveryAlignmentItem]:
    out: List[DeliveryAlignmentItem] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        jpt = str(row.get("job_profile_type") or "")
        cpt = str(row.get("cand_profile_type") or "")
        jt = str(row.get("job_text_snippet") or "")
        ct = str(row.get("cand_text_snippet") or "")
        shared = _shared_terms_for_alignment(jt, ct)
        out.append(
            DeliveryAlignmentItem(
                job_profile_type=jpt,
                cand_profile_type=cpt,
                job_text_snippet=jt,
                cand_text_snippet=ct,
                cosine=float(row.get("cosine") or 0.0),
                bm25=float(row.get("bm25") or 0.0),
                rank_cos=int(row.get("rank_cos") or 0),
                rank_bm25=int(row.get("rank_bm25") or 0),
                rrf=float(row.get("rrf") or 0.0),
                bm25_degenerate=bool(row.get("bm25_degenerate")),
                shared_terms=shared,
            )
        )
    return out


class MatchingService:
    """Service for job-candidate matching and match retrieval."""

    async def run_matching(
        self,
        job_id: uuid.UUID,
        candidate_ids: Optional[List[uuid.UUID]] = None,
        delete_old: bool = True,
    ) -> List[MatchResponse]:
        job = await job_repository.get_by_id(job_id)
        if not job:
            return []
        if candidate_ids is not None:
            candidate_pool: List[CandidateModel] = []
            for cid in candidate_ids:
                c = await candidate_repository.get_by_id(cid)
                if c:
                    candidate_pool.append(c)
        else:
            candidate_pool = await candidate_repository.list(limit=5000, offset=0)
        if delete_old:
            await match_repository.delete_by_job(job_id)
        job_struct = job.structured if isinstance(job.structured, dict) else None

        after_education, education_gate_by_candidate = await filter_candidates_by_resume_education(
            job, job_struct, candidate_pool
        )
        if not after_education:
            return []

        semantic_map = await compute_semantic_scores_for_candidates_bulk(
            job.id, after_education, vector_store
        )

        results: List[MatchResponse] = []
        for candidate in after_education:
            edu_gate = education_gate_by_candidate.get(candidate.id, {})
            sem_s, sem_details = semantic_map.get(
                candidate.id, (0.0, {"semantic_status": "not_indexed"})
            )
            sem_status = (
                sem_details.get("semantic_status")
                if isinstance(sem_details, dict)
                else None
            )
            overall = round(float(sem_s), 1)
            pros, cons, rec = _pros_cons_recommendation(overall, sem_status)
            explanation = _build_match_explanation(edu_gate, overall, sem_status)
            if isinstance(sem_details, dict):
                explanation.semantic_status = sem_details.get("semantic_status")
            evidence_snippets = (
                sem_details.get("evidence_snippets")
                if isinstance(sem_details, dict)
                else None
            )
            if evidence_snippets:
                explanation.semantic_evidence = [
                    SemanticSnippet(
                        source_type=e.get("source_type"),
                        text=e.get("text", ""),
                        score=float(e.get("similarity", 0.0)),
                        rationale=e.get("rationale"),
                    )
                    for e in evidence_snippets
                ]
            dd = (
                sem_details.get("delivery_detail")
                if isinstance(sem_details, dict)
                else None
            )
            raw_aln = dd.get("delivery_alignments") if isinstance(dd, dict) else None
            if isinstance(raw_aln, list) and raw_aln:
                explanation.delivery_alignments = _enrich_delivery_alignments(raw_aln)
            match = await match_repository.create(
                job_id=job_id,
                candidate_id=candidate.id,
                overall_score=overall,
                skill_score=None,
                experience_score=None,
                education_score=None,
                semantic_score=overall,
                industry_score=None,
                pros=pros if pros else None,
                cons=cons if cons else None,
                recommendation=rec,
                explanation=explanation.model_dump(),
                status=statuses.MATCH_STATUS_COMPLETED,
            )
            results.append(MatchResponse.model_validate(match))
        results.sort(key=lambda m: (m.overall_score or 0), reverse=True)
        return results

    async def list_candidates_passing_education_gate(
        self,
        job_id: uuid.UUID,
        candidate_ids: Optional[List[uuid.UUID]] = None,
    ) -> EducationFilterResponse:
        """
        Only the resume-backed education hard filter (no vectors, no match rows).
        Same candidate pool rules as run_matching.
        """
        job = await job_repository.get_by_id(job_id)
        if not job:
            return EducationFilterResponse(
                job_id=job_id,
                total_input=0,
                passed_count=0,
                candidates=[],
            )
        if candidate_ids is not None:
            pool: List[CandidateModel] = []
            for cid in candidate_ids:
                c = await candidate_repository.get_by_id(cid)
                if c:
                    pool.append(c)
        else:
            pool = await candidate_repository.list(limit=5000, offset=0)
        total_input = len(pool)
        job_struct = job.structured if isinstance(job.structured, dict) else None
        kept, meta = await filter_candidates_by_resume_education(job, job_struct, pool)
        items: List[EducationFilterCandidateItem] = []
        for c in kept:
            m = meta.get(c.id, {})
            items.append(
                EducationFilterCandidateItem(
                    candidate_id=c.id,
                    candidate_name=c.name or None,
                    candidate_email=c.email,
                    meets_requirement=bool(m.get("meets_requirement", True)),
                    education_gate_source=str(
                        m.get("education_gate_source", "resume_parsed")
                    ),
                    required_education=m.get("required_education"),
                    resume_degree_levels=list(m.get("resume_degree_levels") or []),
                    education_required_min_rank=m.get("education_required_min_rank"),
                    resume_best_degree_rank=m.get("resume_best_degree_rank"),
                )
            )
        return EducationFilterResponse(
            job_id=job_id,
            total_input=total_input,
            passed_count=len(items),
            candidates=items,
        )

    async def get_matches_by_job(
        self,
        job_id: uuid.UUID,
        limit: int = 200,
        offset: int = 0,
    ) -> List[MatchWithCandidate]:
        matches = await match_repository.list_by_job(
            job_id=job_id,
            limit=limit,
            offset=offset,
            order_by_score=True,
        )
        out: List[MatchWithCandidate] = []
        for m in matches:
            item = MatchWithCandidate(
                id=m.id,
                job_id=m.job_id,
                candidate_id=m.candidate_id,
                overall_score=m.overall_score,
                skill_score=m.skill_score,
                experience_score=m.experience_score,
                education_score=m.education_score,
                semantic_score=m.semantic_score,
                industry_score=m.industry_score,
                pros=m.pros,
                cons=m.cons,
                recommendation=m.recommendation,
                status=m.status,
                created_at=m.created_at,
                candidate_name=getattr(m.candidate, "name", None) if m.candidate else None,
                candidate_email=getattr(m.candidate, "email", None) if m.candidate else None,
                explanation=MatchExplanation.model_validate(m.explanation)
                if getattr(m, "explanation", None)
                else None,
            )
            out.append(item)
        return out


matching_service = MatchingService()
