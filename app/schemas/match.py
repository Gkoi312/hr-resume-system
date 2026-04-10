# app/schemas/match.py
"""Pydantic schemas for CandidateJobMatch."""

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app import statuses


class SemanticSnippet(BaseModel):
    """Single semantic evidence snippet used in explanations."""

    source_type: Optional[str] = None
    text: str
    score: Optional[float] = None
    rationale: Optional[str] = None


class DeliveryAlignmentItem(BaseModel):
    """职责块 ↔ 经历/项目块对齐明细，供 HR 可视化（表格展开、侧栏等）。"""

    job_profile_type: str
    cand_profile_type: str
    job_text_snippet: str = ""
    cand_text_snippet: str = ""
    cosine: float = 0.0
    bm25: float = 0.0
    rank_cos: int = 0
    rank_bm25: int = 0
    rrf: float = 0.0
    bm25_degenerate: bool = False
    shared_terms: List[str] = []


class ScoreBreakdown(BaseModel):
    """Top-level score composition for one match."""

    skill_score: Optional[float] = None
    semantic_score: Optional[float] = None
    llm_quality_score: Optional[float] = None
    skill_weight: Optional[float] = None
    semantic_weight: Optional[float] = None
    llm_quality_weight: Optional[float] = None
    overall_score: Optional[float] = None


class LLMQualityBreakdown(BaseModel):
    """Structured LLM quality dimensions and evidence."""

    impact_score: Optional[float] = None
    evidence_quality_score: Optional[float] = None
    consistency_risk: Optional[float] = None
    llm_quality_score: Optional[float] = None
    summary: Optional[str] = None
    status: Optional[str] = None


class MatchExplanation(BaseModel):
    """
    Structured, HR-facing explanation for a single candidate-job match.

    - hard_requirements_met: 已满足的硬性门槛（必备技能 / 年限 / 学历等）
    - missing_requirements: 未满足的硬性门槛
    - strong_signals: 加分项 / 亮点
    - risk_signals: 风险点 / 需要 HR 注意的地方
    - summary_for_hr: 预留给 LLM 生成的自然语言总结（规则版可留空）
    - interview_focus_points: 建议在面试中重点追问或验证的点
    - suggested_action: 机器可读的建议动作，用于前端筛选和排序
    """

    hard_requirements_met: List[str] = []
    missing_requirements: List[str] = []
    strong_signals: List[str] = []
    risk_signals: List[str] = []
    summary_for_hr: Optional[str] = None
    interview_focus_points: List[str] = []
    suggested_action: Literal[
        "recommend_interview",
        "further_screening",
        "not_recommended",
    ]
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    job_skill_terms: List[str] = []
    candidate_skill_terms: List[str] = []
    semantic_status: Optional[str] = None
    semantic_evidence: Optional[List[SemanticSnippet]] = None
    delivery_alignments: Optional[List[DeliveryAlignmentItem]] = None
    llm_quality: Optional[LLMQualityBreakdown] = None
    score_breakdown: Optional[ScoreBreakdown] = None


class MatchBase(BaseModel):
    """Base match schema."""

    overall_score: Optional[float] = None
    skill_score: Optional[float] = None
    experience_score: Optional[float] = None
    llm_quality_score: Optional[float] = None
    education_score: Optional[float] = None
    semantic_score: Optional[float] = None
    industry_score: Optional[float] = None
    pros: Optional[List[str]] = None
    cons: Optional[List[str]] = None
    recommendation: Optional[str] = None
    status: str = statuses.MATCH_STATUS_COMPLETED
    explanation: Optional[MatchExplanation] = None


class MatchCreate(MatchBase):
    """Schema for creating a match."""

    job_id: UUID
    candidate_id: UUID


class MatchUpdate(BaseModel):
    """Schema for partial match update."""

    overall_score: Optional[float] = None
    skill_score: Optional[float] = None
    experience_score: Optional[float] = None
    llm_quality_score: Optional[float] = None
    education_score: Optional[float] = None
    semantic_score: Optional[float] = None
    industry_score: Optional[float] = None
    pros: Optional[List[str]] = None
    cons: Optional[List[str]] = None
    recommendation: Optional[str] = None
    status: Optional[str] = None
    explanation: Optional[MatchExplanation] = None


class MatchInDB(MatchBase):
    """Schema for match as stored."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    job_id: UUID
    candidate_id: UUID
    created_at: datetime
    error_message: Optional[str] = None


class MatchResponse(MatchInDB):
    """Schema for match in API response."""

    pass


class MatchWithCandidate(MatchResponse):
    """Match plus nested candidate summary (for list_by_job)."""

    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None


class EducationFilterCandidateItem(BaseModel):
    """One candidate who passed the resume-backed education gate for a job."""

    candidate_id: UUID
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    meets_requirement: bool = True
    education_gate_source: str
    required_education: Optional[str] = None
    resume_degree_levels: List[str] = []
    education_required_min_rank: Optional[int] = None
    resume_best_degree_rank: Optional[int] = None


class EducationFilterResponse(BaseModel):
    """Result of education-only filtering (no semantic matching)."""

    job_id: UUID
    total_input: int
    passed_count: int
    candidates: List[EducationFilterCandidateItem]
