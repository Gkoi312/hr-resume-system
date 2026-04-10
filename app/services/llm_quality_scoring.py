"""LLM-based quality scoring for stored candidate profile evidence."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from app.database.models import CandidateModel
from app.llm.chat_client import ChatLLMClient, LLMClientError


QUALITY_LLM_SYSTEM = """你是一个用于 HR 初筛的候选人经历质量评估器。

你的任务不是判断候选人是否匹配岗位，也不是给出录用建议，
而是只根据候选人画像中的技能、项目、实习/工作经历，评估其经历描述质量。

你必须只返回 JSON，对象结构如下：
{
  "impact_score": number,
  "evidence_quality_score": number,
  "consistency_risk": number,
  "summary": string
}

评分规则：
- 所有分数范围都是 0 到 100。
- impact_score：考察是否体现结果、指标、业务价值、性能提升、效率提升。
- evidence_quality_score：考察描述是否具体，是否出现技术动作、模块名、职责边界、上下文。
- consistency_risk：考察是否存在明显空泛、堆词、技术名词乱贴、前后不一致。风险越高，分越高。
- 只能依据输入文本打分，禁止脑补不存在的能力、结果或职责。
- 如果证据弱、描述泛泛、缺少技术细节，impact_score 和 evidence_quality_score 应偏低。
- 如果没有量化结果或明确业务收益，不要给高 impact_score。
- summary 用中文简要概括评分依据，必须基于输入内容，不要空泛。
"""

quality_llm_client = ChatLLMClient(env_prefix="MATCH_LLM_")


def quality_llm_enabled() -> bool:
    v = (os.getenv("MATCH_LLM_ENABLED") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _clip_score(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(100.0, num)), 1)


def _candidate_experience_payload(candidate: CandidateModel) -> str:
    parts: List[str] = []
    skill_terms = [str(x).strip() for x in (candidate.skills or []) if str(x or "").strip()]
    if skill_terms:
        parts.append("技能:\n" + ", ".join(skill_terms))
    for idx, row in enumerate(candidate.work_experience or [], start=1):
        if not isinstance(row, dict):
            continue
        lines = [
            f"工作经历{idx}:",
            f"公司: {str(row.get('company') or '').strip()}",
            f"职位: {str(row.get('title') or row.get('position') or '').strip()}",
            f"描述: {str(row.get('description') or '').strip()}",
        ]
        parts.append("\n".join(lines))
    for idx, row in enumerate(candidate.projects or [], start=1):
        if not isinstance(row, dict):
            continue
        lines = [
            f"项目{idx}:",
            f"名称: {str(row.get('name') or '').strip()}",
            f"角色: {str(row.get('role') or '').strip()}",
            f"描述: {str(row.get('description') or '').strip()}",
        ]
        parts.append("\n".join(lines))
    return "\n\n".join(p for p in parts if p.strip())


async def score_candidate_quality_with_llm(
    candidate: CandidateModel,
) -> Dict[str, Any]:
    payload = _candidate_experience_payload(candidate)
    if not payload.strip():
        return {
            "impact_score": 0.0,
            "evidence_quality_score": 0.0,
            "consistency_risk": 100.0,
            "llm_quality_score": 0.0,
            "summary": "候选人缺少可供 LLM 评估的经历描述。",
            "status": "no_candidate_evidence",
        }
    if not quality_llm_enabled():
        return {
            "impact_score": 0.0,
            "evidence_quality_score": 0.0,
            "consistency_risk": 100.0,
            "llm_quality_score": 0.0,
            "summary": "LLM 质量评分未启用。",
            "status": "disabled",
        }
    user_prompt = (
        "请仅评估候选人画像里的技能、项目和实习/工作经历质量，不要判断岗位匹配，不要做录用建议。\n\n"
        f"{payload}\n"
    )
    try:
        obj = await quality_llm_client.generate_json(
            system_prompt=QUALITY_LLM_SYSTEM,
            user_prompt=user_prompt,
        )
    except LLMClientError as exc:
        return {
            "impact_score": 0.0,
            "evidence_quality_score": 0.0,
            "consistency_risk": 100.0,
            "llm_quality_score": 0.0,
            "summary": "LLM 质量评分失败，已回退为 0 分。",
            "status": "unavailable",
        }

    impact_score = _clip_score(obj.get("impact_score"))
    evidence_quality_score = _clip_score(obj.get("evidence_quality_score"))
    consistency_risk = _clip_score(obj.get("consistency_risk"))
    risk_safety_score = round(100.0 - consistency_risk, 1)
    llm_quality_score = round(
        0.4 * impact_score + 0.4 * evidence_quality_score + 0.2 * risk_safety_score,
        1,
    )
    return {
        "impact_score": impact_score,
        "evidence_quality_score": evidence_quality_score,
        "consistency_risk": consistency_risk,
        "llm_quality_score": llm_quality_score,
        "summary": str(obj.get("summary") or "").strip() or None,
        "status": "available",
    }
