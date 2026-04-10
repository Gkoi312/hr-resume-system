# app/services/education_resume_gate.py
"""
学历硬性门槛：仅使用简历 parsed 里的结构化教育列表（与 Candidate 行无关）。

解析产物约定只出现四档：博士、硕士、本科、大专。档位：博士 > 硕士 > 本科 > 大专；
岗位写的是「最低档」，候选人任一教育行达到该档或以上即通过（多行取最高档）。
"""

from __future__ import annotations

import unicodedata
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app import statuses
from app.database.models import CandidateModel, JobModel, ResumeModel
from app.database.repository.resume_repository import resume_repository

_USABLE_STATUSES = frozenset(
    {
        statuses.RESUME_STATUS_PARSED,
        statuses.RESUME_STATUS_CANDIDATE_BOUND,
    }
)

RANK_COLLEGE = 0  # 大专
RANK_BACHELOR = 1
RANK_MASTER = 2
RANK_DOCTOR = 3


def _nfkc_lower(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").lower()


def job_education_requirement_text(
    job: JobModel,
    job_structured: Optional[Dict[str, Any]],
) -> Optional[str]:
    raw = getattr(job, "education_requirement", None)
    if raw and str(raw).strip():
        return str(raw).strip()
    if job_structured:
        js = job_structured.get("education_requirement")
        if js and str(js).strip():
            return str(js).strip()
    return None


def education_entries_from_parsed(parsed: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not parsed or not isinstance(parsed, dict):
        return []
    layer1 = parsed.get("layer_1_extracted")
    if isinstance(layer1, dict):
        edu = layer1.get("education")
        if isinstance(edu, list):
            return [e for e in edu if isinstance(e, dict)]
    edu_top = parsed.get("education")
    if isinstance(edu_top, list):
        return [e for e in edu_top if isinstance(e, dict)]
    return []


def degree_levels_from_entries(entries: List[Dict[str, Any]]) -> List[str]:
    levels: List[str] = []
    for e in entries:
        degree = (e.get("degree") or "").strip().lower()
        if degree:
            levels.append(degree)
    return levels


def degree_rank_from_text(raw: Any) -> Optional[int]:
    s = _nfkc_lower(str(raw or ""))
    if not s.strip():
        return None
    if "博士" in s:
        return RANK_DOCTOR
    if "硕士" in s:
        return RANK_MASTER
    if "本科" in s:
        return RANK_BACHELOR
    if "大专" in s:
        return RANK_COLLEGE
    return None


def job_minimum_degree_rank(req_raw: str) -> Optional[int]:
    """岗位要求文案中识别最低档；未出现四档关键字则返回 None（不启用门槛）。"""
    s = _nfkc_lower(req_raw)
    if not s.strip():
        return None
    if "博士" in s:
        return RANK_DOCTOR
    if "硕士" in s:
        return RANK_MASTER
    if "本科" in s:
        return RANK_BACHELOR
    if "大专" in s:
        return RANK_COLLEGE
    return None


def _max_degree_rank_from_entries(entries: List[Dict[str, Any]]) -> Optional[int]:
    ranks = [
        degree_rank_from_text(e.get("degree"))
        for e in entries
    ]
    valid = [x for x in ranks if x is not None]
    return max(valid) if valid else None


def _entries_pass_requirement(
    req_rank: int,
    entries: List[Dict[str, Any]],
) -> Tuple[bool, List[str], Optional[int]]:
    levels = degree_levels_from_entries(entries)
    best = _max_degree_rank_from_entries(entries)
    if best is not None and best >= req_rank:
        return True, levels, best
    return False, levels, best


def _evaluate_resumes_for_requirement(
    req_rank: int,
    resumes: List[ResumeModel],
) -> Tuple[bool, List[str], Optional[int]]:
    for r in resumes:
        if r.parsed is None or r.status not in _USABLE_STATUSES:
            continue
        entries = education_entries_from_parsed(r.parsed)
        ok, levels, best = _entries_pass_requirement(req_rank, entries)
        if ok:
            return True, levels, best
    return False, [], None


async def filter_candidates_by_resume_education(
    job: JobModel,
    job_structured: Optional[Dict[str, Any]],
    candidates: List[CandidateModel],
) -> Tuple[List[CandidateModel], Dict[uuid.UUID, Dict[str, Any]]]:
    req_raw = job_education_requirement_text(job, job_structured)
    if not req_raw or not candidates:
        meta = {
            c.id: {
                "required_education": None,
                "meets_requirement": True,
                "resume_degree_levels": [],
                "education_gate_source": "skipped_no_requirement",
                "education_required_min_rank": None,
                "resume_best_degree_rank": None,
            }
            for c in candidates
        }
        return list(candidates), meta

    req_min = job_minimum_degree_rank(req_raw)
    if req_min is None:
        meta = {
            c.id: {
                "required_education": req_raw,
                "meets_requirement": True,
                "resume_degree_levels": [],
                "education_gate_source": "skipped_unrecognized_requirement",
                "education_required_min_rank": None,
                "resume_best_degree_rank": None,
            }
            for c in candidates
        }
        return list(candidates), meta

    ids = [c.id for c in candidates]
    all_resumes = await resume_repository.list_by_candidate_ids(ids)
    by_cand: Dict[uuid.UUID, List[ResumeModel]] = {}
    for r in all_resumes:
        by_cand.setdefault(r.candidate_id, []).append(r)

    kept: List[CandidateModel] = []
    meta: Dict[uuid.UUID, Dict[str, Any]] = {}

    for c in candidates:
        resumes = by_cand.get(c.id, [])
        ok, levels, best_rank = _evaluate_resumes_for_requirement(req_min, resumes)
        meta[c.id] = {
            "required_education": req_raw,
            "meets_requirement": ok,
            "resume_degree_levels": levels,
            "education_gate_source": "resume_parsed",
            "education_required_min_rank": req_min,
            "resume_best_degree_rank": best_rank,
        }
        if ok:
            kept.append(c)

    return kept, meta
