"""Rule-based skill scoring for job-candidate matching."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from app.database.models import CandidateModel, JobModel
from app.parsers.resume_parser.skill_evidence import (
    map_phrase_to_canonical,
    normalize_skill_list,
)


def _clean_terms(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        s = str(v or "").strip()
        if not s:
            continue
        canon = map_phrase_to_canonical(s)
        final = canon or s.lower()
        if final in seen:
            continue
        seen.add(final)
        out.append(final)
    return out

def score_candidate_skills(job: JobModel, candidate: CandidateModel) -> Dict[str, Any]:
    """Return 0-100 skill score with normalized hit details."""
    structured = job.structured if isinstance(job.structured, dict) else {}
    job_skill_terms = _clean_terms(structured.get("required_skills"))
    job_skill_terms.extend(
        term for term in _clean_terms(structured.get("preferred_skills")) if term not in job_skill_terms
    )
    if not job_skill_terms:
        return {
            "skill_score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "job_skill_terms": [],
            "candidate_skill_terms": [],
            "status": "no_job_skills",
        }

    candidate_terms = normalize_skill_list(
        [str(x) for x in (candidate.skills or []) if str(x or "").strip()]
    )

    if not candidate_terms:
        return {
            "skill_score": 0.0,
            "matched_skills": [],
            "missing_skills": list(job_skill_terms),
            "job_skill_terms": job_skill_terms,
            "candidate_skill_terms": [],
            "status": "no_candidate_skills",
        }

    cand_set = set(candidate_terms)
    matched_skills = [term for term in job_skill_terms if term in cand_set]
    missing_skills = [term for term in job_skill_terms if term not in cand_set]
    skill_score = round(100.0 * len(matched_skills) / len(job_skill_terms), 1)
    return {
        "skill_score": skill_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "job_skill_terms": job_skill_terms,
        "candidate_skill_terms": candidate_terms,
        "status": "available",
    }
