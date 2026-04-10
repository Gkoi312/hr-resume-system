"""LLM extraction for canonical JobStructured JSON."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from app.llm.chat_client import ChatLLMClient, LLMClientError
from app.schemas.job import JobStructured

JOB_JSON_SYSTEM = """You are an expert HR JD parser.
Extract job requirements into strict JSON schema:
{
  "job_title": string|null,
  "required_skills": string[]|null,
  "preferred_skills": string[]|null,
  "responsibilities": string[]|null,
  "min_years": integer|null,
  "education_requirement": string|null,
  "industry_preference": string[]|null,
  "keywords": string[]|null,
  "job_summary": string|null
}

Rules:
- Return JSON object only.
- Use Chinese when source is Chinese.
- required_skills and preferred_skills must be short skill tokens.
- responsibilities should be short duty statements from JD.
- min_years must be integer number only.
- If unknown, use null.
"""

job_llm_client = ChatLLMClient(env_prefix="JOB_LLM_")


def job_llm_enabled() -> bool:
    v = (os.getenv("JOB_LLM_ENABLED") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _clean_list(values: Any) -> List[str] | None:
    if not isinstance(values, list):
        return None
    out: List[str] = []
    seen: set[str] = set()
    for v in values:
        s = str(v or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out or None


def coerce_job_structured(obj: Dict[str, Any]) -> JobStructured:
    raw = dict(obj or {})
    min_years = raw.get("min_years")
    if min_years in ("", None):
        min_years = None
    else:
        try:
            min_years = int(min_years)
        except (TypeError, ValueError):
            min_years = None
    clean = {
        "job_title": (str(raw.get("job_title")).strip() if raw.get("job_title") else None),
        "required_skills": _clean_list(raw.get("required_skills")),
        "preferred_skills": _clean_list(raw.get("preferred_skills")),
        "responsibilities": _clean_list(raw.get("responsibilities")),
        "min_years": min_years,
        "education_requirement": (
            str(raw.get("education_requirement")).strip()
            if raw.get("education_requirement")
            else None
        ),
        "industry_preference": _clean_list(raw.get("industry_preference")),
        "keywords": _clean_list(raw.get("keywords")),
        "job_summary": (str(raw.get("job_summary")).strip() if raw.get("job_summary") else None),
    }
    return JobStructured.model_validate(clean)


async def extract_job_structured_from_text(raw_jd_text: str) -> JobStructured:
    if not job_llm_enabled():
        raise LLMClientError("JOB_LLM_ENABLED is off")
    user_prompt = (
        "Extract canonical JobStructured JSON from JD text below.\n\n"
        f"JD TEXT:\n{raw_jd_text or ''}\n"
    )
    obj = await job_llm_client.generate_json(
        system_prompt=JOB_JSON_SYSTEM,
        user_prompt=user_prompt,
    )
    return coerce_job_structured(obj)

