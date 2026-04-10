# app/parsers/resume_llm/map_simple_to_layer1.py
"""Map layer1_simple_v1 JSON -> layer_1_extracted (same field names as simple)."""

from __future__ import annotations

from typing import Any, Dict, List

from app.parsers.resume_parser.skill_evidence import normalize_skill_list
from app.parsers.resume_parser.layer1_normalizer import normalize_layer1_blocks
from app.schemas.resume_frame import Layer1Extracted


def map_simple_to_layer1(simple: Dict[str, Any]) -> Dict[str, Any]:
    """Return layer_1_extracted dict (validated by Layer1Extracted); skills merged + lexicon-normalized."""
    raw_skills: List[str] = []
    for x in simple.get("skills") or []:
        t = str(x).strip()
        if t:
            raw_skills.append(t)
    add_in = simple.get("additional")
    if isinstance(add_in, dict):
        for x in add_in.get("skills") or []:
            t = str(x).strip()
            if t:
                raw_skills.append(t)
    payload = {
        "basic": simple.get("basic") or {},
        "education": simple.get("education") or [],
        "work_experience": simple.get("work_experience") or [],
        "projects": simple.get("projects") or [],
        "skills": normalize_skill_list(raw_skills),
        "additional": simple.get("additional") or {},
    }
    layer1 = Layer1Extracted.model_validate(payload).model_dump(mode="json")
    return normalize_layer1_blocks(layer1)
