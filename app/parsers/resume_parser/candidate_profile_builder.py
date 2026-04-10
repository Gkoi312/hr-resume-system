"""Candidate bind builder (former "layer2" role).

Responsibilities:
- Generate `candidate_bind` from `parsed["layer_1_extracted"]`
- Infer `direction_tags` (multi-label) from work/projects/skills evidence

Layer1 remains responsible for all normalization (time, sorting, skill normalization,
education school tier, etc.).
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Tuple, Optional

from app.parsers.resume_parser.skill_evidence import normalize_skill_list


_DIRECTION_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "前端",
        (
            "前端",
            "react",
            "vue",
            "typescript",
            "javascript",
            "webpack",
            "vite",
            "小程序",
            "uni-app",
            "flutter",
            "android",
            "ios",
            "移动端",
        ),
    ),
    (
        "后端",
        (
            "后端",
            "spring",
            "java",
            "golang",
            "go语言",
            "flask",
            "django",
            "fastapi",
            "微服务",
            "rest",
            "rpc",
            "mysql",
            "redis",
            "kafka",
        ),
    ),
    (
        "算法",
        (
            "算法",
            "机器学习",
            "深度学习",
            "pytorch",
            "tensorflow",
            "nlp",
            "cv",
            "大模型",
            "推荐系统",
            "神经网络",
        ),
    ),
    (
        "数据",
        (
            "数据分析",
            "数据挖掘",
            "数仓",
            "hive",
            "spark",
            "flink",
            "sql",
            "bi",
            "etl",
        ),
    ),
    (
        "测试",
        (
            "测试",
            "pytest",
            "junit",
            "自动化测试",
            "接口测试",
            "性能测试",
            "selenium",
        ),
    ),
    (
        "运维",
        (
            "运维",
            "devops",
            "docker",
            "kubernetes",
            "k8s",
            "linux",
            "ci/cd",
            "jenkins",
        ),
    ),
]


def _nfkc_lower(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").lower()


def _clean_str(v: Any) -> str:
    return str(v or "").strip()


def _collect_skill_evidence_text(
    work_norm: List[Dict[str, Any]], projects_norm: List[Dict[str, Any]]
) -> str:
    """Build a light evidence blob used for direction_tags inference."""

    parts: List[str] = []
    for w in work_norm or []:
        if not isinstance(w, dict):
            continue
        for k in ("company", "job_role", "raw_block"):
            v = w.get(k)
            if v:
                parts.append(str(v))
        for d in w.get("descriptions") or []:
            if d:
                parts.append(str(d))

    for p in projects_norm or []:
        if not isinstance(p, dict):
            continue
        for k in ("project_name", "role", "raw_block"):
            v = p.get(k)
            if v:
                parts.append(str(v))
        for d in p.get("descriptions") or []:
            if d:
                parts.append(str(d))

    return "\n".join(parts)


def infer_direction_tags(
    work_norm: List[Dict[str, Any]], projects_norm: List[Dict[str, Any]], header_skills: List[str]
) -> List[str]:
    blob = _nfkc_lower(
        _collect_skill_evidence_text(work_norm, projects_norm)
        + "\n"
        + " ".join(header_skills or [])
    )
    tags: List[str] = []
    for label, keys in _DIRECTION_RULES:
        if any(_nfkc_lower(k) in blob for k in keys) and label not in tags:
            tags.append(label)
    return tags


def get_candidate_bind_for_resume(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Build candidate bind payload from `parsed["layer_1_extracted"]`."""

    layer1 = parsed.get("layer_1_extracted") or {}

    basic = layer1.get("basic") or {}

    skills_raw = layer1.get("skills") or []
    if not isinstance(skills_raw, list):
        skills_raw = []
    skills = normalize_skill_list([str(x) for x in skills_raw if str(x).strip()]) or None

    work = layer1.get("work_experience") or []
    if not isinstance(work, list):
        work = []
    projects = layer1.get("projects") or []
    if not isinstance(projects, list):
        projects = []

    direction_tags = infer_direction_tags(work, projects, skills or [])

    education_out: List[Dict[str, Any]] = []
    for e in (layer1.get("education") or []) or []:
        if not isinstance(e, dict):
            continue
        education_out.append(
            {
                "degree": _clean_str(e.get("degree")) or "未知",
                "school": _clean_str(e.get("school")),
                "major": _clean_str(e.get("major")),
                "school_tier": _clean_str(e.get("school_tier")) or "other",
            }
        )
    education_final = education_out or None

    work_out: List[Dict[str, Any]] = []
    for w in (layer1.get("work_experience") or []) or []:
        if not isinstance(w, dict):
            continue
        start = _clean_str(w.get("start"))
        end = _clean_str(w.get("end"))
        period = f"{start}-{end}".strip("-") if start or end else ""
        descs = w.get("descriptions") or []
        desc_blob = "\n".join(str(x) for x in descs if x) if isinstance(descs, list) else ""
        raw_b = _clean_str(w.get("raw_block"))
        description = (desc_blob or raw_b)[:500] or None

        title = _clean_str(w.get("job_role"))
        work_out.append(
            {
                "period": period,
                "company": _clean_str(w.get("company")),
                "title": title,
                "position": title,
                "description": description,
            }
        )
    work_final = work_out or None

    projects_out: List[Dict[str, Any]] = []
    for p in (layer1.get("projects") or []) or []:
        if not isinstance(p, dict):
            continue
        descs = p.get("descriptions") or []
        desc_blob = "\n".join(str(x) for x in descs if x) if isinstance(descs, list) else ""
        raw_b = _clean_str(p.get("raw_block"))
        description = (desc_blob or raw_b)[:1000] or None
        projects_out.append(
            {
                "name": _clean_str(p.get("project_name")),
                "role": _clean_str(p.get("role")),
                "description": description,
            }
        )
    projects_final = projects_out or None

    return {
        "name": _clean_str(basic.get("name")) or None,
        "email": _clean_str(basic.get("email")) or None,
        "phone": _clean_str(basic.get("phone")) or None,
        "skills": skills,
        "direction_tags": direction_tags,
        "education": education_final,
        "work_experience": work_final,
        "projects": projects_final,
    }

