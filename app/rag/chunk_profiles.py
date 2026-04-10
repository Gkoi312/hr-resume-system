"""Job / candidate semantic chunk specs for multi-vector indexing.

See docs/job_candidate_embedding_plan.md. profile_type must be <= 32 chars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import hashlib

from app.database.models import CandidateModel, JobModel


def stable_hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

MAX_RESP_CHARS_PER_CHUNK = 4000
_MAX_PROFILE_TYPE_LEN = 32


def _assert_profile_type(name: str) -> None:
    if len(name) > _MAX_PROFILE_TYPE_LEN:
        raise ValueError(f"profile_type too long ({len(name)}): {name!r}")


def _norm_str_list(vals: Any) -> List[str]:
    if not vals or not isinstance(vals, list):
        return []
    out: List[str] = []
    for x in vals:
        s = str(x or "").strip()
        if s and s not in out:
            out.append(s)
    return out

# 职责描述如果比较短，直接一个chunk；如果比较长，就按照“|”拆成条目；如果都不行，按4000字符做批量处理。
def _split_resp_into_chunks(resp_text: str) -> List[Tuple[str, str]]:
    if len(resp_text) <= MAX_RESP_CHARS_PER_CHUNK:
        _assert_profile_type("resp")
        return [("resp", resp_text)]
    items = [x.strip() for x in resp_text.split(" | ") if x.strip()]
    if not items:
        return [("resp", resp_text[:MAX_RESP_CHARS_PER_CHUNK])]
    batches: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for it in items:
        add = len(it) + (3 if cur else 0)
        if cur_len + add > MAX_RESP_CHARS_PER_CHUNK and cur:
            batches.append(" | ".join(cur))
            cur = [it]
            cur_len = len(it)
        else:
            cur_len = cur_len + add if cur else len(it)
            cur.append(it)
    if cur:
        batches.append(" | ".join(cur))
    out: List[Tuple[str, str]] = []
    for i, b in enumerate(batches):
        ptype = f"resp_{i}"
        _assert_profile_type(ptype)
        out.append((ptype, b))
    return out


@dataclass(frozen=True)
class ChunkSpec:
    profile_type: str
    text: str
    content_hash: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_profile_type(self.profile_type)


def build_job_chunks(job: JobModel) -> List[ChunkSpec]:
    s = job.structured if isinstance(job.structured, dict) else {}
    out: List[ChunkSpec] = []

    # 必备 + 加分合并为单一 ``skill`` chunk，入库一条向量，与候选人 skill 对齐匹配。
    req = _norm_str_list(s.get("required_skills"))
    pref = _norm_str_list(s.get("preferred_skills"))
    if req or pref:
        lines = []
        if req:
            lines.append("必备技能: " + ", ".join(req))
        if pref:
            lines.append("加分技能: " + ", ".join(pref))
        text = "\n".join(lines)
        meta = {"block": "skill", "semantic_text": text[:2000]}
        out.append(
            ChunkSpec("skill", text, stable_hash_text(text), meta),
        )

    role_parts: List[str] = []
    if getattr(job, "title", None) and str(job.title).strip():
        role_parts.append(str(job.title).strip())
    jt = s.get("job_title")
    if jt and str(jt).strip():
        role_parts.append("岗位: " + str(jt).strip())
    js = s.get("job_summary")
    if js and str(js).strip():
        role_parts.append("摘要: " + str(js).strip())
    kw = s.get("keywords")
    kws = _norm_str_list(kw) if isinstance(kw, list) else []
    if kws:
        role_parts.append("关键词: " + ", ".join(kws))
    role_text = "\n".join(role_parts)

    resp_items = _norm_str_list(s.get("responsibilities"))
    resp_joined = " | ".join(resp_items) if resp_items else ""

    resp_pairs = _split_resp_into_chunks(resp_joined) if resp_joined else []

    for ptype, rtext in resp_pairs:
        meta = {"block": "resp", "semantic_text": rtext[:2000]}
        out.append(ChunkSpec(ptype, rtext, stable_hash_text(rtext), meta))

    role_has_substance = bool(
        (jt and str(jt).strip())
        or (js and str(js).strip())
        or kws
    )
    use_jd_raw_fallback = (
        not resp_joined
        and not role_has_substance
        and not (getattr(job, "title", None) and str(job.title).strip())
    )
    if use_jd_raw_fallback:
        raw = (getattr(job, "raw_jd_text", None) or "").strip()
        if raw:
            meta = {"block": "jd_raw", "fallback": True, "semantic_text": raw[:2000]}
            out.append(ChunkSpec("jd_raw", raw, stable_hash_text(raw), meta))

    if role_text:
        meta = {"block": "role", "semantic_text": role_text[:2000]}
        out.append(ChunkSpec("role", role_text, stable_hash_text(role_text), meta))

    return out


def _project_text(item: Dict[str, Any]) -> str:
    name = str(item.get("name") or "").strip()
    role = str(item.get("role") or "").strip()
    desc = str(item.get("description") or "").strip()
    parts = []
    if name:
        parts.append(f"项目: {name}")
    if role:
        parts.append(f"角色: {role}")
    if desc:
        parts.append(desc)
    return "\n".join(parts)


def _work_text(item: Dict[str, Any]) -> str:
    company = str(item.get("company") or "").strip()
    title = str(item.get("title") or item.get("position") or "").strip()
    desc = str(item.get("description") or "").strip()
    parts = []
    if company:
        parts.append(f"公司: {company}")
    if title:
        parts.append(f"职位: {title}")
    if desc:
        parts.append(desc)
    return "\n".join(parts)


def build_candidate_chunks(candidate: CandidateModel) -> List[ChunkSpec]:
    out: List[ChunkSpec] = []

    skills = _norm_str_list(candidate.skills)
    if skills:
        text = "技能: " + ", ".join(skills)
        meta = {"block": "skill", "semantic_text": text[:2000]}
        out.append(ChunkSpec("skill", text, stable_hash_text(text), meta))

    projects = candidate.projects or []
    if isinstance(projects, list):
        for i, p in enumerate(projects):
            if not isinstance(p, dict):
                continue
            t = _project_text(p)
            if not t.strip():
                continue
            ptype = f"proj_{i}"
            _assert_profile_type(ptype)
            meta = {"block": "project", "index": i, "semantic_text": t[:2000]}
            out.append(ChunkSpec(ptype, t, stable_hash_text(t), meta))

    work = candidate.work_experience or []
    if isinstance(work, list):
        for i, w in enumerate(work):
            if not isinstance(w, dict):
                continue
            t = _work_text(w)
            if not t.strip():
                continue
            ptype = f"work_{i}"
            _assert_profile_type(ptype)
            meta = {"block": "work", "index": i, "semantic_text": t[:2000]}
            out.append(ChunkSpec(ptype, t, stable_hash_text(t), meta))

    tags = _norm_str_list(candidate.direction_tags)
    pname: List[str] = []
    if isinstance(projects, list):
        for p in projects:
            if isinstance(p, dict) and (p.get("name") or "").strip():
                pname.append(str(p.get("name")).strip())
    wlines: List[str] = []
    if isinstance(work, list):
        for w in work:
            if not isinstance(w, dict):
                continue
            c = str(w.get("company") or "").strip()
            t = str(w.get("title") or w.get("position") or "").strip()
            if c or t:
                wlines.append("·".join(x for x in [c, t] if x))

    n_proj = sum(
        1
        for i, p in enumerate(projects if isinstance(projects, list) else [])
        if isinstance(p, dict) and _project_text(p).strip()
    )
    n_work = sum(
        1
        for i, w in enumerate(work if isinstance(work, list) else [])
        if isinstance(w, dict) and _work_text(w).strip()
    )

    role_lines: List[str] = []
    if tags:
        role_lines.append("方向标签: " + ", ".join(tags))
    role_lines.append(f"经历摘要: 项目 {n_proj} 个, 工作/实习 {n_work} 个")
    if pname:
        role_lines.append("项目名: " + "; ".join(pname[:20]))
    if wlines:
        role_lines.append("公司与职位: " + "; ".join(wlines[:20]))
    cand_role_text = "\n".join(role_lines)
    meta = {"block": "cand_role", "semantic_text": cand_role_text[:2000]}
    out.append(
        ChunkSpec("cand_role", cand_role_text, stable_hash_text(cand_role_text), meta),
    )

    return out
