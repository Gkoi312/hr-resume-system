# app/parsers/resume_llm/simple_postprocess.py
"""Coerce LLM output toward layer1_simple_v1, validate shape, strip ungrounded list items."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

from app.llm.chat_client import LLMClientError
from app.parsers.resume_parser.skill_evidence import normalize_skill_list

_BASIC_KEYS = (
    "name",
    "phone",
    "email",
    "location",
    "birth_text",
    "target_role",
    "links",
    "raw_block",
)
_EDU_KEYS = ("school", "degree", "major", "start", "end", "raw_block")
_WORK_KEYS = ("company", "job_role", "job_type", "start", "end", "descriptions", "raw_block")
_PROJ_KEYS = ("project_name", "role", "start", "end", "descriptions", "raw_block")
_ADD_KEYS = ("languages", "certificates", "awards", "self_evaluation", "raw_block")
_JOB_TYPES = frozenset({"internship", "fulltime", "parttime", "unknown"})


def _s(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        t = v.strip()
        return [t] if t else []
    if isinstance(v, list):
        out: List[str] = []
        for x in v:
            t = _s(x)
            if t:
                out.append(t)
        return out
    return []


def _coerce_basic(b: Any) -> Dict[str, Any]:
    if not isinstance(b, dict):
        b = {}
    out: Dict[str, Any] = {}
    for k in _BASIC_KEYS:
        if k == "links":
            out["links"] = _str_list(b.get("links"))
            continue
        out[k] = _s(b.get(k))
    if not out["name"]:
        out["name"] = _s(b.get("full_name") or b.get("candidate_name") or b.get("姓名"))
    if not out["phone"]:
        out["phone"] = _s(b.get("mobile") or b.get("tel"))
    if not out["email"]:
        out["email"] = _s(b.get("mail"))
    if not out["location"]:
        out["location"] = _s(b.get("city") or b.get("address"))
    if not out["birth_text"]:
        out["birth_text"] = _s(b.get("birth") or b.get("birthday"))
    if not out["target_role"]:
        out["target_role"] = _s(b.get("summary") or b.get("objective") or b.get("求职意向"))
    rb = out["raw_block"]
    extra = _s(b.get("summary"))
    if extra and extra not in rb:
        out["raw_block"] = f"{rb}\n{extra}".strip() if rb else extra
    return out


def _coerce_edu_row(row: Any) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {k: "" for k in _EDU_KEYS}
    r = dict(row)
    school = _s(r.get("school") or r.get("school_name") or r.get("university"))
    degree = _s(r.get("degree"))
    major = _s(r.get("major") or r.get("field_of_study") or r.get("专业"))
    start = _s(r.get("start") or r.get("start_date") or r.get("from"))
    end = _s(r.get("end") or r.get("end_date") or r.get("to"))
    rb = _s(r.get("raw_block"))
    legacy_lines = (
        _str_list(r.get("details"))
        + _str_list(r.get("bullets"))
        + _str_list(r.get("highlights"))
    )
    if not rb and legacy_lines:
        rb = "\n".join(legacy_lines)
    return {
        "school": school,
        "degree": degree,
        "major": major,
        "start": start,
        "end": end,
        "raw_block": rb,
    }


def _normalize_job_type(raw: str) -> str:
    t = _s(raw).lower()
    if t == "contract":
        return "unknown"
    return t if t in _JOB_TYPES else "unknown"


def _infer_job_type(row: Dict[str, Any]) -> str:
    t = _normalize_job_type(row.get("job_type") or row.get("type") or "")
    if t in _JOB_TYPES and t != "unknown":
        return t
    blob = f"{row.get('job_role','') or row.get('title','')} {row.get('company','')} {row.get('raw_block','')}"
    blob_l = blob.lower()
    if "实习" in blob or "实习生" in blob or "intern" in blob_l:
        return "internship"
    if t in _JOB_TYPES:
        return t
    return "unknown"


def _coerce_work_row(row: Any) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "company": "",
            "job_role": "",
            "job_type": "unknown",
            "start": "",
            "end": "",
            "descriptions": [],
            "raw_block": "",
        }
    r = dict(row)
    company = _s(r.get("company") or r.get("employer") or r.get("organization"))
    job_role = _s(
        r.get("job_role")
        or r.get("title")
        or r.get("position")
        or r.get("job_title")
        or r.get("role")
    )
    start = _s(r.get("start") or r.get("start_date") or r.get("from"))
    end = _s(r.get("end") or r.get("end_date") or r.get("to"))
    descriptions = _str_list(r.get("descriptions"))
    if not descriptions:
        descriptions = _str_list(r.get("bullets") or r.get("responsibilities") or r.get("highlights"))
    rb = _s(r.get("raw_block"))
    tmp = {
        "company": company,
        "job_role": job_role,
        "job_type": _normalize_job_type(r.get("job_type") or r.get("type") or ""),
        "start": start,
        "end": end,
        "descriptions": descriptions,
        "raw_block": rb,
    }
    tmp["job_type"] = _infer_job_type(tmp)
    return tmp


def _coerce_proj_row(row: Any) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "project_name": "",
            "role": "",
            "start": "",
            "end": "",
            "descriptions": [],
            "raw_block": "",
        }
    r = dict(row)
    project_name = _s(r.get("project_name") or r.get("name") or r.get("title") or r.get("project"))
    role = _s(r.get("role"))
    start = _s(r.get("start") or r.get("start_date"))
    end = _s(r.get("end") or r.get("end_date"))
    descriptions = _str_list(r.get("descriptions"))
    if not descriptions:
        descriptions = _str_list(r.get("bullets") or r.get("highlights") or r.get("description"))
    rb = _s(r.get("raw_block"))
    return {
        "project_name": project_name,
        "role": role,
        "start": start,
        "end": end,
        "descriptions": descriptions,
        "raw_block": rb,
    }


def _coerce_additional(a: Any) -> Dict[str, Any]:
    if not isinstance(a, dict):
        a = {}
    langs = _str_list(a.get("languages"))
    certs = _str_list(a.get("certificates") or a.get("certifications"))
    awards = _str_list(a.get("awards"))
    self_eval = _s(a.get("self_evaluation") or a.get("self_intro"))
    rb = _s(a.get("raw_block"))
    return {
        "languages": langs,
        "certificates": certs,
        "awards": awards,
        "self_evaluation": self_eval,
        "raw_block": rb,
    }


def _collect_skills_before_additional_coerce(obj: Dict[str, Any]) -> List[str]:
    raw: List[str] = []
    raw.extend(_str_list(obj.get("skills")))
    add = obj.get("additional")
    if isinstance(add, dict):
        raw.extend(_str_list(add.get("skills")))
        if not raw and add.get("skill"):
            raw.extend(_str_list(add.get("skill")))
    return raw


def _split_skill_items(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[、,，;/；|]+", text)
    out: List[str] = []
    for p in parts:
        t = _s(p)
        if t and t not in out:
            out.append(t)
    return out


def _extract_after_label(line: str, labels: List[str]) -> str:
    t = _s(line)
    if not t:
        return ""
    low = t.lower()
    for lb in labels:
        lb_low = lb.lower()
        if t.startswith(lb) or low.startswith(lb_low):
            m = re.match(r"^[^:：]{1,20}[:：]\s*(.+)$", t)
            return _s(m.group(1)) if m else ""
    return ""


def _is_probably_narrative_line(text: str) -> bool:
    t = _s(text)
    if not t:
        return False
    if re.search(r"^\d{4}[.\-/]\d{1,2}", t):
        return False
    if re.match(r"^[\u4e00-\u9fffA-Za-z]{1,18}$", t) and len(t) <= 10:
        # likely title only (项目名/公司名/岗位名)
        return False
    if "技术栈" in t or re.match(r"^(技术|skills?)[:：]", t, re.I):
        return False
    if re.match(r"^\d+[\.、）)]", t):
        return len(t) >= 12
    return len(t) >= 12


def _extract_narrative_lines_from_raw(
    raw_block: str,
    *,
    skip_exact: List[str] | None = None,
    max_items: int = 5,
) -> List[str]:
    skip = {_s(x) for x in (skip_exact or []) if _s(x)}
    out: List[str] = []
    for ln in raw_block.splitlines():
        t = _s(ln)
        if not t or t in skip:
            continue
        if _is_probably_narrative_line(t) and t not in out:
            out.append(t)
        if len(out) >= max_items:
            break
    return out


def _backfill_project_descriptions_from_raw(obj: Dict[str, Any]) -> None:
    keys = ["项目描述", "工作内容", "项目成果", "职责", "负责", "描述", "成果", "details", "description"]
    for row in obj.get("projects") or []:
        if not isinstance(row, dict):
            continue
        if row.get("descriptions"):
            continue
        rb = _s(row.get("raw_block"))
        if not rb:
            continue
        hits: List[str] = []
        for ln in rb.splitlines():
            t = _s(ln)
            if not t:
                continue
            picked = _extract_after_label(t, keys)
            if picked:
                if picked not in hits:
                    hits.append(picked)
        if not hits:
            hits = _extract_narrative_lines_from_raw(
                rb,
                skip_exact=[_s(row.get("project_name"))],
                max_items=5,
            )
        row["descriptions"] = hits


def _backfill_work_descriptions_from_raw(obj: Dict[str, Any]) -> None:
    keys = ["工作内容", "主要职责", "职责", "负责", "业绩", "成果", "描述", "工作描述", "responsibilities"]
    for row in obj.get("work_experience") or []:
        if not isinstance(row, dict):
            continue
        if row.get("descriptions"):
            continue
        rb = _s(row.get("raw_block"))
        if not rb:
            continue
        hits: List[str] = []
        for ln in rb.splitlines():
            t = _s(ln)
            if not t:
                continue
            picked = _extract_after_label(t, keys)
            if picked and picked not in hits:
                hits.append(picked)
        if not hits:
            hits = _extract_narrative_lines_from_raw(
                rb,
                skip_exact=[_s(row.get("company")), _s(row.get("job_role"))],
                max_items=6,
            )
        row["descriptions"] = hits


def _backfill_skills_from_raw(obj: Dict[str, Any]) -> None:
    sk = obj.get("skills")
    add = obj.get("additional")
    if not isinstance(add, dict):
        add = {}
    labels = ["技术栈", "技术", "技能", "skills", "skill"]
    found: List[str] = []
    sources: List[str] = []
    sources.append(_s(add.get("raw_block")))
    for p in obj.get("projects") or []:
        if isinstance(p, dict):
            sources.append(_s(p.get("raw_block")))
    for w in obj.get("work_experience") or []:
        if isinstance(w, dict):
            sources.append(_s(w.get("raw_block")))
    for rb in sources:
        if not rb:
            continue
        for ln in rb.splitlines():
            picked = _extract_after_label(ln, labels)
            if not picked:
                continue
            for it in _split_skill_items(picked):
                if it and it not in found:
                    found.append(it)
    if found:
        existing_list: List[str] = []
        if isinstance(sk, list):
            existing_list = [str(x).strip() for x in sk if str(x).strip()]
        # Merge existing LLM skills + backfilled ones, then the caller will
        # normalize_skill_list() to dedupe + canonicalize.
        obj["skills"] = existing_list + found


def coerce_layer1_simple(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy with canonical keys so map_simple_to_layer1 receives expected shape."""
    out = dict(obj)
    out["basic"] = _coerce_basic(obj.get("basic"))
    out["education"] = [_coerce_edu_row(x) for x in (obj.get("education") or []) if x is not None]
    out["work_experience"] = [_coerce_work_row(x) for x in (obj.get("work_experience") or []) if x is not None]
    out["projects"] = [_coerce_proj_row(x) for x in (obj.get("projects") or []) if x is not None]
    out["additional"] = _coerce_additional(obj.get("additional"))
    out["skills"] = normalize_skill_list(_collect_skills_before_additional_coerce(obj))
    out["warnings"] = _str_list(obj.get("warnings"))
    _backfill_work_descriptions_from_raw(out)
    _backfill_project_descriptions_from_raw(out)
    _backfill_skills_from_raw(out)
    out["skills"] = normalize_skill_list(_str_list(out.get("skills")))
    return out


def validate_layer1_simple_shape(obj: Dict[str, Any]) -> None:
    """Raise LLMClientError if object does not match required layout (after coerce)."""
    b = obj.get("basic")
    if not isinstance(b, dict):
        raise LLMClientError("Invalid LLM output: basic must be object")
    for k in _BASIC_KEYS:
        if k == "links":
            if not isinstance(b.get("links"), list):
                raise LLMClientError("Invalid LLM output: basic.links must be array")
            continue
        if k not in b or not isinstance(b[k], str):
            raise LLMClientError(f"Invalid LLM output: basic.{k} must be string")

    for i, row in enumerate(obj.get("education") or []):
        if not isinstance(row, dict):
            raise LLMClientError(f"Invalid LLM output: education[{i}] must be object")
        for k in _EDU_KEYS:
            if k not in row or not isinstance(row[k], str):
                raise LLMClientError(f"Invalid LLM output: education[{i}].{k} must be string")

    for i, row in enumerate(obj.get("work_experience") or []):
        if not isinstance(row, dict):
            raise LLMClientError(f"Invalid LLM output: work_experience[{i}] must be object")
        for k in _WORK_KEYS:
            if k == "descriptions":
                if not isinstance(row.get("descriptions"), list):
                    raise LLMClientError(f"Invalid LLM output: work_experience[{i}].descriptions must be array")
                continue
            if k == "job_type":
                jt = _normalize_job_type(str(row.get("job_type")))
                if jt not in _JOB_TYPES:
                    raise LLMClientError(f"Invalid LLM output: work_experience[{i}].job_type invalid enum")
                continue
            if k not in row or not isinstance(row[k], str):
                raise LLMClientError(f"Invalid LLM output: work_experience[{i}].{k} must be string")

    for i, row in enumerate(obj.get("projects") or []):
        if not isinstance(row, dict):
            raise LLMClientError(f"Invalid LLM output: projects[{i}] must be object")
        for k in _PROJ_KEYS:
            if k == "descriptions":
                if not isinstance(row.get("descriptions"), list):
                    raise LLMClientError(f"Invalid LLM output: projects[{i}].descriptions must be array")
                continue
            if k not in row or not isinstance(row[k], str):
                raise LLMClientError(f"Invalid LLM output: projects[{i}].{k} must be string")

    sk = obj.get("skills")
    if not isinstance(sk, list):
        raise LLMClientError("Invalid LLM output: skills must be array")

    a = obj.get("additional")
    if not isinstance(a, dict):
        raise LLMClientError("Invalid LLM output: additional must be object")
    for k in _ADD_KEYS:
        if k in ("languages", "certificates", "awards"):
            if not isinstance(a.get(k), list):
                raise LLMClientError(f"Invalid LLM output: additional.{k} must be array")
            continue
        if k not in a or not isinstance(a[k], str):
            raise LLMClientError(f"Invalid LLM output: additional.{k} must be string")

    w = obj.get("warnings")
    if not isinstance(w, list):
        raise LLMClientError("Invalid LLM output: warnings must be array")


def _norm_match(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "").strip())


def _is_grounded_fragment(fragment: str, source: str, *, min_len: int = 12) -> bool:
    """True if fragment appears in source (NFKC). Substring match uses casefold() so skills
    normalized to lowercase (e.g. labview, llm) still ground against OCR (LabView, LLM).
    min_len is kept for call-site compatibility; matching is always substring-based.
    """
    _ = min_len
    frag = _norm_match(fragment)
    src = _norm_match(source)
    if not frag:
        return True
    return frag.casefold() in src.casefold()


_RELAX_KEEP = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


def _normalize_relaxed(s: str) -> str:
    """
    Relaxed normalization for long narrative matching:
    keep CJK/letters/digits only, drop punctuation/whitespace/newlines.
    """
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", s)
    return "".join(_RELAX_KEEP.findall(t))


def _is_grounded_fragment_relaxed(fragment: str, source: str, *, min_len: int = 12) -> bool:
    _ = min_len
    frag = _normalize_relaxed(fragment)
    src = _normalize_relaxed(source)
    if not frag:
        return True
    return frag.casefold() in src.casefold()


def compact_ungrounded_work(
    obj: Dict[str, Any],
    source_text: str,
    *,
    skip_source_grounding: bool = False,
) -> None:
    """Remove work rows whose company/title/raw_block do not appear in source (likely invented)."""
    if skip_source_grounding:
        return
    src_n = _norm_match(source_text)
    if not src_n.strip():
        obj["work_experience"] = []
        return
    kept: List[Dict[str, Any]] = []
    for row in obj.get("work_experience") or []:
        if not isinstance(row, dict):
            continue
        c = _norm_match(str(row.get("company") or ""))
        t = _norm_match(str(row.get("job_role") or row.get("title") or ""))
        rb = _norm_match(str(row.get("raw_block") or ""))
        anchored = False
        if c and len(c) > 1 and c in src_n:
            anchored = True
        if t and len(t) > 1 and t in src_n:
            anchored = True
        if rb and len(rb) > 3 and rb in src_n:
            anchored = True
        if anchored:
            kept.append(row)
    obj["work_experience"] = kept


def compact_ungrounded_projects(
    obj: Dict[str, Any],
    source_text: str,
    *,
    skip_source_grounding: bool = False,
) -> None:
    """Remove project rows not anchored by name or raw_block substring in source."""
    if skip_source_grounding:
        return
    src_n = _norm_match(source_text)
    if not src_n.strip():
        obj["projects"] = []
        return
    kept: List[Dict[str, Any]] = []
    for row in obj.get("projects") or []:
        if not isinstance(row, dict):
            continue
        n = _norm_match(str(row.get("project_name") or row.get("name") or ""))
        rb = _norm_match(str(row.get("raw_block") or ""))
        anchored = False
        if n and len(n) > 1 and n in src_n:
            anchored = True
        if rb and len(rb) > 3 and rb in src_n:
            anchored = True
        if anchored:
            kept.append(row)
    obj["projects"] = kept


def filter_list_fields_against_source(obj: Dict[str, Any], source_text: str) -> None:
    """
    Drop work.descriptions / project.descriptions (and legacy list keys) not substrings of resume text.
    Mutates obj in place.
    """
    src = source_text or ""

    for row in obj.get("work_experience") or []:
        if not isinstance(row, dict):
            continue
        for key in ("descriptions", "bullets"):
            b = row.get(key)
            if isinstance(b, list):
                row[key] = [x for x in b if _is_grounded_fragment_relaxed(str(x), src)]

    for row in obj.get("projects") or []:
        if not isinstance(row, dict):
            continue
        for key in ("descriptions", "bullets"):
            b = row.get(key)
            if isinstance(b, list):
                row[key] = [x for x in b if _is_grounded_fragment_relaxed(str(x), src)]

    skl = obj.get("skills")
    if isinstance(skl, list):
        obj["skills"] = [x for x in skl if _is_grounded_fragment(str(x), src, min_len=4)]

    add = obj.get("additional")
    if isinstance(add, dict):
        for key in ("languages", "certificates", "awards"):
            lst = add.get(key)
            if isinstance(lst, list):
                add[key] = [x for x in lst if _is_grounded_fragment(str(x), src, min_len=4)]


def strip_invented_entries_when_unreadable(obj: Dict[str, Any], source_text: str) -> None:
    """
    If resume text is too short / looks non-text, clear structured entries that are likely hallucinated.
    Mutates obj.
    """
    src = (source_text or "").strip()
    if len(src) >= 80:
        return
    warns = list(obj.get("warnings") or [])
    tag = "input_too_short_or_unreadable_cleared_llm_fields"
    if tag not in warns:
        warns.append(tag)
    obj["warnings"] = warns
    obj["education"] = []
    obj["work_experience"] = []
    obj["projects"] = []
    b = obj.get("basic")
    if isinstance(b, dict):
        for k in _BASIC_KEYS:
            if k == "links":
                b["links"] = []
            else:
                b[k] = ""
    obj["skills"] = []
    a = obj.get("additional")
    if isinstance(a, dict):
        a["languages"] = []
        a["certificates"] = []
        a["awards"] = []
        a["self_evaluation"] = ""
        a["raw_block"] = ""


_RE_GARBLED = re.compile(
    r"(?:PDF|二进制|binary|font|encoding|不可读|无法解析)",
    re.IGNORECASE,
)


def apply_garbled_input_heuristic(obj: Dict[str, Any], source_text: str) -> None:
    """If model flagged garbage PDF and source lacks CJK letters, clear invented blocks."""
    src = source_text or ""
    warns = " ".join(str(w) for w in (obj.get("warnings") or []))
    cjk = bool(re.search(r"[\u4e00-\u9fff]", src))
    if not cjk and (_RE_GARBLED.search(warns) or len(src.strip()) < 40):
        strip_invented_entries_when_unreadable(obj, src)
