# app/parsers/resume_layer1/aligned_from_rules.py
"""Convert rule-based extract_* output (legacy nested shapes) -> layer_1_extracted aligned with simple."""

from __future__ import annotations

from typing import Any, Dict, List


def _fv(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, dict):
        return str(x.get("value") or "").strip()
    return str(x).strip()


def _links_to_strings(links: Any) -> List[str]:
    out: List[str] = []
    for L in links or []:
        if isinstance(L, dict):
            v = (L.get("value") or L.get("raw_text") or "").strip()
            if v:
                out.append(v)
        elif isinstance(L, str) and L.strip():
            out.append(L.strip())
    return out[:32]


def basic_from_rule_extract(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": _fv(d.get("name")),
        "phone": _fv(d.get("phone")),
        "email": _fv(d.get("email")),
        "location": _fv(d.get("location")),
        "birth_text": _fv(d.get("birth_date")),
        "target_role": _fv(d.get("target_role")),
        "links": _links_to_strings(d.get("personal_links")),
        "raw_block": str(d.get("raw_block") or ""),
    }


def education_from_rule_extract(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        sd = e.get("start_date")
        ed = e.get("end_date")
        out.append(
            {
                "school": _fv(e.get("school_name")),
                "degree": _fv(e.get("degree")),
                "major": _fv(e.get("major")),
                "start": _fv(sd),
                "end": _fv(ed),
                "raw_block": str(e.get("raw_block") or ""),
            }
        )
    return out


def work_from_rule_extract(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for w in entries or []:
        if not isinstance(w, dict):
            continue
        descs: List[str] = []
        for b in w.get("responsibilities_raw") or []:
            if isinstance(b, dict):
                t = (b.get("text") or "").strip()
                if t:
                    descs.append(t)
        jt = str(w.get("experience_type") or "unknown").strip().lower()
        if jt not in ("internship", "fulltime", "parttime", "unknown"):
            jt = "unknown"
        out.append(
            {
                "company": _fv(w.get("company_name")),
                "job_role": _fv(w.get("role_title")),
                "job_type": jt,
                "start": _fv(w.get("start_date")),
                "end": _fv(w.get("end_date")),
                "descriptions": descs,
                "raw_block": str(w.get("raw_block") or ""),
            }
        )
    return out


def projects_from_rule_extract(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in entries or []:
        if not isinstance(p, dict):
            continue
        descs: List[str] = []
        for b in p.get("responsibilities_raw") or []:
            if isinstance(b, dict):
                t = (b.get("text") or "").strip()
                if t:
                    descs.append(t)
        out.append(
            {
                "project_name": _fv(p.get("project_name")),
                "role": _fv(p.get("role")),
                "start": _fv(p.get("start_date")),
                "end": _fv(p.get("end_date")),
                "descriptions": descs,
                "raw_block": str(p.get("raw_block") or ""),
            }
        )
    return out


def skills_from_rule_extract(extra: Dict[str, Any]) -> List[str]:
    return [
        str(s.get("value", "")).strip()
        for s in (extra or {}).get("skills_raw") or []
        if isinstance(s, dict) and str(s.get("value", "")).strip()
    ]


def additional_from_rule_extract(
    extra: Dict[str, Any],
    raw_block: str,
    markers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    languages: List[str] = []
    for lang in (extra or {}).get("languages") or []:
        if not isinstance(lang, dict):
            continue
        rt = str(lang.get("raw_text") or "").strip()
        if rt:
            languages.append(rt)
            continue
        nm, lv = str(lang.get("name") or "").strip(), str(lang.get("level") or "").strip()
        if nm or lv:
            languages.append(f"{nm} {lv}".strip())
    certificates = [
        str(c.get("name", "")).strip()
        for c in (extra or {}).get("certificates") or []
        if isinstance(c, dict) and str(c.get("name", "")).strip()
    ]
    awards = [
        str(a.get("name", "")).strip()
        for a in (extra or {}).get("awards") or []
        if isinstance(a, dict) and str(a.get("name", "")).strip()
    ]
    self_eval = ""
    text = raw_block or ""
    for m in markers or []:
        if (m or {}).get("kind") != "self_introduction":
            continue
        s, e = int(m.get("start", 0)), int(m.get("end", 0))
        if 0 <= s <= e <= len(text):
            self_eval = text[s:e].strip()[:4000]
        break
    return {
        "languages": languages[:30],
        "certificates": certificates[:30],
        "awards": awards[:30],
        "self_evaluation": self_eval,
        "raw_block": text.strip(),
    }
