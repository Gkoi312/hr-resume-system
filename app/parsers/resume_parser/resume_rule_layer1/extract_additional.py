# app/parsers/resume_layer1/extract_additional.py
"""Rule-based additional_info arrays using subsection_markers when present."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _skill_item(val: str, section: str) -> Dict[str, Any]:
    v = val.strip()
    return {
        "value": v,
        "raw_text": v,
        "category_hint": "",
        "confidence": 0.5 if v else 0.0,
        "source_section": section,
    }


def _parse_skills_chunk(chunk: str) -> List[Dict[str, Any]]:
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    out: List[Dict[str, Any]] = []
    for ln in lines:
        if re.match(
            r"^(专业技能|掌握技能|技术栈|技能|Skills?|Technical\s*Skills)\s*[:：]?$",
            ln,
            re.I,
        ):
            continue
        if "：" in ln or ":" in ln:
            rest = ln.split("：", 1)[-1].split(":", 1)[-1].strip()
        else:
            rest = ln
        parts = re.split(r"[,，、;；/\s|]+", rest)
        for p in parts:
            if len(p.strip()) >= 2:
                out.append(_skill_item(p.strip(), "skills"))
    return out[:50]


def _parse_language_lines(chunk: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ln in chunk.splitlines():
        t = ln.strip()
        if not t:
            continue
        if re.match(r"^(语言能力|Languages?)\s*[:：]?$", t, re.I):
            continue
        m = re.match(r"^(.{1,40}?)[：:]\s*(.+)$", t)
        if m:
            out.append(
                {
                    "name": m.group(1).strip(),
                    "raw_text": t,
                    "level": m.group(2).strip()[:80],
                    "confidence": 0.55,
                }
            )
        elif re.search(r"CET|TEM|IELTS|TOEFL|日语|英语|English", t, re.I):
            out.append(
                {
                    "name": t[:40],
                    "raw_text": t,
                    "level": "",
                    "confidence": 0.45,
                }
            )
    return out[:20]


def extract_additional_info(
    raw_block: str,
    markers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    text = raw_block or ""
    skills_raw: List[Dict[str, Any]] = []
    certificates: List[Dict[str, Any]] = []
    awards: List[Dict[str, Any]] = []
    languages: List[Dict[str, Any]] = []
    competitions: List[Dict[str, Any]] = []
    publications: List[Dict[str, Any]] = []

    if markers:
        for m in markers:
            kind = m.get("kind") or ""
            s = int(m.get("start", 0))
            e = int(m.get("end", 0))
            if s < 0 or e < s or e > len(text):
                e = min(max(e, s), len(text))
            chunk = text[s:e]
            if kind == "skills":
                skills_raw.extend(_parse_skills_chunk(chunk))
            elif kind == "languages":
                languages.extend(_parse_language_lines(chunk))
            elif kind == "certificates":
                for ln in chunk.splitlines():
                    t = ln.strip()
                    if t and not re.match(r"^(证书|Certifications?)", t, re.I):
                        certificates.append(
                            {
                                "name": t[:120],
                                "raw_text": t,
                                "issuer": "",
                                "issue_date": "",
                                "confidence": 0.45,
                            }
                        )
            elif kind == "awards":
                for ln in chunk.splitlines():
                    t = ln.strip()
                    if t and not re.match(r"^(荣誉|获奖|Awards?)", t, re.I):
                        awards.append(
                            {
                                "name": t[:120],
                                "raw_text": t,
                                "level": "",
                                "date": "",
                                "confidence": 0.45,
                            }
                        )
            elif kind == "competitions":
                for ln in chunk.splitlines():
                    t = ln.strip()
                    if t:
                        competitions.append(
                            {
                                "name": t[:120],
                                "raw_text": t,
                                "award": "",
                                "date": "",
                                "confidence": 0.4,
                            }
                        )
            elif kind == "publications":
                for ln in chunk.splitlines():
                    t = ln.strip()
                    if t:
                        publications.append(
                            {
                                "title": t[:200],
                                "raw_text": t,
                                "venue": "",
                                "date": "",
                                "confidence": 0.4,
                            }
                        )
    else:
        skills_raw = _parse_skills_chunk(text)

    return {
        "skills_raw": skills_raw,
        "certificates": certificates[:30],
        "awards": awards[:30],
        "languages": languages,
        "competitions": competitions[:20],
        "publications": publications[:20],
    }
