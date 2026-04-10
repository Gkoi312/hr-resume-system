# app/parsers/resume_layer1/extract_education.py
"""Rule-based education[] extraction."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Tuple

_EDU_HEADER_SKIP = re.compile(
    r"^(教育背景|教育经历|学历|在读经历|Education|Academic\s*Background)\s*[:：]?\s*$",
    re.I,
)

_DEGREE_RE = re.compile(
    r"(本科|硕士|研究生|博士|大专|专科|学士|本科以上|"
    r"B\.?S\.?|M\.?S\.?|Ph\.?D\.?|Bachelor|Master|Doctor)",
    re.I,
)


def _ef(value: str, raw: str, conf: float, span: Tuple[int, int]) -> Dict[str, Any]:
    return {
        "value": value,
        "raw_text": raw,
        "confidence": conf,
        "source_span": [span[0], span[1]],
    }


def _edf(value: str, raw: str, conf: float) -> Dict[str, Any]:
    return {"value": value, "raw_text": raw, "confidence": conf}


def _abs(block_start: int, ls: int, le: int) -> Tuple[int, int]:
    return (block_start + ls, block_start + le)


def extract_education(raw_block: str, block_start: int = 0) -> List[Dict[str, Any]]:
    text = raw_block or ""
    if not text.strip():
        return []

    chunks: List[str] = []
    cur: List[str] = []
    for para in re.split(r"\n\s*\n+", text):
        p = para.strip()
        if not p:
            continue
        if cur and (
            _DEGREE_RE.search(p[:40])
            or re.match(r"^\d{4}\s*[-–—.]", p)
            or re.search(r"(大学|学院|学校|University|College)", p, re.I)
        ):
            chunks.append("\n".join(cur))
            cur = [p]
        else:
            cur.append(p)
    if cur:
        chunks.append("\n".join(cur))

    if len(chunks) == 1 and "\n" in chunks[0]:
        lines = [ln for ln in chunks[0].splitlines() if ln.strip()]
        if len(lines) >= 2:
            chunks = []
            buf: List[str] = []
            for ln in lines:
                if buf and (
                    re.match(r"^\d{4}\s*[-–—.]", ln.strip())
                    or (
                        _DEGREE_RE.search(ln[:30])
                        and re.search(r"(大学|学院|University)", ln, re.I)
                    )
                ):
                    chunks.append("\n".join(buf))
                    buf = [ln]
                else:
                    buf.append(ln)
            if buf:
                chunks.append("\n".join(buf))

    out: List[Dict[str, Any]] = []
    for ch in chunks[:12]:
        local = text.find(ch)
        if local < 0:
            local = 0
        bs = block_start + local
        degree_m = _DEGREE_RE.search(ch)
        degree = degree_m.group(1) if degree_m else ""
        school = ""
        for ln in ch.splitlines():
            st = ln.strip()
            if not st or _EDU_HEADER_SKIP.match(st):
                continue
            if re.search(r"(大学|学院|学校|University|College)", ln, re.I):
                school = st[:200]
                break
        if not school:
            for ln in ch.splitlines():
                st = ln.strip()
                if st and not _EDU_HEADER_SKIP.match(st):
                    school = st[:200]
                    break

        dates = re.findall(
            r"(\d{4})\s*[-–—.年]\s*(\d{4}|至今)",
            ch,
        )
        start_d, end_d = "", ""
        if dates:
            start_d, end_d = dates[0][0], dates[0][1]

        eid = f"edu_{uuid.uuid4().hex[:10]}"
        span_line = (0, min(len(ch), 200))
        out.append(
            {
                "education_id": eid,
                "school_name": _ef(school, school, 0.55 if school else 0.0, _abs(bs, 0, len(ch))),
                "college_name": _ef("", "", 0.0, (0, 0)),
                "degree": _ef(degree, degree, 0.6 if degree else 0.0, _abs(bs, 0, len(ch))),
                "major": _ef("", "", 0.0, (0, 0)),
                "start_date": _edf(start_d, start_d, 0.5 if start_d else 0.0),
                "end_date": _edf(end_d, end_d, 0.5 if end_d else 0.0),
                "duration_text": "",
                "gpa": _ef("", "", 0.0, (0, 0)),
                "rank": _ef("", "", 0.0, (0, 0)),
                "courses_raw": [],
                "honors_raw": [],
                "raw_block": ch,
            }
        )
    return out
