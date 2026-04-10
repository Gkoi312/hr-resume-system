# app/parsers/resume_layer1/extract_work.py
"""Rule-based work_experience[] extraction."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Tuple

_INTERN_HINT = re.compile(r"实习|internship", re.I)
_DATE_HEAD = re.compile(
    r"^(\d{4}\s*[-–—.]\s*\d{4}|\d{4}\s*[-–—.]\s*至今|\d{4}\s*至今)",
)


def _wf(value: str, raw: str, conf: float) -> Dict[str, Any]:
    return {"value": value, "raw_text": raw, "confidence": conf}


def _bullet(text: str, span: Tuple[int, int]) -> Dict[str, Any]:
    return {"text": text, "source_span": [span[0], span[1]], "confidence": 0.45}


def extract_work_experience(raw_block: str, block_start: int = 0) -> List[Dict[str, Any]]:
    text = raw_block or ""
    if not text.strip():
        return []

    lines = text.splitlines()
    entries: List[Dict[str, Any]] = []
    cur: List[str] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        block = "\n".join(cur).strip()
        if not block:
            cur = []
            return
        local = text.find(block)
        if local < 0:
            local = 0
        bs = block_start + local

        company = ""
        role = ""
        start_d, end_d = "", ""
        head = cur[0].strip() if cur else ""
        m = re.match(
            r"^(\d{4}\s*[-–—.]\s*\d{4}|\d{4}\s*[-–—.]\s*至今|\d{4}\s*至今)\s+(.+)$",
            head,
        )
        if m:
            period = m.group(1).replace(" ", "")
            rest = m.group(2).strip()
            parts = re.split(r"\s{2,}|\t+|　+", rest, maxsplit=1)
            company = parts[0][:120] if parts else rest[:120]
            role = parts[1][:120] if len(parts) > 1 else ""
            dm = re.match(r"(\d{4}).*?(\d{4}|至今)", period)
            if dm:
                start_d, end_d = dm.group(1), dm.group(2)
        else:
            company = head[:120]

        exp_type = "internship" if _INTERN_HINT.search(block) else "fulltime"

        bullets: List[Dict[str, Any]] = []
        pos = 0
        for ln in cur[1:]:
            t = ln.strip()
            if not t:
                continue
            if re.match(r"^[•·\-\*]\s*", t) or re.match(r"^\d+[\.、）)]\s*", t):
                clean = re.sub(r"^[•·\-\*\d\.、）)]\s*", "", t)
                li_local = text.find(ln, local)
                if li_local < 0:
                    li_local = local + pos
                ls = block_start + li_local + (len(ln) - len(ln.lstrip()))
                le = block_start + li_local + len(ln)
                bullets.append(_bullet(clean[:500], (ls, le)))
            pos += len(ln) + 1

        eid = f"exp_{uuid.uuid4().hex[:10]}"
        entries.append(
            {
                "experience_id": eid,
                "experience_type": exp_type,
                "company_name": _wf(company, company, 0.55 if company else 0.0),
                "department": _wf("", "", 0.0),
                "role_title": _wf(role, role, 0.5 if role else 0.0),
                "location": _wf("", "", 0.0),
                "start_date": _wf(start_d, start_d, 0.5 if start_d else 0.0),
                "end_date": _wf(end_d, end_d, 0.5 if end_d else 0.0),
                "duration_text": "",
                "responsibilities_raw": bullets,
                "achievements_raw": [],
                "tech_stack_raw": [],
                "raw_description": "",
                "raw_block": block,
            }
        )
        cur = []

    header_line = re.compile(
        r"^(工作经历|工作经验|实习经历|Employment|Work\s*Experience)\s*[:：]?\s*$",
        re.I,
    )
    for ln in lines:
        if header_line.match(ln.strip()):
            continue
        st = ln.strip()
        if st and _DATE_HEAD.match(st) and cur:
            flush()
        cur.append(ln)
    flush()

    if not entries and text.strip():
        eid = f"exp_{uuid.uuid4().hex[:10]}"
        entries.append(
            {
                "experience_id": eid,
                "experience_type": "fulltime",
                "company_name": _wf("", "", 0.0),
                "department": _wf("", "", 0.0),
                "role_title": _wf("", "", 0.0),
                "location": _wf("", "", 0.0),
                "start_date": _wf("", "", 0.0),
                "end_date": _wf("", "", 0.0),
                "duration_text": "",
                "responsibilities_raw": [],
                "achievements_raw": [],
                "tech_stack_raw": [],
                "raw_description": text.strip()[:2000],
                "raw_block": text.strip(),
            }
        )
    return entries[:20]
