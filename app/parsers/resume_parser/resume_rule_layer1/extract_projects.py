# app/parsers/resume_layer1/extract_projects.py
"""Rule-based projects[] extraction."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List

from app.parsers.resume_parser.resume_rule_layer1.extract_work import _DATE_HEAD, _bullet, _wf


def extract_projects(raw_block: str, block_start: int = 0) -> List[Dict[str, Any]]:
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
        head = cur[0].strip() if cur else ""
        pname = head[:120]
        role = ""
        start_d, end_d = "", ""
        m = re.match(
            r"^(\d{4}\s*[-–—.]\s*\d{4}|\d{4}\s*[-–—.]\s*至今|\d{4}\s*至今)\s+(.+)$",
            head,
        )
        if m:
            rest = m.group(2).strip()
            pname = rest[:120]
            period = m.group(1)
            dm = re.match(r"(\d{4}).*?(\d{4}|至今)", period.replace(" ", ""))
            if dm:
                start_d, end_d = dm.group(1), dm.group(2)

        bullets: List[Dict[str, Any]] = []
        for ln in cur[1:]:
            t = ln.strip()
            if not t:
                continue
            if re.match(r"^[•·\-\*]\s*", t) or re.match(r"^\d+[\.、）)]\s*", t):
                clean = re.sub(r"^[•·\-\*\d\.、）)]\s*", "", t)
                li_local = text.find(ln, local)
                if li_local < 0:
                    li_local = local
                ls = block_start + li_local + (len(ln) - len(ln.lstrip()))
                le = block_start + li_local + len(ln)
                bullets.append(_bullet(clean[:500], (ls, le)))

        pid = f"proj_{uuid.uuid4().hex[:10]}"
        entries.append(
            {
                "project_id": pid,
                "project_name": _wf(pname, pname, 0.55 if pname else 0.0),
                "role": _wf(role, role, 0.0),
                "start_date": _wf(start_d, start_d, 0.5 if start_d else 0.0),
                "end_date": _wf(end_d, end_d, 0.5 if end_d else 0.0),
                "project_description_raw": "",
                "responsibilities_raw": bullets,
                "achievements_raw": [],
                "tech_stack_raw": [],
                "raw_block": block,
            }
        )
        cur = []

    header_line = re.compile(
        r"^(项目经历|项目经验|项目|Projects?)\s*[:：]?\s*$",
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
        pid = f"proj_{uuid.uuid4().hex[:10]}"
        entries.append(
            {
                "project_id": pid,
                "project_name": _wf("", "", 0.0),
                "role": _wf("", "", 0.0),
                "start_date": _wf("", "", 0.0),
                "end_date": _wf("", "", 0.0),
                "project_description_raw": text.strip()[:2000],
                "responsibilities_raw": [],
                "achievements_raw": [],
                "tech_stack_raw": [],
                "raw_block": text.strip(),
            }
        )
    return entries[:20]
