# app/parsers/resume_layer1/extract_basic.py
"""Rule-based basic_info field extraction from preamble raw_block."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"1[3-9]\d{9}|"
    r"\+?86\s*[-]?\s*1[3-9]\d{9}|"
    r"\(\d{3}\)\s*\d{3}[-]?\d{4}|"
    r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}"
)
_URL_RE = re.compile(
    r"(https?://[^\s]+|(?:www\.)[^\s]+|github\.com/[^\s]+)",
    re.I,
)


def _fv(
    value: str,
    raw_text: str,
    confidence: float,
    source_section: str,
    span: Tuple[int, int],
) -> Dict[str, Any]:
    return {
        "value": value,
        "raw_text": raw_text,
        "confidence": confidence,
        "source_section": source_section,
        "source_span": [span[0], span[1]],
    }


def _abs_span(block_start: int, local_start: int, local_end: int) -> Tuple[int, int]:
    return (block_start + local_start, block_start + local_end)


def extract_basic_info(raw_block: str, block_start: int = 0) -> Dict[str, Any]:
    full = raw_block or ""
    lines = [ln.strip() for ln in full.splitlines() if ln.strip()]

    email_m = _EMAIL_RE.search(full)
    email_val = email_m.group(0) if email_m else ""
    email_span = _abs_span(block_start, email_m.start(), email_m.end()) if email_m else (0, 0)

    phone_m = _PHONE_RE.search(full)
    phone_val = phone_m.group(0).strip() if phone_m else ""
    phone_span = _abs_span(block_start, phone_m.start(), phone_m.end()) if phone_m else (0, 0)

    name_val = ""
    name_span = (0, 0)
    for line in lines[:8]:
        if _EMAIL_RE.search(line) or _PHONE_RE.search(line):
            continue
        if re.match(r"^[\u4e00-\u9fa5a-zA-Z\s·]{2,30}$", line) and len(line) <= 30:
            idx = full.find(line)
            if idx >= 0:
                name_val = line
                name_span = _abs_span(block_start, idx, idx + len(line))
            break

    loc_val = ""
    loc_span = (0, 0)
    mloc = re.search(
        r"(现居|居住地|地址)[:：]\s*([^\n]+)",
        full,
    )
    if mloc:
        loc_val = mloc.group(2).strip()[:120]
        s = mloc.start(2)
        e = mloc.end(2)
        loc_span = _abs_span(block_start, s, e)

    target_role = ""
    tr_span = (0, 0)
    mtr = re.search(r"(求职意向|期望职位|意向岗位)[:：]\s*([^\n]+)", full)
    if mtr:
        target_role = mtr.group(2).strip()[:120]
        tr_span = _abs_span(block_start, mtr.start(2), mtr.end(2))

    exp_loc = ""
    el_span = (0, 0)
    mel = re.search(r"(期望城市|期望地点)[:：]\s*([^\n]+)", full)
    if mel:
        exp_loc = mel.group(2).strip()[:120]
        el_span = _abs_span(block_start, mel.start(2), mel.end(2))

    salary_val = ""
    sal_cur = ""
    sal_period = ""
    sal_span = (0, 0)
    msal = re.search(
        r"(期望薪资|薪资期望|期望工资)[:：]\s*([^\n]+)",
        full,
    )
    if msal:
        raw_s = msal.group(2).strip()
        salary_val = raw_s[:80]
        sal_span = _abs_span(block_start, msal.start(2), msal.end(2))
        if "k" in raw_s.lower() or "K" in raw_s:
            sal_period = "month"
        if re.search(r"元|万|k|K", raw_s):
            sal_cur = "CNY"

    links: List[Dict[str, Any]] = []
    for um in _URL_RE.finditer(full):
        url = um.group(0).strip().rstrip(").,;，。")
        lt = "link"
        if "github.com" in url.lower():
            lt = "github"
        elif "linkedin" in url.lower():
            lt = "linkedin"
        links.append(
            {
                "type": lt,
                "value": url,
                "raw_text": url,
                "confidence": 0.75,
                "source_section": "basic_info",
                "source_span": list(_abs_span(block_start, um.start(), um.end())),
            }
        )
        if len(links) >= 8:
            break

    return {
        "name": _fv(name_val, name_val, 0.7 if name_val else 0.0, "basic_info", name_span),
        "gender": _fv("", "", 0.0, "basic_info", (0, 0)),
        "birth_year": _fv("", "", 0.0, "basic_info", (0, 0)),
        "birth_date": _fv("", "", 0.0, "basic_info", (0, 0)),
        "phone": _fv(phone_val, phone_val, 0.85 if phone_val else 0.0, "basic_info", phone_span),
        "email": _fv(email_val, email_val, 0.9 if email_val else 0.0, "basic_info", email_span),
        "location": _fv(loc_val, loc_val, 0.65 if loc_val else 0.0, "basic_info", loc_span),
        "hometown": _fv("", "", 0.0, "basic_info", (0, 0)),
        "expected_salary": {
            "value": salary_val,
            "raw_text": salary_val,
            "currency": sal_cur,
            "period": sal_period,
            "confidence": 0.6 if salary_val else 0.0,
            "source_section": "basic_info",
            "source_span": [sal_span[0], sal_span[1]],
        },
        "target_role": _fv(
            target_role, target_role, 0.65 if target_role else 0.0, "basic_info", tr_span
        ),
        "expected_location": _fv(
            exp_loc, exp_loc, 0.65 if exp_loc else 0.0, "basic_info", el_span
        ),
        "personal_links": links,
        "raw_block": full,
    }
