# app/parsers/resume_parser/layer1_normalizer.py
"""Normalize Layer1 blocks into a stable, sortable, downstream-friendly shape.

This module is shared by both:
- LLM simple -> layer_1_extracted (map_simple_to_layer1)
- Rule/Paddle layer1 build (resume_rule_layer1/assemble)

It adds:
- school_tier for education
- stable sorting (latest first) based on normalized dates
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from app.parsers.resume_parser.school_tiers import tier_for_school

_PRESENT_TOKENS = {"至今", "现在", "present", "current", "ongoing", "在读"}
_PRESENT_SORT_KEY = (9999, 12)

_YEARS_RE = re.compile(
    r"(19|20)\d{2}\s*[.\-/年]\s*(0?[1-9]|1[0-2])"
)  # only used by _parse_year_month

# Capture full year (e.g. 2024) and month (e.g. 09).
# Previously this only captured the leading "19"/"20", which caused outputs like "0020-09".
_START_END_DATE_RE = re.compile(
    # IMPORTANT: month alternation order must prefer 10-12 (1[0-2])
    # otherwise "2020/12" could match month="1" from "12".
    r"((?:19|20)\d{2})\s*[./\-年]\s*(1[0-2]|0?[1-9])"
)
_YEAR_ONLY_RE = re.compile(r"(19|20)\d{2}")


def _nfkc_lower(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").lower()


def _clean_str(v: Any) -> str:
    return str(v or "").strip()


def _norm_month(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _parse_year_month(raw: str) -> Tuple[str, Optional[Tuple[int, int]], bool]:
    s = _nfkc_lower(_clean_str(raw))
    if not s:
        return "", None, False

    if any(tok in s for tok in _PRESENT_TOKENS):
        return "present", _PRESENT_SORT_KEY, True

    ym = _START_END_DATE_RE.search(s)
    if ym:
        year = int(ym.group(1))
        month = int(ym.group(2))
        return _norm_month(year, month), (year, month), False

    y = _YEAR_ONLY_RE.search(s)
    if y:
        year = int(y.group(0))
        return f"{year:04d}", (year, 1), False

    return "", None, False


def _safe_sort_key(key: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    return key if key is not None else (-1, -1)


def normalize_layer1_blocks(layer1: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutates a shallow copy of `layer1`:
    - sorts education/work/projects by end/start normalized dates
    - adds school_tier
    - overwrites `start`/`end` with normalized values only
    """
    out = dict(layer1 or {})

    # Education
    edu_rows: List[Dict[str, Any]] = []
    for item in (out.get("education") or []) or []:
        if not isinstance(item, dict):
            continue
        start_norm, start_key, _ = _parse_year_month(_clean_str(item.get("start")))
        end_norm, end_key, _ = _parse_year_month(_clean_str(item.get("end")))
        school = _clean_str(item.get("school"))
        edu_rows.append(
            {
                **item,
                "school": school,
                "degree": _clean_str(item.get("degree")),
                "major": _clean_str(item.get("major")),
                "start": start_norm,
                "end": end_norm,
                "raw_block": _clean_str(item.get("raw_block")),
                "school_tier": tier_for_school(school),
                "_start_sort": _safe_sort_key(start_key),
                "_end_sort": _safe_sort_key(end_key),
            }
        )

    edu_rows.sort(key=lambda x: (x["_end_sort"], x["_start_sort"]), reverse=True)
    for row in edu_rows:
        row.pop("_start_sort", None)
        row.pop("_end_sort", None)
    out["education"] = edu_rows

    # Work
    work_rows: List[Dict[str, Any]] = []
    for item in (out.get("work_experience") or []) or []:
        if not isinstance(item, dict):
            continue
        start_norm, start_key, _ = _parse_year_month(_clean_str(item.get("start")))
        end_norm, end_key, _ = _parse_year_month(_clean_str(item.get("end")))
        work_rows.append(
            {
                **item,
                "company": _clean_str(item.get("company")),
                "job_role": _clean_str(item.get("job_role")),
                "job_type": _clean_str(item.get("job_type")) or "unknown",
                "start": start_norm,
                "end": end_norm,
                "raw_block": _clean_str(item.get("raw_block")),
                "descriptions": [
                    str(x).strip() for x in (item.get("descriptions") or []) if str(x).strip()
                ],
                "_start_sort": _safe_sort_key(start_key),
                "_end_sort": _safe_sort_key(end_key),
            }
        )

    work_rows.sort(key=lambda x: (x["_end_sort"], x["_start_sort"]), reverse=True)
    for row in work_rows:
        row.pop("_start_sort", None)
        row.pop("_end_sort", None)
    out["work_experience"] = work_rows

    # Projects
    proj_rows: List[Dict[str, Any]] = []
    for item in (out.get("projects") or []) or []:
        if not isinstance(item, dict):
            continue
        start_norm, start_key, _ = _parse_year_month(_clean_str(item.get("start")))
        end_norm, end_key, _ = _parse_year_month(_clean_str(item.get("end")))
        proj_rows.append(
            {
                **item,
                "project_name": _clean_str(item.get("project_name")),
                "role": _clean_str(item.get("role")),
                "start": start_norm,
                "end": end_norm,
                "raw_block": _clean_str(item.get("raw_block")),
                "descriptions": [
                    str(x).strip() for x in (item.get("descriptions") or []) if str(x).strip()
                ],
                "_start_sort": _safe_sort_key(start_key),
                "_end_sort": _safe_sort_key(end_key),
            }
        )

    proj_rows.sort(key=lambda x: (x["_end_sort"], x["_start_sort"]), reverse=True)
    for row in proj_rows:
        row.pop("_start_sort", None)
        row.pop("_end_sort", None)
    out["projects"] = proj_rows

    return out

