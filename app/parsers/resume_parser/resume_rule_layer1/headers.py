# app/parsers/resume_layer1/headers.py
"""Section header classification (education > work > projects > additional)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class SegmentKind(str, Enum):
    BASIC = "basic_info"
    EDUCATION = "education"
    WORK = "work_experience"
    PROJECTS = "projects"
    ADDITIONAL = "additional_info"


# (regex, segment, subsection for ADDITIONAL else None)
# Priority: first matching rule wins — list is ordered education, work, projects, then additional.
_HEADER_RULES: list[tuple[re.Pattern[str], SegmentKind, Optional[str]]] = [
    (
        re.compile(
            r"^(教育背景|教育经历|学历|在读经历|Education|Academic\s*Background)\s*[:：]?\s*$",
            re.I,
        ),
        SegmentKind.EDUCATION,
        None,
    ),
    (
        re.compile(
            r"^(工作经历|工作经验|实习经历|全职经历|职业经历|任职经历|"
            r"Employment|Work\s*Experience|Professional\s*Experience|Career\s*History)\s*[:：]?\s*$",
            re.I,
        ),
        SegmentKind.WORK,
        None,
    ),
    (
        re.compile(
            r"^(项目经历|项目经验|主要项目|科研成果|Research\s*Projects|"
            r"Project\s*Experience|Selected\s*Projects)\s*[:：]?\s*$",
            re.I,
        ),
        SegmentKind.PROJECTS,
        None,
    ),
    (
        re.compile(r"^项目\s*[:：]?\s*$"),
        SegmentKind.PROJECTS,
        None,
    ),
    (
        re.compile(
            r"^(自我评价|个人评价|自我描述|个人简介|Summary|About\s*Me)\s*[:：]?\s*$",
            re.I,
        ),
        SegmentKind.ADDITIONAL,
        "self_introduction",
    ),
    (
        re.compile(
            r"^(专业技能|掌握技能|技术栈|职业技能|技能|Skills?|Technical\s*Skills)\s*[:：]?\s*$",
            re.I,
        ),
        SegmentKind.ADDITIONAL,
        "skills",
    ),
    (
        re.compile(
            r"^(证书|资质认证|Certifications?)\s*[:：]?\s*$",
            re.I,
        ),
        SegmentKind.ADDITIONAL,
        "certificates",
    ),
    (
        re.compile(r"^(语言能力|Languages?)\s*[:：]?\s*$", re.I),
        SegmentKind.ADDITIONAL,
        "languages",
    ),
    (
        re.compile(
            r"^(荣誉|获奖|奖项|Awards?|Honors?)\s*[:：]?\s*$",
            re.I,
        ),
        SegmentKind.ADDITIONAL,
        "awards",
    ),
    (
        re.compile(r"^(竞赛|比赛)\s*[:：]?\s*$"),
        SegmentKind.ADDITIONAL,
        "competitions",
    ),
    (
        re.compile(r"^(论文|发表|Publications?)\s*[:：]?\s*$", re.I),
        SegmentKind.ADDITIONAL,
        "publications",
    ),
    (
        re.compile(
            r"^(基本信息|个人信息|联系方式|个人资料|概况|Profile|Contact|Personal\s*Information)\s*[:：]?\s*$",
            re.I,
        ),
        SegmentKind.BASIC,
        None,
    ),
]


@dataclass
class HeaderMatch:
    segment: SegmentKind
    subsection: Optional[str]
    confidence: float


def is_probable_header_line(
    stripped: str,
    line_index: int,
    prev_blank: bool,
    next_blank: bool,
) -> bool:
    if not stripped or len(stripped) > 40:
        return False
    if re.match(r"^[•·\-\*]\s*", stripped):
        return False
    if re.match(r"^\d+[\.、）)]\s*", stripped):
        return False
    if line_index < 30:
        return True
    return prev_blank or next_blank


def classify_header(stripped: str) -> Optional[HeaderMatch]:
    t = stripped.strip()
    for pat, seg, sub in _HEADER_RULES:
        if pat.match(t):
            return HeaderMatch(segment=seg, subsection=sub, confidence=0.95)
    return None
