# app/parsers/resume_layer1/preprocess.py
"""Normalize resume text and build line offset index."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


def normalize_newlines(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    return t


def build_line_map(text: str) -> List[Tuple[int, str]]:
    """Return (start_char_offset, line_without_newline) for each line."""
    if not text:
        return []
    out: List[Tuple[int, str]] = []
    pos = 0
    for part in text.split("\n"):
        out.append((pos, part))
        pos += len(part) + 1
    return out


_FOOTER_NOISE = re.compile(
    r"^(\d+|第\s*\d+\s*页|page\s*\d+\s*/\s*\d+)\s*$",
    re.I,
)


@dataclass
class LineMeta:
    index: int
    start: int
    text: str
    stripped: str
    is_blank: bool
    noise_hint: bool


def build_line_metas(text: str) -> List[LineMeta]:
    line_map = build_line_map(text)
    metas: List[LineMeta] = []
    for i, (start, raw) in enumerate(line_map):
        stripped = raw.strip()
        metas.append(
            LineMeta(
                index=i,
                start=start,
                text=raw,
                stripped=stripped,
                is_blank=not stripped,
                noise_hint=bool(stripped and _FOOTER_NOISE.match(stripped)),
            )
        )
    return metas
