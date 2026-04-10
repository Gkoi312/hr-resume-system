# app/parsers/resume_layer1/segmenter.py
"""Five-way resume segmentation + additional_info subsection markers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional, Tuple

from app.parsers.resume_parser.resume_rule_layer1.headers import (
    HeaderMatch,
    SegmentKind,
    classify_header,
    is_probable_header_line,
)
from app.parsers.resume_parser.resume_rule_layer1.preprocess import build_line_metas, normalize_newlines


@dataclass
class SectionSpan:
    start: int = 0
    end: int = 0
    raw_block: str = ""
    header_hits: List[dict] = field(default_factory=list)


@dataclass
class SectioningResult:
    raw_text: str
    segments: Dict[SegmentKind, SectionSpan]
    warnings: List[str]
    subsection_markers: List[dict]


def _line_end(start: int, line: str) -> int:
    return start + len(line)


def build_subsection_markers(
    lines: List[Tuple[int, str]],
) -> List[dict]:
    """Offsets relative to joined additional raw_block."""
    if not lines:
        return []
    markers: List[dict] = []
    pos = 0
    open_m: Optional[dict] = None
    for i, (_gs, line) in enumerate(lines):
        stripped = line.strip()
        prev_blank = i == 0 or not lines[i - 1][1].strip()
        next_blank = i + 1 >= len(lines) or not lines[i + 1][1].strip()
        is_header = is_probable_header_line(stripped, i, prev_blank, next_blank)
        hm: Optional[HeaderMatch] = classify_header(stripped) if is_header else None
        if hm and hm.segment == SegmentKind.ADDITIONAL and hm.subsection:
            if open_m is not None:
                open_m["end"] = pos
                markers.append(open_m)
            open_m = {
                "kind": hm.subsection,
                "start": pos,
                "end": 0,
                "header_raw": stripped,
            }
        if i > 0:
            pos += 1
        pos += len(line)
    if open_m is not None:
        open_m["end"] = pos
        markers.append(open_m)
    return markers


def segment_resume(raw_text: str) -> SectioningResult:
    text = normalize_newlines(raw_text or "")
    line_metas = build_line_metas(text)
    if not line_metas:
        empty = SectionSpan(0, 0, "", [])
        segs = {k: SectionSpan(0, 0, "", []) for k in SegmentKind}
        return SectioningResult(
            raw_text=text,
            segments=segs,
            warnings=["empty_text"],
            subsection_markers=[],
        )

    lines_by_seg: DefaultDict[SegmentKind, List[Tuple[int, str]]] = defaultdict(list)
    current = SegmentKind.BASIC
    buffer: List[Tuple[int, str]] = []
    header_hits: DefaultDict[SegmentKind, List[dict]] = defaultdict(list)
    saw_nonbasic_header = False

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        lines_by_seg[current].extend(buffer)
        buffer = []

    n = len(line_metas)
    for i, lm in enumerate(line_metas):
        prev_blank = line_metas[i - 1].is_blank if i > 0 else True
        next_blank = line_metas[i + 1].is_blank if i + 1 < n else True

        if lm.is_blank:
            buffer.append((lm.start, lm.text))
            continue

        hm: Optional[HeaderMatch] = None
        if not lm.noise_hint and is_probable_header_line(
            lm.stripped, lm.index, prev_blank, next_blank
        ):
            hm = classify_header(lm.stripped)

        if hm:
            if hm.segment == SegmentKind.BASIC and current == SegmentKind.BASIC:
                buffer.append((lm.start, lm.text))
                continue
            if (
                hm.segment == SegmentKind.ADDITIONAL
                and current == SegmentKind.ADDITIONAL
                and hm.subsection
            ):
                buffer.append((lm.start, lm.text))
                continue

            flush()
            current = hm.segment
            saw_nonbasic_header = saw_nonbasic_header or hm.segment != SegmentKind.BASIC
            header_hits[current].append(
                {
                    "line_index": lm.index,
                    "kind": hm.segment.value,
                    "subsection": hm.subsection,
                    "confidence": hm.confidence,
                    "matched": lm.stripped[:80],
                }
            )
            buffer.append((lm.start, lm.text))
            continue

        buffer.append((lm.start, lm.text))

    flush()

    warnings: List[str] = []
    if not saw_nonbasic_header:
        only_basic = bool(lines_by_seg[SegmentKind.BASIC]) and not any(
            lines_by_seg[k] for k in SegmentKind if k != SegmentKind.BASIC
        )
        if only_basic or not any(lines_by_seg.values()):
            warnings.append("no_section_headers_detected")

    segments: Dict[SegmentKind, SectionSpan] = {}
    for kind in SegmentKind:
        pairs = lines_by_seg[kind]
        if not pairs:
            segments[kind] = SectionSpan(0, 0, "", list(header_hits[kind]))
            continue
        raw_block = "\n".join(t for _, t in pairs)
        start = pairs[0][0]
        last_s, last_t = pairs[-1]
        end = _line_end(last_s, last_t)
        segments[kind] = SectionSpan(
            start=start,
            end=end,
            raw_block=raw_block,
            header_hits=list(header_hits[kind]),
        )

    add_lines = lines_by_seg[SegmentKind.ADDITIONAL]
    subsection_markers = build_subsection_markers(add_lines)

    return SectioningResult(
        raw_text=text,
        segments=segments,
        warnings=warnings,
        subsection_markers=subsection_markers,
    )
