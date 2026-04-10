# app/parsers/resume_paddle/flatten.py
"""PP-Structure JSON (parsing_res_list) -> plain text for rule-based Layer1."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Skip non-text layout blocks; keep header/text/table titles as lines.
_SKIP_LABELS = frozenset(
    {
        "image",
        "chart",
        "seal",
        "footer_image",
        "header_image",
        "number",
        "footnote",
        "aside_text",
    }
)


def _bbox_key(block: Dict[str, Any]) -> Tuple[int, int]:
    bb = block.get("block_bbox") or [0, 0, 0, 0]
    try:
        return (int(bb[1]), int(bb[0]))
    except (TypeError, ValueError, IndexError):
        return (0, 0)


def _page_index(page_dict: Dict[str, Any]) -> int:
    v = page_dict.get("page_index")
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def flatten_ppstructure_pages(page_dicts: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Merge per-page PP-Structure JSON dicts into reading-order plain text.

    Returns (text, warnings).
    """
    warnings: List[str] = []
    if not page_dicts:
        warnings.append("paddle_empty_pages")
        return "", warnings

    blocks: List[Tuple[int, int, int, Dict[str, Any]]] = []
    for pd in page_dicts:
        pidx = _page_index(pd)
        lst = pd.get("parsing_res_list") or []
        if not isinstance(lst, list):
            warnings.append("paddle_bad_parsing_res_list")
            continue
        for b in lst:
            if not isinstance(b, dict):
                continue
            label = (b.get("block_label") or "").strip()
            if label in _SKIP_LABELS:
                continue
            order = b.get("block_order")
            try:
                o = int(order) if order is not None else 10**9
            except (TypeError, ValueError):
                o = 10**9
            blocks.append((pidx, o, len(blocks), b))

    blocks.sort(key=lambda t: (t[0], t[1], _bbox_key(t[3]), t[2]))

    lines: List[str] = []
    for _p, _o, _i, b in blocks:
        content = (b.get("block_content") or "").strip()
        if not content:
            continue
        label = (b.get("block_label") or "").strip()
        if label == "paragraph_title":
            lines.append(content)
            lines.append("")
        elif label == "table":
            lines.append(content)
            lines.append("")
        else:
            lines.append(content)
            lines.append("")

    text = "\n".join(lines).strip()
    if not text:
        warnings.append("paddle_flatten_empty")
    return text, warnings
