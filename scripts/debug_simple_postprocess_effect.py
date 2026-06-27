from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parsers.resume_parser.resume_llm_layer1.simple_postprocess import (  # noqa: E402
    apply_garbled_input_heuristic,
    coerce_layer1_simple,
    compact_ungrounded_projects,
    compact_ungrounded_work,
    filter_list_fields_against_source,
    validate_layer1_simple_shape,
)


def _collect_rec_texts(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "rec_texts" and isinstance(v, list):
                out.extend(str(x) for x in v)
            else:
                out.extend(_collect_rec_texts(v))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_collect_rec_texts(it))
    return out


def _build_source_text_from_paddle_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [x for x in _collect_rec_texts(data) if str(x).strip()]
    return "\n".join(lines)


def _project_desc_counts(simple: Dict[str, Any]) -> List[int]:
    rows = simple.get("projects") or []
    out: List[int] = []
    for r in rows:
        if isinstance(r, dict):
            desc = r.get("descriptions")
            out.append(len(desc) if isinstance(desc, list) else 0)
        else:
            out.append(0)
    return out


def _skills_count(simple: Dict[str, Any]) -> int:
    n = 0
    sk = simple.get("skills")
    if isinstance(sk, list):
        n += len(sk)
    add = simple.get("additional")
    if isinstance(add, dict):
        s2 = add.get("skills")
        if isinstance(s2, list):
            n += len(s2)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply simple_postprocess to an existing simple JSON and compare before/after."
    )
    parser.add_argument(
        "--simple",
        default=str(ROOT / "testOCR" / "output" / "gjx_debug_simple.json"),
        help="Path to input simple JSON",
    )
    parser.add_argument(
        "--source-text",
        default="",
        help="Optional source text file path. If omitted, use --paddle-json rec_texts.",
    )
    parser.add_argument(
        "--paddle-json",
        default=str(ROOT / "testOCR" / "output" / "gjx_debug_paddle_pages.json"),
        help="Path to Paddle *_res.json used to build source text from rec_texts.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output path for postprocessed JSON. Default: <simple>_postprocessed.json",
    )
    args = parser.parse_args()

    simple_path = Path(args.simple)
    if not simple_path.is_file():
        raise SystemExit(f"simple json not found: {simple_path}")

    if args.source_text:
        st_path = Path(args.source_text)
        if not st_path.is_file():
            raise SystemExit(f"source text file not found: {st_path}")
        source_text = st_path.read_text(encoding="utf-8")
    else:
        pj_path = Path(args.paddle_json)
        if not pj_path.is_file():
            raise SystemExit(f"paddle json not found: {pj_path}")
        source_text = _build_source_text_from_paddle_json(pj_path)

    before = json.loads(simple_path.read_text(encoding="utf-8"))
    after = coerce_layer1_simple(before)
    apply_garbled_input_heuristic(after, source_text)
    validate_layer1_simple_shape(after)
    filter_list_fields_against_source(after, source_text)
    compact_ungrounded_work(after, source_text)
    compact_ungrounded_projects(after, source_text)

    out_path = Path(args.out) if args.out else simple_path.with_name(simple_path.stem + "_postprocessed.json")
    out_path.write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "simple_in": str(simple_path),
        "source_len": len(source_text),
        "out": str(out_path),
        "before": {
            "projects": len(before.get("projects") or []),
            "project_description_counts": _project_desc_counts(before),
            "skills_count": _skills_count(before),
            "warnings": before.get("warnings") or [],
        },
        "after": {
            "projects": len(after.get("projects") or []),
            "project_description_counts": _project_desc_counts(after),
            "skills_count": _skills_count(after),
            "warnings": after.get("warnings") or [],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

