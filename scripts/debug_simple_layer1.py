from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parsers.resume_paddle.ppstructure_client import predict_file_bytes_to_page_dicts
from app.parsers.resume_parser.resume_llm_layer1.extract import extract_resume_simple_json
from app.parsers.resume_parser.resume_llm_layer1.map_simple_to_layer1 import map_simple_to_layer1
from app.parsers.resume_parser.resume_llm_layer1.pipeline import build_document_from_layer1_mapped
from app.parsers.resume_parser import merge_parsed_layer1_document


def collect_rec_texts(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "rec_texts" and isinstance(v, list):
                out.extend([str(x) for x in v])
            else:
                out.extend(collect_rec_texts(v))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(collect_rec_texts(it))
    return out


async def main() -> None:
    pdf = ROOT / "testOCR" / "111.png"
    out_dir = ROOT / "testOCR" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_bytes = pdf.read_bytes()

    # 1) Paddle 抽取
    pages = predict_file_bytes_to_page_dicts(file_bytes, pdf.name)
    (out_dir / "111_debug_paddle_pages.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2) rec_texts -> simple json（不做清洗）
    rec_lines: List[str] = []
    for p in pages:
        rec_lines.extend(collect_rec_texts(p))
    source_text = "\n".join(rec_lines)

    simple = await extract_resume_simple_json(source_text)
    (out_dir / "111_debug_simple.json").write_text(
        json.dumps(simple, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) simple -> layer1 -> merged parsed
    layer1 = map_simple_to_layer1(simple)
    warn = [str(w) for w in (simple.get("warnings") or []) if str(w).strip()]
    warn.append("parsed_via_resume_llm_from_paddle_rec_texts")

    doc = build_document_from_layer1_mapped(
        layer1,
        document_id="debug-111",
        candidate_id="debug-cand",
        file_name=pdf.name,
        text_extraction_method="paddle_rec_texts_llm",
        ocr_used=True,
        total_pages=len(pages),
        source_text=source_text,
        extra_warnings=warn,
    )
    merged = merge_parsed_layer1_document(doc, full_text=source_text)
    (out_dir / "111_debug_layer1_merged.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())