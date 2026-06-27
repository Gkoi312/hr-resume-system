from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parsers.resume_paddle.ppstructure_client import (  # noqa: E402
    predict_file_bytes_to_page_dicts,
)
from app.parsers.resume_parser.candidate_profile_builder import (  # noqa: E402
    get_candidate_bind_for_resume,
)
from app.parsers.resume_parser.resume_llm_layer1.extract import (  # noqa: E402
    extract_resume_simple_json,
)
from app.parsers.resume_parser.resume_llm_layer1.map_simple_to_layer1 import (  # noqa: E402
    map_simple_to_layer1,
)
from app.parsers.resume_parser.resume_llm_layer1.pipeline import (  # noqa: E402
    build_document_from_layer1_mapped,
)
from app.parsers.resume_parser import merge_parsed_layer1_document  # noqa: E402


def collect_rec_texts(obj: Any) -> List[str]:
    """Collect OCR-recognized texts from PP-Structure output."""
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


async def dump_one(path: Path, out_dir: Path) -> None:
    file_bytes = path.read_bytes()

    # 1) Paddle 抽取 -> pages
    pages = predict_file_bytes_to_page_dicts(file_bytes, path.name)
    (out_dir / f"{path.stem}_paddle_pages.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2) rec_texts -> simple json (LLM)
    rec_lines: List[str] = []
    for p in pages:
        rec_lines.extend(collect_rec_texts(p))
    source_text = "\n".join(rec_lines)

    simple_path = out_dir / f"{path.stem}_paddle_llm_simple.json"
    if simple_path.exists():
        simple = json.loads(simple_path.read_text(encoding="utf-8"))
    else:
        simple = await extract_resume_simple_json(source_text)
        simple_path.write_text(
            json.dumps(simple, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 3) simple -> layer1 -> merged parsed (layer_1_extracted)
    layer1 = map_simple_to_layer1(simple)
    (out_dir / f"{path.stem}_layer1_extracted.json").write_text(
        json.dumps(layer1, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    doc = build_document_from_layer1_mapped(
        layer1,
        document_id=f"debug-{path.stem}",
        candidate_id=f"debug-cand-{path.stem}",
        file_name=path.name,
        text_extraction_method="paddle_rec_texts_llm",
        ocr_used=True,
        total_pages=len(pages),
        source_text=source_text,
        extra_warnings=[],
    )
    merged = merge_parsed_layer1_document(doc, full_text=source_text)
    (out_dir / f"{path.stem}_layer1_merged.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4) merged parsed -> candidate profile bind (formerly "layer2")
    candidate_profile = get_candidate_bind_for_resume(merged)
    (out_dir / f"{path.stem}_candidate_profile.json").write_text(
        json.dumps(candidate_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"DONE: {path.name}")


async def main() -> None:
    out_dir = ROOT / "testOCR" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ROOT / "testOCR" / "gjx.pdf",
        ROOT / "testOCR" / "jhz.pdf",
        ROOT / "testOCR" / "wyf.png",
    ]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit("Missing input files:\n" + "\n".join(missing))

    for p in files:
        await dump_one(p, out_dir)


if __name__ == "__main__":
    asyncio.run(main())

