from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parsers.resume_parser.candidate_profile_builder import (  # noqa: E402
    get_candidate_bind_for_resume,
)
from app.parsers.resume_parser.resume_llm_layer1.map_simple_to_layer1 import (  # noqa: E402
    map_simple_to_layer1,
)
from app.parsers.resume_parser.resume_llm_layer1.pipeline import (  # noqa: E402
    build_document_from_layer1_mapped,
)
from app.parsers.resume_parser import merge_parsed_layer1_document  # noqa: E402


def dump_one(simple_path: Path, out_dir: Path) -> None:
    simple = json.loads(simple_path.read_text(encoding="utf-8"))

    layer1 = map_simple_to_layer1(simple)
    base = simple_path.stem.replace("_paddle_llm_simple", "")
    layer1_out = out_dir / f"{base}_layer1_extracted.json"
    layer1_out.write_text(json.dumps(layer1, ensure_ascii=False, indent=2), encoding="utf-8")

    candidate_profile = get_candidate_bind_for_resume({"layer_1_extracted": layer1})
    cand_out = out_dir / f"{base}_candidate_profile.json"
    cand_out.write_text(
        json.dumps(candidate_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Also overwrite layer1_merged.json to reflect corrected date normalization.
    file_name_map = {"gjx_paddle_llm_simple": "gjx.pdf", "jhz_paddle_llm_simple": "jhz.pdf", "wyf_paddle_llm_simple": "wyf.png"}
    ocr_used = True
    text_method = "simple_json_debug"
    source_text = ""
    basic = layer1.get("basic") if isinstance(layer1, dict) else {}
    if isinstance(basic, dict):
        source_text = str(basic.get("raw_block") or "")

    doc = build_document_from_layer1_mapped(
        layer1,
        document_id=f"debug-{base}",
        candidate_id=f"debug-cand-{base}",
        file_name=file_name_map.get(simple_path.stem, f"{base}.bin"),
        text_extraction_method=text_method,
        ocr_used=ocr_used,
        total_pages=None,
        source_text=source_text,
        extra_warnings=[],
    )
    merged = merge_parsed_layer1_document(doc, full_text=source_text)
    merged_out = out_dir / f"{base}_layer1_merged.json"
    merged_out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"DONE: {simple_path.name} -> {layer1_out.name}, {cand_out.name}, {merged_out.name}")


def main() -> None:
    out_dir = ROOT / "testOCR" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed filenames as requested: gjx/jhz/wyf.
    inputs: List[str] = [
        "gjx_paddle_llm_simple.json",
        "jhz_paddle_llm_simple.json",
        "wyf_paddle_llm_simple.json",
    ]
    missing = [name for name in inputs if not (out_dir / name).exists()]
    if missing:
        raise SystemExit("Missing simple json files:\n" + "\n".join(missing))

    for name in inputs:
        dump_one(out_dir / name, out_dir)


if __name__ == "__main__":
    main()

