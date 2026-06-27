from __future__ import annotations

import os
import sys
import uuid
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi.testclient import TestClient

# ---- configure runtime env (must be before importing app.main) ----
# Prefer "Paddle + rules Layer1" for this integration test:
# - RESUME_LLM_ENABLED=0 to avoid needing LLM credentials
# - RESUME_PADDLE_ENABLED=1 to force PP-Structure path
os.environ.setdefault("RESUME_LLM_ENABLED", "1")
os.environ.setdefault("RESUME_LLM_VISION", "0")
os.environ.setdefault("RESUME_VLLM_ENABLED", "0")
os.environ.setdefault("RESUME_PADDLE_ENABLED", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# PaddleOCR hard dependency check (skip integration test if missing).
try:
    import paddleocr  # noqa: F401
except ModuleNotFoundError:
    print(
        "Skip integration test: `paddleocr` not found in current Python env.\n"
        "Please run this script inside your conda env `hr_resume`.\n"
        "Ensure PaddleOCR dependencies are installed in that same env, then rerun.\n"
        "The script already forces RESUME_PADDLE_ENABLED=1.\n"
    )
    raise SystemExit(0)

from app.main import app  # noqa: E402


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _upload_pdf(client: TestClient, pdf_path: Path) -> Dict[str, Any]:
    pdf_bytes = _read_bytes(pdf_path)
    files = {
        "file": (pdf_path.name, pdf_bytes, "application/pdf"),
    }
    # candidate_id not provided -> endpoint will create a placeholder candidate first
    r = client.post(
        "/api/v1/resumes/upload",
        files=files,
        data={},
    )
    if r.status_code != 200:
        raise RuntimeError(f"upload failed: {pdf_path} status={r.status_code} body={r.text}")
    return r.json()


def _get_candidate(client: TestClient, candidate_id: uuid.UUID) -> Dict[str, Any]:
    r = client.get(f"/api/v1/candidates/{candidate_id}")
    if r.status_code != 200:
        raise RuntimeError(f"get candidate failed: {candidate_id} status={r.status_code} body={r.text}")
    return r.json()


def _non_empty_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0


def _extract_resume_from_response(client: TestClient, resume_id: uuid.UUID) -> Dict[str, Any]:
    r = client.get(f"/api/v1/resumes/{resume_id}")
    if r.status_code != 200:
        raise RuntimeError(f"get resume failed: {resume_id} status={r.status_code} body={r.text}")
    return r.json()


def _check_paddle_hint(resume: Dict[str, Any]) -> Tuple[bool, str]:
    parsed = resume.get("parsed") if isinstance(resume, dict) else None
    parsed = parsed if isinstance(parsed, dict) else {}
    meta = (parsed.get("document_meta") or {}) if isinstance(parsed, dict) else {}
    warnings = meta.get("warnings") or []
    if isinstance(warnings, list):
        warnings_str = " ".join(str(x) for x in warnings)
    else:
        warnings_str = str(warnings)

    method = meta.get("text_extraction_method") or ""
    used_paddle = "paddle" in str(method).lower() or "paddle" in warnings_str.lower()
    return used_paddle, warnings_str[:500]


def main() -> None:
    pdf_gjx = ROOT / "testOCR" / "gjx.pdf"
    pdf_jhz = ROOT / "testOCR" / "jhz.pdf"
    if not pdf_gjx.exists():
        raise SystemExit(f"missing file: {pdf_gjx}")
    if not pdf_jhz.exists():
        raise SystemExit(f"missing file: {pdf_jhz}")

    # TestClient will trigger app lifespan: init_db + create tables.
    with TestClient(app, base_url="http://test") as client:
        results: list[Dict[str, Any]] = []
        for pdf_path in (pdf_gjx, pdf_jhz):
            uploaded = _upload_pdf(client, pdf_path)
            resume_id = uuid.UUID(str(uploaded["id"]))
            candidate_id = uuid.UUID(str(uploaded["candidate_id"]))

            resume = _extract_resume_from_response(client, resume_id)
            candidate = _get_candidate(client, candidate_id)

            used_paddle, warnings_hint = _check_paddle_hint(resume)
            parsed = resume.get("parsed") or {}
            l1 = parsed.get("layer_1_extracted") or {}
            l1_basic = l1.get("basic") or {}
            l1_edu = l1.get("education") or []
            l1_work = l1.get("work_experience") or []
            l1_proj = l1.get("projects") or []
            l1_add = l1.get("additional") or {}
            l1_skills = l1.get("skills") or l1_add.get("skills") or []

            # These fields are expected to be filled if Paddle+Layer1 succeeded.
            has_edu = _non_empty_list(candidate.get("education"))
            has_work = _non_empty_list(candidate.get("work_experience"))
            has_proj = _non_empty_list(candidate.get("projects"))
            has_skills = _non_empty_list(candidate.get("skills"))

            results.append(
                {
                    "pdf": pdf_path.name,
                    "resume_id": str(resume_id),
                    "candidate_id": str(candidate_id),
                    "resume_status": resume.get("status"),
                    "resume_file_name": resume.get("file_name"),
                    "resume_file_path": resume.get("file_path"),
                    "resume_parse_status": (((resume.get("parsed") or {}).get("document_meta") or {}).get("parse_status")),
                    "text_extraction_method": (((resume.get("parsed") or {}).get("document_meta") or {}).get("text_extraction_method")),
                    "parsed_l1_counts": {
                        "education": len(l1_edu) if isinstance(l1_edu, list) else None,
                        "work_experience": len(l1_work) if isinstance(l1_work, list) else None,
                        "projects": len(l1_proj) if isinstance(l1_proj, list) else None,
                        "skills": len(l1_skills) if isinstance(l1_skills, list) else None,
                    },
                    "l1_basic_sample": {
                        "name": l1_basic.get("name"),
                        "email": l1_basic.get("email"),
                        "phone": l1_basic.get("phone"),
                        "birth_text": l1_basic.get("birth_text"),
                    },
                    "used_paddle_hint": used_paddle,
                    "paddle_warnings_hint": warnings_hint,
                    "candidate_fields": {
                        "name": candidate.get("name"),
                        "email": candidate.get("email"),
                        "phone": candidate.get("phone"),
                        "education_count": len(candidate.get("education") or []),
                        "work_count": len(candidate.get("work_experience") or []),
                        "projects_count": len(candidate.get("projects") or []),
                        "skills_count": len(candidate.get("skills") or []),
                        "years_of_experience": candidate.get("years_of_experience"),
                        "direction_tags": candidate.get("direction_tags"),
                    },
                    "assert_summary": {
                        "has_any_work_or_projects_or_edu": (has_edu or has_work or has_proj),
                        "has_skills": has_skills,
                    },
                }
            )

        print(json.dumps(results, ensure_ascii=False, indent=2))

        # Hard assertion: must write candidate + at least some structured info.
        # If Paddle isn't available, you'll see the warnings_hint and this will fail loudly.
        any_missing = False
        for r in results:
            if not r["assert_summary"]["has_any_work_or_projects_or_edu"]:
                any_missing = True
        if any_missing:
            raise SystemExit(
                "Integration test failed: candidate received no education/work/projects. "
                "Check whether the intended Paddle+LLM path ran successfully.\n"
                "Inspect `paddle_warnings_hint`, `text_extraction_method`, and `parsed_l1_counts` in output."
            )


if __name__ == "__main__":
    main()

