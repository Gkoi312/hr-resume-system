from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Try to force UTF-8 output to avoid seeing escaped \uXXXX and to prevent
# GBK console encoding errors.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.database.repository.resume_repository import resume_repository  # noqa: E402
from app.database.repository.candidate_repository import candidate_repository  # noqa: E402
from app.database.session import init_db  # noqa: E402


def _pretty(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def _counts_or_zero(v: Any) -> int:
    return len(v) if isinstance(v, list) else 0


async def inspect_one(*, resume_id: uuid.UUID, candidate_id: uuid.UUID) -> Dict[str, Any]:
    resume = await resume_repository.get_by_id(resume_id)
    if not resume:
        raise RuntimeError(f"resume not found: {resume_id}")
    if str(resume.candidate_id) != str(candidate_id):
        # Still allow mismatch, but include both IDs to help debugging.
        pass

    cand = await candidate_repository.get_by_id(candidate_id)
    if not cand:
        raise RuntimeError(f"candidate not found: {candidate_id}")

    parsed = resume.parsed or {}
    l1 = parsed.get("layer_1_extracted") or {}
    return {
        "resume": {
            **resume.to_dict(),
            "layer1_counts": {
                "education": _counts_or_zero(l1.get("education")),
                "work_experience": _counts_or_zero(l1.get("work_experience")),
                "projects": _counts_or_zero(l1.get("projects")),
                "skills": _counts_or_zero(l1.get("skills")),
            },
        },
        "candidate": {
            **cand.to_dict(),
        },
    }


async def main_async() -> None:
    # Usage:
    #   python scripts/inspect_resume_and_candidates.py <resume_id> <candidate_id> [<resume_id> <candidate_id> ...]
    if len(sys.argv) < 3 or (len(sys.argv) - 1) % 2 != 0:
        raise SystemExit(
            "Usage: python scripts/inspect_resume_and_candidates.py "
            "<resume_id> <candidate_id> [<resume_id> <candidate_id> ...]"
        )

    pairs: List[tuple[uuid.UUID, uuid.UUID]] = []
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        pairs.append((uuid.UUID(args[i]), uuid.UUID(args[i + 1])))

    # Ensure tables exist (for local runs / clean DB).
    await init_db()

    out: List[Dict[str, Any]] = []
    for resume_id, candidate_id in pairs:
        out.append(await inspect_one(resume_id=resume_id, candidate_id=candidate_id))

    # Also write to a file to make it easy to open/inspect.
    out_dir = ROOT / "testOCR" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / f"db_dump_{pairs[0][0].hex[:8]}_{pairs[0][1].hex[:8]}.json"
    dump_path.write_text(_pretty(out), encoding="utf-8")

    print(_pretty(out))
    print(f"\n[db_dump_saved_to] {dump_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

