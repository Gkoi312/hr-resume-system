"""Task handler for resume_upload tasks."""

import uuid
from pathlib import Path
from typing import Optional

from app import statuses
from app.services.resume_service import resume_service
from app.services.task_service import task_service
from app.parsers.resume_parser.resume_llm_layer1.pipeline import parse_resume_document


async def handle_resume_upload(task) -> None:
    """
    Execute a resume_upload task.

    Expects payload:
      {
        "file_path": "<path on disk>",
        "original_name": "xxx.pdf",
        "candidate_id": "<uuid>|null"
      }
    """
    payload = task.payload or {}
    file_path_str = payload.get("file_path")
    original_name = payload.get("original_name") or "resume.pdf"
    candidate_id_str: Optional[str] = payload.get("candidate_id")

    if not file_path_str:
        raise ValueError("resume_upload task missing file_path in payload")

    path = Path(file_path_str)
    if not path.is_file():
        raise FileNotFoundError(f"resume file not found: {file_path_str}")

    candidate_id = uuid.UUID(candidate_id_str) if candidate_id_str else None

    try:
        await task_service.mark_running(task.id)

        resume = await resume_service.create_resume_record(
            candidate_id=candidate_id,
            file_path=str(path),
            file_name=original_name,
        )
        await resume_service.update_resume_status(
            resume.id, statuses.RESUME_STATUS_EXTRACTING
        )

        content = path.read_bytes()
        if content:
            parsed = await parse_resume_document(
                content,
                original_name,
                document_id=str(resume.id),
                candidate_id=str(resume.candidate_id),
            )
            await resume_service.save_parsed_resume(
                resume.id,
                parsed,
                status=statuses.RESUME_STATUS_PARSED,
            )
            await resume_service.bind_candidate_from_resume(resume.id)
            await resume_service.update_resume_status(
                resume.id, statuses.RESUME_STATUS_CANDIDATE_BOUND
            )

        await task_service.mark_completed(
            task.id,
            result_summary={"resume_id": str(resume.id)},
        )
    except Exception as exc:  # noqa: BLE001
        if "resume" in locals() and resume:
            await resume_service.mark_failed(resume.id, str(exc))
        await task_service.mark_failed(task.id, str(exc))
        raise

