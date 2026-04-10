# app/api/v1/endpoints/resumes.py
"""
Resumes API:
  POST   /api/v1/resumes/upload   - upload file (PDF/DOCX/TXT) -> parse & bind, or JSON create-only
  GET    /api/v1/resumes/{id}     - get resume
  GET    /api/v1/resumes          - list resumes by candidate_id
  POST   /api/v1/resumes/{id}/parsed   - save parsed content
  POST   /api/v1/resumes/{id}/bind     - bind candidate from resume parsed data

  对外：给前端/调用方暴露 HTTP 接口。
  对内：
    使用 resume_service 处理数据库操作和业务逻辑（创建记录、更新状态、保存解析内容、绑定候选人、重试解析等）。
    使用 parse_resume_document：TXT/DOCX 走文本抽取 + 规则或文本 LLM；PDF/图片走视觉 LLM（可选）或 Paddle PP-Structure + 规则 Layer1。
    用 statuses 中预定义的常量来维护简历状态机。
"""

import uuid
from typing import Any, Dict, Optional

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app import statuses
from app.parsers.resume_parser.resume_llm_layer1.pipeline import parse_resume_document
from app.schemas.resume import ResumeResponse
from app.services.resume_service import resume_service

router = APIRouter(prefix="/resumes", tags=["resumes"])


class ResumeUploadBody(BaseModel):
    """Request body for JSON upload (create record only, no parse)."""

    file_name: Optional[str] = "resume.pdf"
    file_path: Optional[str] = None
    candidate_id: Optional[uuid.UUID] = None


# 上传简历并自动解析绑定候选人
@router.post("/upload", response_model=ResumeResponse)
async def upload_resume(
    file: Optional[UploadFile] = File(None),
    file_name: str = Form("resume.pdf"),
    candidate_id: Optional[uuid.UUID] = Form(None),
):
    """
    Two modes (multipart/form-data):
    1) With file: upload PDF/DOCX/TXT -> create record(status=uploaded) -> extracting -> parsed -> candidate_bound / failed.
    2) Without file: create resume record only (file_name, candidate_id from form).
    """
    if file and file.filename:
        try:
            content = await file.read()
            fn = file.filename or file_name

            # Persist upload bytes to a temp path so Resume.file_path is not NULL.
            tmp_path: str | None = None
            if content:
                upload_dir = Path(tempfile.gettempdir()) / "hr_resume_uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                tmp_file = f"{uuid.uuid4().hex}_{Path(fn).name}"
                tmp_path_obj = upload_dir / tmp_file
                tmp_path_obj.write_bytes(content)
                tmp_path = str(tmp_path_obj)

            resume = await resume_service.create_resume_record(
                candidate_id=candidate_id,
                file_path=tmp_path,
                file_name=fn,
            )
            await resume_service.update_resume_status(
                resume.id, statuses.RESUME_STATUS_EXTRACTING
            )

            if content:
                parsed = await parse_resume_document(
                    content,
                    fn,
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
            return await resume_service.get_resume(resume.id) or resume
        except Exception as exc:  # noqa: BLE001
            # Any failure in extract/parse/bind marks resume as failed and records reason.
            await resume_service.mark_failed(resume.id, str(exc))
            raise HTTPException(
                status_code=500,
                detail="Failed to process resume file",
            ) from exc
    resume = await resume_service.create_resume_record(
        candidate_id=candidate_id,
        file_path=None,
        file_name=file_name,
    )
    return resume


# 获取单个简历
@router.get("/{resume_id}", response_model=ResumeResponse)
async def get_resume(resume_id: uuid.UUID):
    """Get a resume by id."""
    resume = await resume_service.get_resume(resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    return resume


# 按候选人列出简历，必须提供 candidate_id 作为查询参数。
@router.get("", response_model=list[ResumeResponse])
async def list_resumes(
    candidate_id: uuid.UUID = Query(..., description="Filter by candidate"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List resumes for a candidate."""
    return await resume_service.list_resumes(
        candidate_id=candidate_id,
        limit=limit,
        offset=offset,
    )


# 保存解析结果（手动或者外部解析用）。
# 适用于已经在别处拿到解析后的结构化数据（可能是后台任务或第三方服务），需要回写到系统中。
@router.post("/{resume_id}/parsed", response_model=ResumeResponse)
async def save_parsed_resume(
    resume_id: uuid.UUID,
    parsed: Dict[str, Any],
    status: str = statuses.RESUME_STATUS_PARSED,
):
    """Save parsed content for a resume."""
    resume = await resume_service.save_parsed_resume(
        resume_id=resume_id,
        parsed=parsed,
        status=status,
    )
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    return resume


# 从简历解析结果同步候选人信息，需要简历的 parsed 字段已经存在。
@router.post("/{resume_id}/bind", response_model=ResumeResponse)
async def bind_candidate_from_resume(resume_id: uuid.UUID):
    """Sync resume parsed content to the linked candidate."""
    resume = await resume_service.bind_candidate_from_resume(resume_id)
    if not resume:
        raise HTTPException(
            status_code=404,
            detail="Resume not found or has no parsed content",
        )
    await resume_service.update_resume_status(
        resume_id, statuses.RESUME_STATUS_CANDIDATE_BOUND
    )
    return await resume_service.get_resume(resume_id) or resume


# 重新解析
@router.post("/{resume_id}/retry-parse", response_model=ResumeResponse)
async def retry_parse_resume(resume_id: uuid.UUID):
    """
    Retry parsing a resume.

    当前实现为轻量级重试：仅将状态重置为 uploaded 并清空 error_message，
    方便上层重新触发解析流程（例如重新上传或后台任务再次处理）。
    """
    resume = await resume_service.retry_parse_resume(resume_id)
    if not resume:
        raise HTTPException(
            status_code=404,
            detail="Resume not found",
        )
    return resume
