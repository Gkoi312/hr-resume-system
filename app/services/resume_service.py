# app/services/resume_service.py
"""
Resume service: create resume record, get/list resumes, save parsed content,
bind candidate from resume (sync parsed data to candidate).
Depends: ResumeRepository, CandidateRepository; schemas.
"""

import uuid
from typing import Any, Dict, List, Optional

from app import statuses
from app.parsers.resume_parser.candidate_profile_builder import get_candidate_bind_for_resume
from app.database.repository.candidate_repository import candidate_repository
from app.database.repository.resume_repository import resume_repository
from app.schemas.resume import ResumeResponse
from app.schemas.task import TaskCreate
from app.services.task_service import task_service


class ResumeService:
    """Service for resume and candidate-from-resume operations."""

    async def create_resume_record(
        self,
        *,
        candidate_id: Optional[uuid.UUID] = None,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> ResumeResponse:
        """
        Create a resume record. If candidate_id is None, creates a placeholder
        candidate first and links the resume to it.
        """
        if candidate_id is None:
            candidate = await candidate_repository.create(
                name="",
                email=None,
                phone=None,
            )
            candidate_id = candidate.id
        resume = await resume_repository.create(
            candidate_id=candidate_id,
            file_path=file_path,
            file_name=file_name,
            parsed=None,
            status=statuses.RESUME_STATUS_UPLOADED,
        )
        return ResumeResponse.model_validate(resume)

    async def get_resume(self, resume_id: uuid.UUID) -> Optional[ResumeResponse]:
        """Get a resume by id."""
        resume = await resume_repository.get_by_id(resume_id)
        return ResumeResponse.model_validate(resume) if resume else None

    async def list_resumes(
        self,
        candidate_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ResumeResponse]:
        """List resumes for a candidate."""
        resumes = await resume_repository.list_by_candidate(
            candidate_id=candidate_id,
            limit=limit,
            offset=offset,
        )
        return [ResumeResponse.model_validate(r) for r in resumes]

    async def save_parsed_resume(
        self,
        resume_id: uuid.UUID,
        parsed: Dict[str, Any],
        status: str = statuses.RESUME_STATUS_PARSED,
    ) -> Optional[ResumeResponse]:
        """Save parsed content and status for a resume."""
        resume = await resume_repository.update(
            resume_id,
            parsed=parsed,
            status=status,
            error_message=None,
        )
        return ResumeResponse.model_validate(resume) if resume else None

    async def update_resume_status(
        self,
        resume_id: uuid.UUID,
        status: str,
    ) -> Optional[ResumeResponse]:
        """Update only the status for a resume."""
        resume = await resume_repository.update(
            resume_id,
            status=status,
        )
        return ResumeResponse.model_validate(resume) if resume else None

    async def mark_failed(
        self,
        resume_id: uuid.UUID,
        error_message: str,
    ) -> Optional[ResumeResponse]:
        """Mark resume as failed with an error message."""
        resume = await resume_repository.update(
            resume_id,
            status=statuses.RESUME_STATUS_FAILED,
            error_message=error_message,
        )
        return ResumeResponse.model_validate(resume) if resume else None

    async def retry_parse_resume(self, resume_id: uuid.UUID) -> Optional[ResumeResponse]:
        """Placeholder for retry logic; actual parsing requires access to file content."""
        # Reset state for re-processing:
        # - clear parsed content so downstream bind doesn't reuse stale parse
        # - clear error_message
        # - set status back to uploaded so upstream workflow can re-trigger parsing
        resume = await resume_repository.update(
            resume_id,
            status=statuses.RESUME_STATUS_UPLOADED,
            parsed=None,
            error_message=None,
        )
        return ResumeResponse.model_validate(resume) if resume else None

    # 从简历解析结果同步候选人信息，需要简历的 parsed 字段已经存在。
    # 用简历里的信息去补齐候选人画像，如果原本候选人不存在，直接返回简历信息，不做任何绑定和更新操作
    # 如果候选人存在，就用传来的解析过的简历去补齐候选人画像（原本的画像可以什么都没有）
    async def bind_candidate_from_resume(
        self, resume_id: uuid.UUID
    ) -> Optional[ResumeResponse]:
        """
        Sync resume parsed content to the linked candidate using Layer2
        ``candidate_bind`` (rebuilt from ``layer_1_extracted`` when missing).
        """
        resume = await resume_repository.get_by_id(resume_id) # 拿到简历
        if not resume or not resume.parsed:
            return None

        # Helper: treat None / "" / [] / {} as empty for merge decisions
        def _is_empty(v: Any) -> bool:
            return v is None or v == "" or v == [] or v == {}

        parsed = resume.parsed
        bind = get_candidate_bind_for_resume(parsed)
        candidate = await candidate_repository.get_by_id(resume.candidate_id)

        # If candidate record is missing for some reason, just return resume info.
        if not candidate:
            return await self.get_resume(resume_id)

        # Name
        parsed_name = (bind.get("name") or "").strip() if isinstance(bind.get("name"), str) else ""
        if not parsed_name and (bind.get("email") or bind.get("phone")):
            parsed_name = "未填写姓名"
        new_name = candidate.name
        if _is_empty(new_name):
            new_name = parsed_name or resume.candidate_id.hex[:8]

        # Simple scalar fields with "candidate first, parsed fallback"
        new_email = candidate.email
        if _is_empty(new_email):
            new_email = bind.get("email") or None

        new_phone = candidate.phone
        if _is_empty(new_phone):
            new_phone = bind.get("phone") or None

        # Complex list / structured fields
        parsed_education = bind.get("education")
        if not isinstance(parsed_education, list):
            parsed_education = None
        new_education = candidate.education
        if _is_empty(new_education):
            new_education = parsed_education

        parsed_work_exp = bind.get("work_experience")
        if not isinstance(parsed_work_exp, list):
            parsed_work_exp = None
        new_work_exp = candidate.work_experience
        if _is_empty(new_work_exp):
            new_work_exp = parsed_work_exp

        parsed_skills = bind.get("skills")
        if not isinstance(parsed_skills, list):
            parsed_skills = None
        # Resume bind: grounded skills / summary come from this parse (evidence-based).
        new_skills = parsed_skills if parsed_skills is not None else candidate.skills

        parsed_projects = bind.get("projects")
        if not isinstance(parsed_projects, list):
            parsed_projects = None
        new_projects = candidate.projects
        if _is_empty(new_projects):
            new_projects = parsed_projects

        parsed_years = bind.get("years_of_experience")
        if not isinstance(parsed_years, (int, float)):
            parsed_years = None
        new_years = candidate.years_of_experience
        if new_years is None:
            # years_of_experience: only treat None as empty, 0 是合法值
            new_years = int(parsed_years) if isinstance(parsed_years, (int, float)) else None

        parsed_direction_tags = bind.get("direction_tags")
        if not isinstance(parsed_direction_tags, list):
            parsed_direction_tags = None
        new_direction_tags = getattr(candidate, "direction_tags", None)
        if _is_empty(new_direction_tags):
            new_direction_tags = parsed_direction_tags

        await candidate_repository.update(
            resume.candidate_id,
            name=new_name,
            email=new_email,
            phone=new_phone,
            education=new_education,
            work_experience=new_work_exp,
            skills=new_skills,
            projects=new_projects,
            years_of_experience=new_years,
            direction_tags=new_direction_tags,
        )
        # V1.2+: this project MVP uses job<->candidate matching only.
        # Resume is only a data source to fill candidate fields; therefore enqueue
        # candidate profile index build (general vector) after candidate update.
        await task_service.create_task(
            TaskCreate(
                task_type="candidate_profile_index_build",
                resource_type="candidate",
                resource_id=resume.candidate_id,
                payload={"candidate_id": str(resume.candidate_id)},
            )
        )
        return await self.get_resume(resume_id)


resume_service = ResumeService()
