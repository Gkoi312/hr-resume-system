# app/api/v1/__init__.py
"""API v1 router: aggregates jobs, resumes, candidates, matching, tasks."""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, candidates, jobs, matching, resumes, tasks

api_router = APIRouter()

api_router.include_router(auth.router, prefix="")
api_router.include_router(jobs.router, prefix="")
api_router.include_router(resumes.router, prefix="")
api_router.include_router(candidates.router, prefix="")
api_router.include_router(matching.router, prefix="")
api_router.include_router(tasks.router, prefix="")
