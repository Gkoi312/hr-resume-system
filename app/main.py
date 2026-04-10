# app/main.py
"""
HR Resume Screening System - FastAPI application.
Mounts API v1 at /api/v1; init_db on startup, close_db on shutdown.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1 import api_router
from app.database.session import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init_db; Shutdown: close_db."""
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="HR Resume Screening System",
    description="企业级 AI 简历筛选与候选人分析 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")
