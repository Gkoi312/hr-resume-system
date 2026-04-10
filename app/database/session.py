# app/database/session.py
"""
Async database session management for HR Resume Screening System.
Uses env DATABASE_URL (可配置 .env 或环境变量，.env 示例见项目根目录 .env.example)。
"""

import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

# Default async URL for local dev (override with DATABASE_URL env)
_DEFAULT_ASYNC_URL = "postgresql+asyncpg://hr_resume:hr_resume@localhost:5432/hr_resume"
_async_url = os.getenv("DATABASE_URL", _DEFAULT_ASYNC_URL)
if _async_url.startswith("postgresql://"):
    _async_url = _async_url.replace("postgresql://", "postgresql+asyncpg://", 1)

_is_sqlite = "sqlite" in _async_url
_engine_kw: dict = {"echo": os.getenv("DB_ECHO", "false").lower() == "true"}
if not _is_sqlite:
    _engine_kw["pool_size"] = int(os.getenv("DB_POOL_SIZE", "5"))
    _engine_kw["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    _engine_kw["pool_timeout"] = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    _engine_kw["pool_recycle"] = int(os.getenv("DB_POOL_RECYCLE", "1800"))

_engine = create_async_engine(_async_url, **_engine_kw)

async_session_factory = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Create tables if they do not exist."""
    from app.database.models import Base

    try:
        async with _engine.begin() as conn:
            if os.getenv("VECTOR_STORE_BACKEND", "db").strip().lower() == "pgvector" and not _is_sqlite:
                # Ensure pgvector extension exists before creating tables with `vector` type.
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)
    except (ProgrammingError, OperationalError) as e:
        msg = str(getattr(e, "orig", e))
        if "already exists" not in msg and "already exists" not in str(e):
            logger.error("Database schema creation failed: %s", msg, exc_info=True)
            raise
        logger.warning("Some schema objects already exist, skipping: %s", msg)
    logger.info("Database tables initialized.")


async def reset_db() -> None:
    """Drop all application tables and recreate schema. Destructive (all data lost)."""
    from app.database.models import Base

    async with _engine.begin() as conn:
        if _is_sqlite:
            await conn.run_sync(Base.metadata.drop_all)
        else:
            # Wipe everything in public (includes tables not registered on Base.metadata,
            # e.g. legacy resume_chunk_embeddings), then recreate.
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            await conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
        if os.getenv("VECTOR_STORE_BACKEND", "db").strip().lower() == "pgvector" and not _is_sqlite:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
    logger.warning("Database reset complete: schema wiped + create_all.")


async def close_db() -> None:
    """Close database connections."""
    await _engine.dispose()
    logger.info("Database connections closed.")


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency: yield async session, commit on success, rollback on error."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for manual session handling (repositories)."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
