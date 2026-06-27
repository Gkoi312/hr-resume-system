# tests/conftest.py
"""Pytest fixtures: sync TestClient with app lifespan."""

import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# 测试默认用内存 SQLite，无需启动 PostgreSQL；若要测真实库可设 TEST_USE_POSTGRES=1
if os.getenv("TEST_USE_POSTGRES") != "1":
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("JWT_SECRET", "pytest-jwt-secret-do-not-use-in-production")

from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app, base_url="http://test") as c:
        yield c
