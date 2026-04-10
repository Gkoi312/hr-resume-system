# app/core/config.py
"""Runtime configuration from environment (see .env.example)."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# JWT for HR auth (change JWT_SECRET in production)
JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-change-me-in-production")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = _get_int("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24)
