# app/schemas/auth.py
"""Request/response schemas for HR authentication."""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.user import UserResponse


class RegisterBody(BaseModel):
    username: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    occupation: Optional[str] = Field(None, max_length=100)


class LoginBody(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
