# app/schemas/user.py
"""Pydantic schemas for User."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class UserBase(BaseModel):
    """Base user schema."""

    username: str
    occupation: Optional[str] = None
    bio: Optional[str] = None
    profile: Optional[str] = None
    user_instruction: Optional[str] = None


class UserCreate(UserBase):
    """Schema for creating a user (auth layer sets password_hash)."""

    password: str


class UserUpdate(BaseModel):
    """Schema for partial user update."""

    occupation: Optional[str] = None
    bio: Optional[str] = None
    profile: Optional[str] = None
    user_instruction: Optional[str] = None


class UserInDB(UserBase):
    """Schema for user as stored (includes id and timestamps)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime


class UserResponse(UserInDB):
    """Schema for user in API response (no password)."""

    pass
