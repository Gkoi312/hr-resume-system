# app/database/repository/user_repository.py
"""Repository for User (HR) CRUD."""

import uuid
from typing import Optional

from sqlalchemy import select

from app.database.models import UserModel
from app.database.session import get_session_context


class UserRepository:
    """Data access for users."""

    async def create(
        self,
        *,
        username: str,
        password_hash: str,
        occupation: Optional[str] = None,
        bio: Optional[str] = None,
        profile: Optional[str] = None,
        user_instruction: Optional[str] = None,
    ) -> UserModel:
        async with get_session_context() as session:
            user = UserModel(
                username=username,
                password_hash=password_hash,
                occupation=occupation,
                bio=bio,
                profile=profile,
                user_instruction=user_instruction,
            )
            session.add(user)
            await session.flush()
            await session.refresh(user)
            return user

    async def get_by_id(self, user_id: uuid.UUID) -> Optional[UserModel]:
        async with get_session_context() as session:
            stmt = select(UserModel).where(UserModel.id == user_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> Optional[UserModel]:
        async with get_session_context() as session:
            stmt = select(UserModel).where(UserModel.username == username)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()


user_repository = UserRepository()
