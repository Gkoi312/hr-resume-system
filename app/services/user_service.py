# app/services/user_service.py
"""HR user registration and authentication."""

import uuid
from typing import Optional

from app.core.security import hash_password, verify_password
from app.database.models import UserModel
from app.database.repository.user_repository import user_repository


class UserService:
    async def register(
        self,
        *,
        username: str,
        password: str,
        occupation: Optional[str] = None,
        bio: Optional[str] = None,
        profile: Optional[str] = None,
        user_instruction: Optional[str] = None,
    ) -> UserModel:
        existing = await user_repository.get_by_username(username)
        if existing:
            raise ValueError("username already registered")
        return await user_repository.create(
            username=username,
            password_hash=hash_password(password),
            occupation=occupation,
            bio=bio,
            profile=profile,
            user_instruction=user_instruction,
        )

    async def authenticate(self, username: str, password: str) -> Optional[UserModel]:
        user = await user_repository.get_by_username(username)
        if not user:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    async def get_by_id(self, user_id: uuid.UUID) -> Optional[UserModel]:
        return await user_repository.get_by_id(user_id)


user_service = UserService()
