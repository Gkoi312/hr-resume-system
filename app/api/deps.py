# app/api/deps.py
"""FastAPI dependencies: current HR user from JWT."""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token, parse_uuid_sub
from app.database.models import UserModel
from app.services.user_service import user_service

_bearer = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(_bearer),
    ],
) -> Optional[UserModel]:
    if credentials is None or not credentials.credentials:
        return None
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = parse_uuid_sub(payload)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await user_service.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_user(
    user: Annotated[Optional[UserModel], Depends(get_current_user_optional)],
) -> UserModel:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
