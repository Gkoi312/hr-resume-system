# app/api/v1/endpoints/auth.py
"""
HR auth API:
  POST   /api/v1/auth/register  - create HR account
  POST   /api/v1/auth/login    - JWT access token + user profile
"""

from fastapi import APIRouter, HTTPException, status

from app.core.security import create_access_token
from app.schemas.auth import LoginBody, RegisterBody, TokenResponse
from app.schemas.user import UserResponse
from app.services.user_service import user_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(body: RegisterBody):
    try:
        user = await user_service.register(
            username=body.username.strip(),
            password=body.password,
            occupation=body.occupation,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    token = create_access_token(subject=str(user.id), extra_claims={"username": user.username})
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginBody):
    user = await user_service.authenticate(body.username.strip(), body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(subject=str(user.id), extra_claims={"username": user.username})
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )
