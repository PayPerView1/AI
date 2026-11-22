from typing import Any, Dict, Optional

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.backend_client import backend_get


settings = get_settings()
security = HTTPBearer(auto_error=False)


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """Verify a PayPerView backend access token and resolve its user and role."""
    if not settings.jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        )

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "role"]},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token.",
        ) from None

    user_id = payload.get("sub") or payload.get("user_id") or payload.get("id")
    role_claim = payload.get("role")
    role = role_claim.upper() if isinstance(role_claim, str) else ""
    if not user_id or role not in {"CLIPPER", "BRAND"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token claims.",
        )

    return {
        "user_id": str(user_id),
        "role": role,
        "email": payload.get("email", ""),
        "full_name": payload.get("name") or payload.get("full_name") or "Platform User",
    }


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    decoded = decode_jwt_token(credentials.credentials)
    session = await backend_get("/api/v1/profile", credentials.credentials)
    if session.get("code") in ("BACKEND_HTTP_401", "BACKEND_HTTP_404"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if session.get("error"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The platform could not validate this session. Please retry shortly.",
        )

    user = session.get("user")
    backend_user_id = user.get("_id") or user.get("id") if isinstance(user, dict) else None
    backend_role = user.get("role") if isinstance(user, dict) else None
    if str(backend_user_id or "") != decoded["user_id"] or str(backend_role or "").upper() != decoded["role"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    decoded["raw_token"] = credentials.credentials
    return decoded
