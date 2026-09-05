from typing import Dict, Any, Optional
import jwt
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import get_settings

settings = get_settings()
security = HTTPBearer(auto_error=False)


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT Bearer token.
    Extracts 'sub' as user_id and 'role' ('CLIPPER' | 'BRAND').
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        user_id = payload.get("sub") or payload.get("user_id") or payload.get("id")
        role = payload.get("role", "CLIPPER").upper()

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing user identifier ('sub').",
            )

        if role not in ("CLIPPER", "BRAND"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: role must be 'CLIPPER' or 'BRAND'.",
            )

        return {
            "user_id": str(user_id),
            "role": role,
            "email": payload.get("email", ""),
            "full_name": payload.get("name") or payload.get("full_name") or "Platform User",
            "raw_payload": payload,
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT token has expired. Please authenticate again.",
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid JWT token: {str(e)}",
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Dict[str, Any]:
    """
    FastAPI security dependency to enforce JWT Bearer authentication.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or invalid format. Required: Bearer <JWT_TOKEN>",
        )

    return decode_jwt_token(credentials.credentials)


def create_access_token(
    user_id: str,
    role: str,
    email: str = "user@example.com",
    name: str = "Test User",
    expires_delta_days: int = 7,
) -> str:
    """
    Generate a signed JWT token for testing.
    """
    expire = datetime.now(timezone.utc) + timedelta(days=expires_delta_days)
    payload = {
        "sub": user_id,
        "role": role.upper(),
        "email": email,
        "name": name,
        "exp": int(expire.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
