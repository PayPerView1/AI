import asyncio
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings


logger = logging.getLogger(__name__)
_MAX_GET_ATTEMPTS = 2


def _authorization_header(token: str) -> str | None:
    value = token.strip()
    if not value:
        return None
    return value if value.lower().startswith("bearer ") else f"Bearer {value}"


async def _request(
    method: str,
    path: str,
    token: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    authorization = _authorization_header(token)
    if not authorization:
        return {"error": "Authentication is required to access account data.", "code": "AUTH_REQUIRED"}
    if not settings.backend_url.strip():
        return {"error": "The platform API is not configured.", "code": "BACKEND_NOT_CONFIGURED"}

    parsed = urlparse(settings.backend_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        logger.error("Backend URL configuration is invalid")
        return {"error": "The platform API is not configured correctly.", "code": "BACKEND_CONFIG_INVALID"}

    url = f"{settings.backend_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Authorization": authorization, "Accept": "application/json"}
    timeout = httpx.Timeout(settings.backend_timeout_seconds, connect=min(5.0, settings.backend_timeout_seconds))
    attempts = _MAX_GET_ATTEMPTS if method == "GET" else 1

    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(method, url, headers=headers, params=params, json=body)
            if response.status_code >= 500 and attempt + 1 < attempts:
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return {"error": "The platform API returned an invalid response.", "code": "BACKEND_INVALID_RESPONSE"}
            return data
        except httpx.HTTPStatusError as exc:
            code = f"BACKEND_HTTP_{exc.response.status_code}"
            logger.warning("Platform API request failed (method=%s status=%s)", method, exc.response.status_code)
            return {"error": "The platform API could not complete this request.", "code": code}
        except httpx.TimeoutException:
            if attempt + 1 < attempts:
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            logger.warning("Platform API request timed out (method=%s)", method)
            return {"error": "The platform API request timed out.", "code": "BACKEND_TIMEOUT"}
        except httpx.RequestError as exc:
            if attempt + 1 < attempts:
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            logger.warning("Platform API request failed (method=%s type=%s)", method, type(exc).__name__)
            return {"error": "The platform API is temporarily unreachable.", "code": "BACKEND_UNAVAILABLE"}
        except ValueError:
            logger.warning("Platform API returned invalid JSON (method=%s)", method)
            return {"error": "The platform API returned an invalid response.", "code": "BACKEND_INVALID_RESPONSE"}

    return {"error": "The platform API is temporarily unreachable.", "code": "BACKEND_UNAVAILABLE"}


async def backend_get(
    path: str,
    token: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return await _request("GET", path, token, params=params)


async def backend_post(
    path: str,
    token: str,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return await _request("POST", path, token, body=body or {})
