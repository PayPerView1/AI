"""
backend_client.py
─────────────────
Centralized async HTTP client for AI Service → Express Backend communication.
Base URL: https://payperview-platform.onrender.com

All tool functions call backend_get() / backend_post() which automatically:
  - Inject Authorization: Bearer <token> header
  - Set Content-Type: application/json
  - Apply a 10-second timeout (Render cold starts can be slow)
  - Return parsed JSON dict, or an error dict on failure
"""

import httpx
from typing import Any, Dict, Optional
from app.core.config import get_settings

settings = get_settings()

# ─── Shared timeout config ────────────────────────────────────────────────────
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


async def backend_get(
    path: str,
    token: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform an authenticated GET request to the Express backend.

    Args:
        path:   API path, e.g. "/api/v1/campaigns"
        token:  JWT Bearer token of the currently authenticated user
        params: Optional query parameters dict

    Returns:
        Parsed JSON response dict, or {"error": ...} on failure.
    """
    url = f"{settings.backend_url}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return {
            "error": f"Backend returned {e.response.status_code}",
            "detail": e.response.text,
        }
    except httpx.RequestError as e:
        return {"error": f"Network error reaching backend: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def backend_post(
    path: str,
    token: str,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform an authenticated POST request to the Express backend.

    Args:
        path:  API path, e.g. "/api/v1/campaigns/bulk-delete"
        token: JWT Bearer token of the currently authenticated user
        body:  Optional JSON request body dict

    Returns:
        Parsed JSON response dict, or {"error": ...} on failure.
    """
    url = f"{settings.backend_url}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(url, headers=headers, json=body or {})
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return {
            "error": f"Backend returned {e.response.status_code}",
            "detail": e.response.text,
        }
    except httpx.RequestError as e:
        return {"error": f"Network error reaching backend: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
