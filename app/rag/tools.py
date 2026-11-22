"""Authenticated, role-scoped tools backed by the PayPerView Express API."""

import logging
import re
from typing import Any, Dict

from app.core.backend_client import backend_get, backend_post


logger = logging.getLogger(__name__)
_OBJECT_ID = re.compile(r"^[0-9a-fA-F]{24}$")
_DISPLAY_STATUSES = {
    "PROCESSING",
    "AI_REJECTED",
    "MANUAL_REVIEW",
    "AWAITING_ADVERTISER",
    "APPROVED",
    "REJECTED",
    "CHANGES_REQUESTED",
    "WITHDRAWN",
}
_CAMPAIGN_STATUSES = {
    "ALL",
    "DRAFT",
    "PENDING_REVIEW",
    "ACTIVE",
    "REJECTED",
    "MANUAL_REVIEW",
    "COMPLETED",
    "CANCELLED",
    "ARCHIVED",
    "EXPIRED",
    "PAUSED",
    "MANUALLY_PAUSED",
}
_CAMPAIGN_SUBMISSION_STATUSES = {
    "ALL",
    "AWAITING_ADVERTISER",
    "APPROVED",
    "REJECTED",
    "CHANGES_REQUESTED",
    "AI_REJECTED",
    "WITHDRAWN",
}


def _bounded_limit(args: Dict[str, Any], default: int = 5) -> int:
    value = args.get("limit", default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("limit must be an integer")
    return max(1, min(value, 10))


def _campaign_id(args: Dict[str, Any]) -> str:
    value = args.get("campaignId") or args.get("campaign_id")
    if not isinstance(value, str) or not _OBJECT_ID.fullmatch(value):
        raise ValueError("campaignId must be a valid campaign identifier")
    return value


async def _get_submissions(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page": 1, "perPage": _bounded_limit(args)}
    display_status = args.get("status")
    if display_status:
        normalized = str(display_status).upper()
        if normalized not in _DISPLAY_STATUSES:
            raise ValueError("status is not supported")
        params["displayStatus"] = normalized
    return await backend_get("/api/v1/submissions/my", token, params=params)


async def _get_eligible_campaigns(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page": 1, "perPage": _bounded_limit(args)}
    search = args.get("search")
    if search:
        params["search"] = str(search)[:80]
    return await backend_get("/api/v1/submissions/eligible-campaigns", token, params=params)


async def _get_brand_campaigns(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page": 1, "limit": _bounded_limit(args, default=5)}
    status = args.get("status")
    if status:
        normalized_status = str(status).strip().upper()
        if normalized_status not in _CAMPAIGN_STATUSES:
            raise ValueError("status is not supported")
        params["status"] = normalized_status
    return await backend_get("/api/v1/campaigns", token, params=params)


async def _get_campaign_details(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    campaign_id = _campaign_id(args)
    return await backend_get(f"/api/v1/campaigns/{campaign_id}", token)


async def _get_campaign_ai_review(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    campaign_id = _campaign_id(args)
    return await backend_get(f"/api/v1/campaigns/{campaign_id}/ai-review", token)


async def _get_brand_dashboard_stats(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return await backend_get("/api/v1/campaigns/statistics", token)


async def _get_campaign_submissions(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    campaign_id = _campaign_id(args)
    params: Dict[str, Any] = {"page": 1, "perPage": _bounded_limit(args)}
    status = args.get("status")
    if status:
        normalized_status = str(status).upper()
        if normalized_status not in _CAMPAIGN_SUBMISSION_STATUSES:
            raise ValueError("status is not supported")
        params["displayStatus"] = normalized_status
    return await backend_get(f"/api/v1/campaigns/{campaign_id}/submissions", token, params=params)


async def _get_campaign_categories(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return await backend_get("/api/v1/campaigns/categories", token)


async def _get_campaign_ai_insights(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    campaign_id = _campaign_id(args)
    return await backend_get(f"/api/v1/campaigns/{campaign_id}/ai-insights", token)


async def _get_campaign_report_summary(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    campaign_id = _campaign_id(args)
    return await backend_get(f"/api/v1/campaigns/{campaign_id}/report-summary", token)


async def _get_brand_budget_fraud_intelligence(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return await backend_post("/api/v1/wallet/intelligence", token, {})


async def _get_creator_earnings_summary(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return await backend_get("/api/v1/creator/earnings", token)


async def _get_sub_categories(user_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return await backend_get("/api/v1/campaigns/categories/sub-categories", token)


_COMMON_PROPERTIES = {
    "limit": {"type": "integer", "description": "Maximum number of records to return (1 to 10)."},
}
_CAMPAIGN_ID_PROPERTIES = {
    "campaignId": {"type": "string", "description": "The 24-character campaign ID."},
}


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_submissions": {
        "roles": {"CLIPPER"},
        "handler": _get_submissions,
        "definition": {
            "name": "get_submissions",
            "description": "Fetch the authenticated creator's real submissions from the platform.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": sorted(_DISPLAY_STATUSES)},
                    **_COMMON_PROPERTIES,
                },
                "additionalProperties": False,
            },
        },
    },
    "get_available_campaigns_for_clipper": {
        "roles": {"CLIPPER"},
        "handler": _get_eligible_campaigns,
        "definition": {
            "name": "get_available_campaigns_for_clipper",
            "description": "Fetch campaigns the authenticated creator is eligible to submit to.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search": {"type": "string", "description": "Optional campaign or advertiser name search."},
                    **_COMMON_PROPERTIES,
                },
                "additionalProperties": False,
            },
        },
    },
    "get_brand_campaigns": {
        "roles": {"BRAND"},
        "handler": _get_brand_campaigns,
        "definition": {
            "name": "get_brand_campaigns",
            "description": "Fetch campaigns owned by the authenticated brand account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": sorted(_CAMPAIGN_STATUSES),
                        "description": "Optional campaign status filter.",
                    },
                    **_COMMON_PROPERTIES,
                },
                "additionalProperties": False,
            },
        },
    },
    "get_campaign_full_details": {
        "roles": {"BRAND"},
        "handler": _get_campaign_details,
        "definition": {
            "name": "get_campaign_full_details",
            "description": "Fetch details for a campaign owned by the authenticated brand account.",
            "parameters": {
                "type": "object",
                "properties": _CAMPAIGN_ID_PROPERTIES,
                "required": ["campaignId"],
                "additionalProperties": False,
            },
        },
    },
    "get_campaign_ai_review": {
        "roles": {"BRAND"},
        "handler": _get_campaign_ai_review,
        "definition": {
            "name": "get_campaign_ai_review",
            "description": "Fetch the review status for a campaign owned by the authenticated brand account.",
            "parameters": {
                "type": "object",
                "properties": _CAMPAIGN_ID_PROPERTIES,
                "required": ["campaignId"],
                "additionalProperties": False,
            },
        },
    },
    "get_campaign_ai_insights": {
        "roles": {"BRAND"},
        "handler": _get_campaign_ai_insights,
        "definition": {
            "name": "get_campaign_ai_insights",
            "description": "Fetch historical performance forecasts and archived campaign patterns for a campaign owned by the authenticated brand account.",
            "parameters": {"type": "object", "properties": _CAMPAIGN_ID_PROPERTIES, "required": ["campaignId"], "additionalProperties": False},
        },
    },
    "get_campaign_report_summary": {
        "roles": {"BRAND"},
        "handler": _get_campaign_report_summary,
        "definition": {
            "name": "get_campaign_report_summary",
            "description": "Fetch aggregated confirmed report counts by reason and status for an owned campaign. Reporter identity and report details are not returned.",
            "parameters": {"type": "object", "properties": _CAMPAIGN_ID_PROPERTIES, "required": ["campaignId"], "additionalProperties": False},
        },
    },
    "get_brand_budget_fraud_intelligence": {
        "roles": {"BRAND"},
        "handler": _get_brand_budget_fraud_intelligence,
        "definition": {
            "name": "get_brand_budget_fraud_intelligence",
            "description": "Fetch advisory payment fraud, spending anomaly, daily budget, and refund analysis based on the authenticated brand's own account data. No payment is blocked or changed.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    "get_creator_earnings_summary": {
        "roles": {"CLIPPER"},
        "handler": _get_creator_earnings_summary,
        "definition": {
            "name": "get_creator_earnings_summary",
            "description": "Fetch only the authenticated creator's wallet summary and earnings totals.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    "get_brand_dashboard_stats": {
        "roles": {"BRAND"},
        "handler": _get_brand_dashboard_stats,
        "definition": {
            "name": "get_brand_dashboard_stats",
            "description": "Fetch actual campaign and budget statistics for the authenticated brand account.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    "get_submissions_for_campaign": {
        "roles": {"BRAND"},
        "handler": _get_campaign_submissions,
        "definition": {
            "name": "get_submissions_for_campaign",
            "description": "Fetch submissions on a campaign owned by the authenticated brand account.",
            "parameters": {
                "type": "object",
                "properties": {
                    **_CAMPAIGN_ID_PROPERTIES,
                    "status": {"type": "string", "enum": sorted(_CAMPAIGN_SUBMISSION_STATUSES)},
                    **_COMMON_PROPERTIES,
                },
                "required": ["campaignId"],
                "additionalProperties": False,
            },
        },
    },
    "get_campaign_categories": {
        "roles": {"CLIPPER", "BRAND"},
        "handler": _get_campaign_categories,
        "definition": {
            "name": "get_campaign_categories",
            "description": "Fetch the platform's campaign categories.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    "get_sub_categories": {
        "roles": {"CLIPPER", "BRAND"},
        "handler": _get_sub_categories,
        "definition": {
            "name": "get_sub_categories",
            "description": "Fetch campaign sub-categories.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
}


def tool_definitions_for_role(role: str) -> list[Dict[str, Any]]:
    normalized_role = role.upper()
    return [
        spec["definition"]
        for spec in TOOL_REGISTRY.values()
        if normalized_role in spec["roles"]
    ]


async def execute_tool(
    tool_name: str,
    user_id: str,
    args: Dict[str, Any],
    user_token: str = "",
    role: str = "",
) -> Dict[str, Any]:
    spec = TOOL_REGISTRY.get(tool_name)
    normalized_role = role.upper()
    if spec is None or normalized_role not in spec["roles"]:
        return {"error": "This account cannot use the requested tool.", "code": "TOOL_NOT_ALLOWED"}
    if not isinstance(args, dict):
        return {"error": "Tool arguments must be an object.", "code": "INVALID_TOOL_ARGUMENTS"}

    try:
        return await spec["handler"](user_id, user_token, args)
    except ValueError as exc:
        return {"error": str(exc), "code": "INVALID_TOOL_ARGUMENTS"}
    except Exception as exc:
        logger.warning("AI tool failed (tool=%s type=%s)", tool_name, type(exc).__name__)
        return {"error": "The requested platform data is temporarily unavailable.", "code": "TOOL_UNAVAILABLE"}
