"""
tools.py
────────
Function-calling tools for the PayPerView AI chatbot.

BRAND tools now make real async HTTP calls to:
    https://payperview-platform.onrender.com

CLIPPER tools (earnings, submissions) remain as mock data until
the Express backend exposes dedicated Clipper endpoints.

All async tools accept `user_token` and forward it as
Authorization: Bearer <token> to the backend.
"""

from typing import Dict, Any, List, Optional
import asyncio

from app.core.backend_client import backend_get, backend_post


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CLIPPER TOOLS  (still mock — no Sprint 2 Express endpoints yet)
# ═══════════════════════════════════════════════════════════════════════════════

def get_weekly_earnings(user_id: str, week: str = "current") -> Dict[str, Any]:
    """Fetch weekly earnings breakdown for a creator (Clipper) — mock until backend exposes endpoint."""
    return {
        "user_id": user_id,
        "week": week,
        "total_earnings": 240.00,
        "currency": "USD",
        "approved_submissions": 12,
        "pending_submissions": 2,
        "rejected_submissions": 1,
        "breakdown": [
            {"campaign": "Tech Review Reels", "amount": 100.00, "date": "2026-09-01"},
            {"campaign": "Fitness App Promo", "amount": 140.00, "date": "2026-09-03"},
        ],
    }


def get_monthly_earnings(user_id: str, month: str = "September", year: int = 2026) -> Dict[str, Any]:
    """Fetch monthly earnings breakdown for a creator (Clipper) — mock."""
    return {
        "user_id": user_id,
        "month": month,
        "year": year,
        "total_earnings": 960.00,
        "currency": "USD",
        "total_approved": 48,
        "average_per_submission": 20.00,
    }


def get_submissions(user_id: str, status: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
    """Query creator submissions filtered by status — mock."""
    all_submissions = [
        {"id": "sub_101", "campaignName": "Gaming Laptop Shorts",  "status": "approved", "amount": 80.00,  "submittedAt": "2026-09-04"},
        {"id": "sub_102", "campaignName": "Energy Drink Reel",     "status": "approved", "amount": 60.00,  "submittedAt": "2026-09-03"},
        {"id": "sub_103", "campaignName": "Fashion Haul Clips",    "status": "pending",  "amount": 45.00,  "submittedAt": "2026-09-05"},
        {"id": "sub_104", "campaignName": "Crypto Wallet App",     "status": "rejected", "amount": 0.00,   "submittedAt": "2026-09-02", "reason": "Low video quality"},
    ]
    if status:
        filtered = [s for s in all_submissions if s["status"].lower() == status.lower()]
    else:
        filtered = all_submissions
    return {"user_id": user_id, "count": len(filtered[:limit]), "submissions": filtered[:limit]}


def get_available_campaigns_for_clipper(category: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
    """List active campaigns open for submission (CLIPPERs) — mock."""
    campaigns = [
        {
            "id": "64f1a2b3c4d5e6f7a8b9c0d1",
            "title": "حملة رمضان 2026",
            "category": "CLIPPING",
            "payout_per_1k_views": 10.00,
            "targetCountries": ["SAU", "EGY", "ARE"],
            "brief": {"mainIdea": "تسليط الضوء على منتجنا الجديد", "tone": "ودي وحيوي", "keywords": ["رمضان", "عروض"]},
        },
        {
            "id": "64f1a2b3c4d5e6f7a8b9c0d2",
            "title": "Protein Powder Unboxing",
            "category": "UGC",
            "payout_per_1k_views": 12.00,
            "targetCountries": ["USA", "GBR"],
            "brief": {"mainIdea": "Show yourself using the product", "tone": "Energetic", "keywords": ["fitness", "health"]},
        },
    ]
    if category:
        campaigns = [c for c in campaigns if category.lower() in c["category"].lower()]
    return {"count": len(campaigns[:limit]), "campaigns": campaigns[:limit]}


def get_profile_summary(user_id: str) -> Dict[str, Any]:
    """Summary of creator account status and rating — mock."""
    return {
        "user_id": user_id,
        "role": "CLIPPER",
        "rank": "Gold Creator",
        "rating": 4.9,
        "total_lifetime_earnings": 3450.00,
        "completed_campaigns": 35,
        "approval_rate": "96%",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BRAND TOOLS  (real HTTP calls to Express backend on Render)
# ═══════════════════════════════════════════════════════════════════════════════

async def get_brand_campaigns(
    user_id: str,
    user_token: str,
    status: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Fetch list of campaigns for the authenticated brand.
    -> GET /api/v1/campaigns"""
    params: Dict[str, Any] = {"limit": limit}
    if status and status != "ALL":
        params["status"] = status
    if category:
        params["category"] = category
    return await backend_get("/api/v1/campaigns", user_token, params=params)


async def get_campaign_full_details(
    user_id: str,
    user_token: str,
    campaign_id: str,
) -> Dict[str, Any]:
    """Fetch full details of a specific campaign by ID.
    -> GET /api/v1/campaigns/:campaignId"""
    return await backend_get(f"/api/v1/campaigns/{campaign_id}", user_token)


async def get_campaign_ai_review(
    user_id: str,
    user_token: str,
    campaign_id: str,
) -> Dict[str, Any]:
    """Fetch AI review results for a specific campaign.
    -> GET /api/v1/campaigns/:campaignId/ai-review"""
    return await backend_get(f"/api/v1/campaigns/{campaign_id}/ai-review", user_token)


async def get_brand_dashboard_stats(
    user_id: str,
    user_token: str,
) -> Dict[str, Any]:
    """Fetch dashboard statistics for the brand.
    -> GET /api/v1/campaigns/statistics"""
    return await backend_get("/api/v1/campaigns/statistics", user_token)


async def get_campaign_categories(user_token: str) -> Dict[str, Any]:
    """Fetch all available campaign categories.
    -> GET /api/v1/campaigns/categories (Public)"""
    return await backend_get("/api/v1/campaigns/categories", user_token)


async def get_sub_categories(user_token: str) -> Dict[str, Any]:
    """Fetch sub-categories available for MIXED campaigns.
    -> GET /api/v1/campaigns/categories/sub-categories"""
    return await backend_get("/api/v1/campaigns/categories/sub-categories", user_token)


async def get_campaign_stats(
    user_id: str,
    user_token: str,
    campaign_id: str,
) -> Dict[str, Any]:
    """Fetch performance metrics for a specific brand campaign.
    -> GET /api/v1/campaigns/:campaignId (extracts stats field)"""
    result = await backend_get(f"/api/v1/campaigns/{campaign_id}", user_token)
    campaign = result.get("data", {}).get("campaign", result)
    return {
        "user_id": user_id,
        "campaign_id": campaign_id,
        "title": campaign.get("name", ""),
        "status": campaign.get("status", ""),
        "stats": campaign.get("stats", {}),
        "totalBudget": campaign.get("totalBudget"),
        "remainingBudget": campaign.get("remainingBudget"),
    }


async def list_campaigns(
    user_id: str,
    user_token: str,
    status: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """List campaigns belonging to the brand advertiser.
    -> GET /api/v1/campaigns"""
    params: Dict[str, Any] = {"limit": limit}
    if status:
        params["status"] = status
    return await backend_get("/api/v1/campaigns", user_token, params=params)


async def get_budget_summary(user_id: str, user_token: str) -> Dict[str, Any]:
    """Query advertiser account balances via dashboard statistics.
    -> GET /api/v1/campaigns/statistics"""
    result = await backend_get("/api/v1/campaigns/statistics", user_token)
    stats = result.get("data", {}).get("statistics", {})
    return {
        "user_id": user_id,
        "total_budget_spent": stats.get("totalBudgetSpent", 0),
        "average_cpm": stats.get("averageCpm", 0),
        "active_campaigns": stats.get("activeCampaigns", 0),
        "currency": "USD",
    }


async def get_account_summary(user_id: str, user_token: str) -> Dict[str, Any]:
    """Summary of advertiser brand account stats.
    -> GET /api/v1/campaigns/statistics"""
    result = await backend_get("/api/v1/campaigns/statistics", user_token)
    stats = result.get("data", {}).get("statistics", {})
    return {
        "user_id": user_id,
        "role": "BRAND",
        "active_campaigns_count": stats.get("activeCampaigns", 0),
        "total_campaigns": stats.get("totalCampaigns", 0),
        "total_budget_spent": stats.get("totalBudgetSpent", 0),
    }


async def get_submissions_for_campaign(
    campaign_id: str,
    user_token: str,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Query creator submissions on a brand campaign.
    -> GET /api/v1/campaigns/:campaignId"""
    return await backend_get(f"/api/v1/campaigns/{campaign_id}", user_token)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ASYNC TOOL DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

async def execute_tool(
    tool_name: str,
    user_id: str,
    args: Dict[str, Any],
    user_token: str = "",
) -> Dict[str, Any]:
    """
    Async tool dispatcher. Executes tool functions by name.
    user_token is forwarded to all async BRAND tool calls.
    """
    campaign_id = args.get("campaignId") or args.get("campaign_id", "")

    # ── Sync CLIPPER tools (mock data) ─────────────────────────────────────────
    sync_tool_map = {
        "get_weekly_earnings":                 lambda: get_weekly_earnings(user_id, args.get("week", "current")),
        "get_monthly_earnings":                lambda: get_monthly_earnings(user_id, args.get("month", "September"), args.get("year", 2026)),
        "get_submissions":                     lambda: get_submissions(user_id, args.get("status"), args.get("limit", 5)),
        "get_available_campaigns_for_clipper": lambda: get_available_campaigns_for_clipper(args.get("category"), args.get("limit", 5)),
        "get_campaign_details":                lambda: {"campaign_id": campaign_id, "note": "Use get_campaign_full_details for BRAND campaigns."},
        "get_profile_summary":                 lambda: get_profile_summary(user_id),
    }

    if tool_name in sync_tool_map:
        try:
            return sync_tool_map[tool_name]()
        except Exception as e:
            return {"error": f"Error in sync tool '{tool_name}': {str(e)}"}

    # ── Async BRAND tools (real HTTP to Express backend) ──────────────────────
    async_tool_map = {
        "get_brand_campaigns":          lambda: get_brand_campaigns(user_id, user_token, args.get("status"), args.get("category"), args.get("limit", 20)),
        "get_campaign_full_details":    lambda: get_campaign_full_details(user_id, user_token, campaign_id),
        "get_campaign_ai_review":       lambda: get_campaign_ai_review(user_id, user_token, campaign_id),
        "get_brand_dashboard_stats":    lambda: get_brand_dashboard_stats(user_id, user_token),
        "get_campaign_stats":           lambda: get_campaign_stats(user_id, user_token, campaign_id),
        "list_campaigns":               lambda: list_campaigns(user_id, user_token, args.get("status"), args.get("limit", 5)),
        "get_budget_summary":           lambda: get_budget_summary(user_id, user_token),
        "get_account_summary":          lambda: get_account_summary(user_id, user_token),
        "get_submissions_for_campaign": lambda: get_submissions_for_campaign(campaign_id, user_token, args.get("status")),
        "get_campaign_categories":      lambda: get_campaign_categories(user_token),
        "get_sub_categories":           lambda: get_sub_categories(user_token),
    }

    if tool_name in async_tool_map:
        try:
            return await async_tool_map[tool_name]()
        except Exception as e:
            return {"error": f"Error in async tool '{tool_name}': {str(e)}"}

    return {"error": f"Unknown tool name: '{tool_name}'"}
