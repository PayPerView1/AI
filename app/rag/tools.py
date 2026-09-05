from typing import Dict, Any, List, Optional
import json


# ─── 1. CLIPPER TOOLS ────────────────────────────────────────────────────────

def get_weekly_earnings(user_id: str, week: str = "current") -> Dict[str, Any]:
    """Fetch weekly earnings breakdown for a creator (Clipper)."""
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
        ]
    }


def get_monthly_earnings(user_id: str, month: str = "September", year: int = 2026) -> Dict[str, Any]:
    """Fetch monthly earnings breakdown for a creator (Clipper)."""
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
    """Query creator submissions filtered by status (pending, approved, rejected)."""
    all_submissions = [
        {"id": "sub_101", "campaignName": "Gaming Laptop Shorts", "status": "approved", "amount": 80.00, "submittedAt": "2026-09-04"},
        {"id": "sub_102", "campaignName": "Energy Drink Reel", "status": "approved", "amount": 60.00, "submittedAt": "2026-09-03"},
        {"id": "sub_103", "campaignName": "Fashion Haul Clips", "status": "pending", "amount": 45.00, "submittedAt": "2026-09-05"},
        {"id": "sub_104", "campaignName": "Crypto Wallet App", "status": "rejected", "amount": 0.00, "submittedAt": "2026-09-02", "reason": "Low video quality"},
    ]
    if status:
        filtered = [s for s in all_submissions if s["status"].lower() == status.lower()]
    else:
        filtered = all_submissions
    return {"user_id": user_id, "count": len(filtered[:limit]), "submissions": filtered[:limit]}


def get_available_campaigns(category: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
    """List active campaigns open for submission."""
    campaigns = [
        {"id": "camp_201", "title": "AI SaaS Platform Demo", "category": "Tech", "payout_per_1k_views": 15.00, "max_payout": 300.00, "status": "active"},
        {"id": "camp_202", "title": "Protein Powder Unboxing", "category": "Fitness", "payout_per_1k_views": 12.00, "max_payout": 250.00, "status": "active"},
        {"id": "camp_203", "title": "Mobile RPG Game Trailer", "category": "Gaming", "payout_per_1k_views": 18.00, "max_payout": 500.00, "status": "active"},
    ]
    if category:
        campaigns = [c for c in campaigns if category.lower() in c["category"].lower()]
    return {"count": len(campaigns[:limit]), "campaigns": campaigns[:limit]}


def get_campaign_details(campaign_id: str) -> Dict[str, Any]:
    """View specific campaign requirements and guidelines."""
    return {
        "campaign_id": campaign_id,
        "title": "AI SaaS Platform Demo",
        "description": "Create a 30-60 second vertical video (TikTok/Reels/Shorts) highlighting our AI assistant tool.",
        "requirements": [
            "Must show the user interface on screen",
            "Include call to action in caption: 'Link in bio'",
            "No copyright music background",
        ],
        "payout_rate": "$15 per 1,000 verified views",
        "budget_remaining": "$1,200 / $3,000",
        "deadline": "2026-09-30",
    }


def get_profile_summary(user_id: str) -> Dict[str, Any]:
    """Summary of creator account status and rating."""
    return {
        "user_id": user_id,
        "role": "CLIPPER",
        "rank": "Gold Creator",
        "rating": 4.9,
        "total_lifetime_earnings": 3450.00,
        "completed_campaigns": 35,
        "approval_rate": "96%",
    }


# ─── 2. BRAND TOOLS ──────────────────────────────────────────────────────────

def get_campaign_stats(user_id: str, campaign_id: str) -> Dict[str, Any]:
    """Detailed performance metrics for a specific brand campaign."""
    return {
        "user_id": user_id,
        "campaign_id": campaign_id,
        "title": "Summer Product Launch",
        "status": "active",
        "total_views": 145000,
        "total_submissions": 28,
        "approved_submissions": 22,
        "spent_budget": 2175.00,
        "total_budget": 5000.00,
        "engagement_rate": "4.8%",
    }


def list_campaigns(user_id: str, status: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
    """List campaigns belonging to the Brand advertiser."""
    campaigns = [
        {"id": "camp_b1", "title": "Summer Product Launch", "status": "active", "spent": 2175.00, "budget": 5000.00, "submissions": 28},
        {"id": "camp_b2", "title": "Back to School Promo", "status": "active", "spent": 800.00, "budget": 2000.00, "submissions": 12},
        {"id": "camp_b3", "title": "Q2 Brand Awareness", "status": "ended", "spent": 3000.00, "budget": 3000.00, "submissions": 45},
    ]
    if status:
        campaigns = [c for c in campaigns if c["status"].lower() == status.lower()]
    return {"user_id": user_id, "count": len(campaigns[:limit]), "campaigns": campaigns[:limit]}


def get_budget_summary(user_id: str) -> Dict[str, Any]:
    """Query advertiser account balances and spend rates."""
    return {
        "user_id": user_id,
        "total_account_budget": 10000.00,
        "allocated_budget": 7000.00,
        "spent_budget": 2975.00,
        "remaining_unallocated_balance": 3000.00,
        "currency": "USD",
    }


def get_submissions_for_campaign(campaign_id: str, status: Optional[str] = None) -> Dict[str, Any]:
    """Query creator submissions on a brand campaign."""
    submissions = [
        {"id": "sub_501", "creatorName": "Alex Clipz", "views": 25000, "status": "approved", "payout": 375.00},
        {"id": "sub_502", "creatorName": "Sarah Media", "views": 18000, "status": "approved", "payout": 270.00},
        {"id": "sub_503", "creatorName": "Viral Shorts Daily", "views": 0, "status": "pending", "payout": 0.00},
    ]
    if status:
        submissions = [s for s in submissions if s["status"].lower() == status.lower()]
    return {"campaign_id": campaign_id, "count": len(submissions), "submissions": submissions}


def create_campaign_draft(
    user_id: str,
    title: str,
    description: str,
    budget: float,
    category: str = "general",
    deadline: Optional[str] = None
) -> Dict[str, Any]:
    """Initialize a new draft campaign for the Brand."""
    return {
        "success": True,
        "message": "Draft campaign created successfully.",
        "campaign": {
            "id": "camp_draft_999",
            "brand_id": user_id,
            "title": title,
            "description": description,
            "budget": budget,
            "category": category,
            "deadline": deadline or "2026-10-01",
            "status": "draft",
        }
    }


def get_account_summary(user_id: str) -> Dict[str, Any]:
    """Summary of advertiser brand account stats."""
    return {
        "user_id": user_id,
        "company": "PayPerView Brand Partner",
        "role": "BRAND",
        "active_campaigns_count": 2,
        "total_creators_engaged": 40,
        "total_impressions": 420000,
    }


# ─── TOOL DISPATCHER ─────────────────────────────────────────────────────────

def execute_tool(tool_name: str, user_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute tool functions dynamically by name with strictly scoped user_id.
    """
    tool_map = {
        "get_weekly_earnings": lambda: get_weekly_earnings(user_id, args.get("week", "current")),
        "get_monthly_earnings": lambda: get_monthly_earnings(user_id, args.get("month", "September"), args.get("year", 2026)),
        "get_submissions": lambda: get_submissions(user_id, args.get("status"), args.get("limit", 5)),
        "get_available_campaigns": lambda: get_available_campaigns(args.get("category"), args.get("limit", 5)),
        "get_campaign_details": lambda: get_campaign_details(args.get("campaignId") or args.get("campaign_id", "camp_201")),
        "get_profile_summary": lambda: get_profile_summary(user_id),
        "get_campaign_stats": lambda: get_campaign_stats(user_id, args.get("campaignId") or args.get("campaign_id", "camp_b1")),
        "list_campaigns": lambda: list_campaigns(user_id, args.get("status"), args.get("limit", 5)),
        "get_budget_summary": lambda: get_budget_summary(user_id),
        "get_submissions_for_campaign": lambda: get_submissions_for_campaign(args.get("campaignId") or args.get("campaign_id", "camp_b1"), args.get("status")),
        "create_campaign_draft": lambda: create_campaign_draft(user_id, args.get("title", "New Campaign"), args.get("description", ""), args.get("budget", 1000.0), args.get("category", "general"), args.get("deadline")),
        "get_account_summary": lambda: get_account_summary(user_id),
    }

    if tool_name not in tool_map:
        return {"error": f"Unknown tool name: {tool_name}"}

    try:
        return tool_map[tool_name]()
    except Exception as e:
        return {"error": f"Error executing tool '{tool_name}': {str(e)}"}
