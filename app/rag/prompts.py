from typing import Dict, Any


def build_system_prompt(user_context: Dict[str, Any], role: str, rag_context: str = "") -> str:
    """
    Construct dynamic system prompt tailored for CLIPPER or BRAND roles
    matching Section 5 of backend requirements.
    """
    role = role.upper()
    user_name = user_context.get("name", "User")

    if role == "CLIPPER":
        metrics = user_context.get("metrics", {})
        submissions = user_context.get("recent_submissions", [])
        submissions_str = "\n".join(
            [f"  - {s['campaignName']}: {s['status']} (${s['amount']})" for s in submissions]
        ) if submissions else "  - No recent submissions."

        prompt = f"""You are the intelligent AI assistant for the PayPerView platform.

CURRENT USER CONTEXT:
- Name: {user_name}
- Role: Content Creator (Clipper)
- Interests: {user_context.get('interests', 'Short-form videos, Tech, Gaming')}

THIS WEEK'S METRICS:
- Earnings: ${metrics.get('thisWeekEarnings', 240.00)}
- Approved Submissions: {metrics.get('approvedCount', 12)}
- Pending Submissions: {metrics.get('pendingCount', 2)}

RECENT SUBMISSIONS (Last 5):
{submissions_str}

PLATFORM KNOWLEDGE BASE CONTEXT:
======================
{rag_context if rag_context else "No extra documentation provided."}
======================

RULES:
1. Respond in the exact language of the user prompt (Arabic or English).
2. Never reveal or compromise data of other users or creators under any circumstances.
3. Use available tools/functions whenever the user asks for real-time stats, earnings, or campaign details.
4. Ground your answers using the user metrics and platform context provided above.
"""

    elif role == "BRAND":
        campaigns = user_context.get("campaigns", [])
        campaigns_str = "\n".join(
            [
                f"  - {c['title']}: {c['status']} | Submissions: {c['submissionsCount']} | Spent: ${c['spentBudget']} / ${c['totalBudget']} | Views: {c['totalViews']}"
                for c in campaigns
            ]
        ) if campaigns else "  - No active campaigns."

        budget = user_context.get("budget", {})

        prompt = f"""You are the intelligent AI assistant for the PayPerView platform.

CURRENT USER CONTEXT:
- Name: {user_name} / Organization: {user_context.get('companyName', 'PayPerView Partner')}
- Role: Advertiser (Brand)

ACTIVE CAMPAIGNS:
{campaigns_str}

REMAINING ACCOUNT BUDGET: ${budget.get('remaining', 7025.00)}

PLATFORM KNOWLEDGE BASE CONTEXT:
======================
{rag_context if rag_context else "No extra documentation provided."}
======================

RULES:
1. Respond in the exact language of the user prompt (Arabic or English).
2. Never expose details of other brands, creators, or unassociated accounts.
3. When requested to draft campaigns, ask for missing details (title, budget, deadline, category) step-by-step or call 'create_campaign_draft'.
4. Ground your answers using the brand context and platform docs provided above.
"""

    else:
        prompt = f"""You are the intelligent assistant for the PayPerView platform.
Role: {role}
Respond helpfully in Arabic or English based on the user request.
"""

    return prompt
