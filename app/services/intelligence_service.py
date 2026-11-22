"""Validated, evidence-based intelligence helpers used by the internal AI API."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Check(BaseModel):
    code: str = Field(min_length=2, max_length=50)
    status: str = Field(pattern="^(PASSED|FAILED|NEEDS_REVIEW)$")
    message: str = Field(min_length=1, max_length=500)
    score: Optional[int] = Field(default=None, ge=0, le=100)


class FieldSuggestion(BaseModel):
    field: str = Field(min_length=1, max_length=60)
    suggestion: str = Field(min_length=1, max_length=1000)

    @field_validator("field")
    @classmethod
    def field_must_be_a_campaign_input(cls, value: str) -> str:
        allowed = {
            "name", "contentType", "category", "subCategories", "totalBudget", "cpm", "dailyBudgetLimit",
            "brief.mainIdea", "brief.tone", "brief.keyMessages", "brief.keywords", "brief.visualReferences",
            "targetCountries", "requirements", "submissionRequirements",
        }
        if value not in allowed:
            raise ValueError("AI suggested an unsupported campaign field")
        return value


class DraftLanguageAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: str = Field(pattern="^(APPROVED|REJECTED|MANUAL_REVIEW_REQUIRED)$")
    qualityScore: int = Field(ge=0, le=100)
    category: Literal["CLIPPING", "UGC", "SLIDESHOW", "AUDIO", "LOGO", "MIXED"]
    subCategories: list[Literal["CLIPPING", "UGC", "SLIDESHOW", "AUDIO", "LOGO"]] = Field(default_factory=list, max_length=5)
    feedback: str = Field(min_length=1, max_length=1000)
    checks: list[Check] = Field(default_factory=list, max_length=20)
    suggestions: list[FieldSuggestion] = Field(default_factory=list, max_length=12)


class BriefAlignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = Field(pattern="^(PASSED|FAILED|NEEDS_REVIEW)$")
    score: int = Field(ge=0, le=100)
    message: str = Field(min_length=1, max_length=500)


class VisualAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: str = Field(pattern="^(APPROVED|REJECTED|MANUAL_REVIEW_REQUIRED)$")
    score: int = Field(ge=0, le=100)
    logoDetected: Optional[bool] = None
    commercialElementsDetected: Optional[bool] = None
    videoQuality: str = Field(pattern="^(GOOD|POOR|UNCERTAIN)$")
    feedback: str = Field(min_length=1, max_length=1000)


def campaign_text(value: dict) -> str:
    brief = value.get("brief") or {}
    pieces = [value.get("name"), brief.get("mainIdea"), brief.get("tone"), brief.get("keyMessages"), brief.get("visualReferences")]
    keywords = brief.get("keywords") or []
    if isinstance(keywords, list):
        pieces.extend(keywords[:50])
    return "\n".join(str(item).strip() for item in pieces if isinstance(item, str) and item.strip())[:8000]


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_values, right_values = list(left), list(right)
    if not left_values or len(left_values) != len(right_values):
        return 0.0
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    return max(-1.0, min(1.0, dot / (left_norm * right_norm))) if left_norm and right_norm else 0.0


def embedding_matches(query: str, candidates: list[dict], vectors: list[list[float]], threshold: float = 0.78) -> list[dict]:
    if not query or not candidates or len(vectors) != len(candidates) + 1:
        return []
    scored = [
        {"id": str(item.get("id", "")), "similarity": round(cosine(vectors[0], vector), 4)}
        for item, vector in zip(candidates, vectors[1:])
        if item.get("id")
    ]
    return sorted((item for item in scored if item["similarity"] >= threshold), key=lambda item: item["similarity"], reverse=True)[:10]


def _word_set(value: Any) -> set[str]:
    return {item.casefold() for item in re.findall(r"[^\W_]+", str(value or ""), flags=re.UNICODE) if len(item) > 1}


def _jaccard(left: Any, right: Any) -> float:
    a, b = _word_set(left), _word_set(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def rank_campaign_applicants(campaign: dict, applicants: list[dict], similarities: Optional[list[float]] = None) -> list[dict]:
    """Score applicants from campaign fit and supplied real creator data only."""
    target_countries = {str(value).upper() for value in campaign.get("targetCountries", []) if value}
    campaign_categories = {str(value).upper() for value in campaign.get("categories", []) if value}
    performance_values = [
        max(0, int(item.get("historicalPerformance", {}).get("validatedViews", 0)))
        for item in applicants
        if int(item.get("historicalPerformance", {}).get("sampleSize", 0)) > 0
    ]
    ranked = []
    for index, applicant in enumerate(applicants):
        breakdown: dict[str, Optional[float]] = {
            "contentCompatibility": None,
            "audienceCountryMatch": None,
            "categoryMatch": None,
            "historicalPerformance": None,
        }
        weights = {"contentCompatibility": 0.35, "audienceCountryMatch": 0.25, "categoryMatch": 0.2, "historicalPerformance": 0.2}
        content = str(applicant.get("content") or "")
        campaign_content = str(campaign.get("content") or "")
        if content and campaign_content:
            similarity = similarities[index] if similarities and index < len(similarities) else _jaccard(campaign_content, content)
            breakdown["contentCompatibility"] = round(min(1.0, max(0.0, float(similarity))), 4)

        audience = {str(value).upper() for value in applicant.get("audienceCountries", []) if value}
        if target_countries and audience:
            breakdown["audienceCountryMatch"] = round(len(target_countries & audience) / len(target_countries), 4)

        categories = {str(value).upper() for value in applicant.get("categories", []) if value}
        if campaign_categories and categories:
            breakdown["categoryMatch"] = round(len(campaign_categories & categories) / len(campaign_categories), 4)

        performance = applicant.get("historicalPerformance") or {}
        sample_size = int(performance.get("sampleSize", 0))
        validated_views = max(0, int(performance.get("validatedViews", 0)))
        if sample_size > 0 and performance_values:
            # Relative percentile avoids comparing campaign-specific raw counts as a universal target.
            breakdown["historicalPerformance"] = round(sum(value <= validated_views for value in performance_values) / len(performance_values), 4)

        active_weight = sum(weights[key] for key, value in breakdown.items() if value is not None)
        score = sum(weights[key] * value for key, value in breakdown.items() if value is not None) / active_weight if active_weight else 0.0
        reasons = [
            code for key, code, threshold in [
                ("contentCompatibility", "CONTENT_COMPATIBILITY", 0.55),
                ("audienceCountryMatch", "AUDIENCE_COUNTRY_MATCH", 0.5),
                ("categoryMatch", "CATEGORY_MATCH", 0.5),
                ("historicalPerformance", "VALIDATED_PERFORMANCE_HISTORY", 0.5),
            ] if breakdown[key] is not None and breakdown[key] >= threshold
        ]
        ranked.append({
            "applicantId": str(applicant.get("id", "")),
            "compatibilityScore": round(min(1.0, max(0.0, score)), 4),
            "confidence": round(min(1.0, active_weight * (0.65 + min(sample_size, 5) * 0.07 if sample_size else 0.8)), 4),
            "reasonCodes": reasons,
            "breakdown": breakdown,
            "dataStatus": "AVAILABLE" if active_weight else "INSUFFICIENT_DATA",
        })
    ranked.sort(key=lambda item: (-item["compatibilityScore"], item["applicantId"]))
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    return ranked


def expiry_status(draft: dict, now: Optional[datetime] = None) -> dict:
    current = now or datetime.now(timezone.utc)
    expires = draft.get("expiresAt")
    last_saved = draft.get("lastSavedAt")
    if not expires:
        return {"status": "INSUFFICIENT_DATA", "notify": False, "daysUntilExpiry": None, "basis": "MISSING_EXPIRY_DATE"}
    try:
        expires_at = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
        saved_at = datetime.fromisoformat(str(last_saved).replace("Z", "+00:00")) if last_saved else None
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if saved_at and saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return {"status": "INSUFFICIENT_DATA", "notify": False, "daysUntilExpiry": None, "basis": "INVALID_DATE"}
    days = math.ceil((expires_at - current).total_seconds() / 86400)
    notify = days <= 7 and days >= 0 and saved_at is not None and (current - saved_at).days >= 7
    return {
        "status": "EXPIRING_SOON" if notify else "NOT_EXPIRING_SOON" if days >= 0 else "EXPIRED",
        "notify": notify,
        "daysUntilExpiry": days,
        "basis": "SCHEDULED_EXPIRY_AND_LAST_SAVE",
    }


def campaign_performance(current: dict, history: list[dict], matches: list[dict]) -> dict:
    match_ids = {item["id"] for item in matches}
    matched = [
        item for item in history
        if str(item.get("id", "")) in match_ids
        and item.get("stats", {}).get("totalViews") is not None
        and item.get("stats", {}).get("totalSpent") is not None
        and float(item.get("stats", {}).get("totalSpent") or 0) > 0
    ]
    if len(matched) < 3:
        return {"status": "INSUFFICIENT_DATA", "sampleSize": len(matched), "predictedViews": None, "confidence": "LOW", "recommendations": []}
    rates = [float(item["stats"]["totalViews"]) / float(item["stats"]["totalSpent"]) for item in matched]
    rate = median(rates)
    budget = float(current.get("totalBudget") or 0)
    if budget <= 0:
        return {"status": "INSUFFICIENT_DATA", "sampleSize": len(matched), "predictedViews": None, "confidence": "LOW", "recommendations": []}
    predicted = max(0, round(rate * budget))
    recommendation = []
    current_stats = current.get("stats") or {}
    current_spend = float(current_stats.get("totalSpent") or 0)
    current_views = float(current_stats.get("totalViews") or 0)
    if current_spend > 0 and current_views / current_spend < rate * 0.75:
        recommendation.append("Observed views per dollar are below the median of similar completed campaigns; review targeting and creative fit.")
    return {"status": "ESTIMATED", "sampleSize": len(matched), "predictedViews": predicted, "viewsPerDollarMedian": round(rate, 2), "confidence": "MEDIUM" if len(matched) >= 8 else "LOW", "recommendations": recommendation}


def category_analytics(campaigns: list[dict]) -> dict:
    grouped: dict[str, list[float]] = {}
    for campaign in campaigns:
        stats = campaign.get("stats") or {}
        spent, views = stats.get("totalSpent"), stats.get("totalViews")
        category = campaign.get("category") or campaign.get("contentType")
        if category and isinstance(spent, (int, float)) and spent > 0 and isinstance(views, (int, float)) and views >= 0:
            grouped.setdefault(str(category), []).append(float(views) / float(spent))
    results = [
        {"category": name, "sampleSize": len(values), "medianViewsPerDollar": round(median(values), 2)}
        for name, values in grouped.items() if len(values) >= 3
    ]
    return {"status": "AVAILABLE" if results else "INSUFFICIENT_DATA", "categories": sorted(results, key=lambda item: item["medianViewsPerDollar"], reverse=True)}


def campaign_business_analysis(history: list[dict], policy_acceptance: Optional[dict] = None) -> dict:
    outcomes = []
    for campaign in history:
        stats = campaign.get("stats") or {}
        spent, views = stats.get("totalSpent"), stats.get("totalViews")
        if isinstance(spent, (int, float)) and not isinstance(spent, bool) and spent > 0 and isinstance(views, (int, float)) and not isinstance(views, bool) and views >= 0:
            outcomes.append((campaign, float(views) / float(spent)))

    archived = [campaign for campaign in history if campaign.get("isArchived") is True or campaign.get("status") == "ARCHIVED"]
    archived_outcomes = [rate for campaign, rate in outcomes if campaign in archived]
    archive_analysis = {
        "status": "AVAILABLE" if archived else "INSUFFICIENT_DATA",
        "archivedCampaigns": len(archived),
        "archivedCampaignsWithMeasuredOutcomes": len(archived_outcomes),
        "medianArchivedViewsPerDollar": round(median(archived_outcomes), 2) if archived_outcomes else None,
    }

    overall_median = median([rate for _, rate in outcomes]) if len(outcomes) >= 3 else None
    grouped: dict[str, list[float]] = {}
    if overall_median is not None:
        for campaign, rate in outcomes:
            category = campaign.get("category") or campaign.get("contentType")
            if category:
                grouped.setdefault(str(category), []).append(rate)
    patterns = [
        {"category": category, "sampleSize": len(rates), "medianViewsPerDollar": round(median(rates), 2)}
        for category, rates in grouped.items()
        if len(rates) >= 3 and median(rates) > overall_median
    ]
    successful = {
        "status": "AVAILABLE" if patterns else "INSUFFICIENT_DATA",
        "sampleSize": len(outcomes),
        "patterns": sorted(patterns, key=lambda item: item["medianViewsPerDollar"], reverse=True)[:20],
        "basis": "Categories with at least three completed or archived owner campaigns and median views per dollar above the owner's overall median.",
    }

    total = int((policy_acceptance or {}).get("totalSubmissions") or 0)
    accepted = int((policy_acceptance or {}).get("acceptedSubmissions") or 0)
    policy = {"status": "AVAILABLE" if total > 0 else "INSUFFICIENT_DATA", "totalSubmissions": total, "acceptedSubmissions": accepted, "acceptanceRate": round(accepted / total, 4) if total else None, "recommendations": []}
    if total and accepted / total < 0.8:
        policy["recommendations"].append("Review campaign policy and pledge wording for clarity; low pledge acceptance alone does not establish a policy violation.")
    return {"archivedCampaignAnalysis": archive_analysis, "successfulCampaignPatterns": successful, "policyAcceptanceAnalytics": policy}


def robust_outlier(value: float, history: list[float]) -> bool:
    if len(history) < 5:
        return False
    center = median(history)
    deviations = [abs(item - center) for item in history]
    mad = median(deviations)
    if mad == 0:
        return value > center * 3 if center > 0 else value > 0
    return abs(value - center) / (1.4826 * mad) >= 3.5


def robust_two_sided_outlier(value: float, history: list[float]) -> bool:
    """Detect high or low outliers in matured earnings yield after five samples."""
    if len(history) < 5:
        return False
    center = median(history)
    deviations = [abs(item - center) for item in history]
    mad = median(deviations)
    if mad == 0:
        if center <= 0:
            return value > 0
        return value > center * 3 or value < center / 3
    return abs(value - center) / (1.4826 * mad) >= 3.5


def budget_analysis(payload: dict) -> dict:
    transactions = payload.get("transactions", [])
    payment_events = [item for item in transactions if item.get("type") == "CREDIT"]
    completed = [item for item in payment_events if item.get("status") == "COMPLETED"]
    failed = [item for item in payment_events if item.get("status") in {"FAILED", "CANCELLED"}]
    amounts = [float(item["amount"]) for item in completed if isinstance(item.get("amount"), (int, float)) and item["amount"] > 0]
    funding = payload.get("candidateDeposit") or {}
    candidate_amount = funding.get("amount")
    fraud = {"status": "INSUFFICIENT_DATA", "risk": "UNKNOWN", "reasonCodes": [], "requiresHumanReview": False}
    terminal_payment_events = [item for item in payment_events if item.get("status") in {"COMPLETED", "FAILED", "CANCELLED"}]
    if len(payment_events) >= 5 and terminal_payment_events:
        reasons = []
        failure_ratio = len(failed) / max(1, len(payment_events))
        if failure_ratio >= 0.6:
            reasons.append("HIGH_RECENT_FAILURE_RATIO")
        if isinstance(candidate_amount, (int, float)) and candidate_amount > 0 and robust_outlier(float(candidate_amount), amounts):
            reasons.append("AMOUNT_OUTLIER_VS_COMPLETED_HISTORY")
        fraud = {"status": "ASSESSED", "risk": "HIGH" if len(reasons) == 2 else "MEDIUM" if reasons else "LOW", "reasonCodes": reasons, "requiresHumanReview": len(reasons) == 2}

    country = str(payload.get("country") or "").upper()
    country_methods = payload.get("paymentMethodsByCountry", {}) or {}
    configured_methods = set(payload.get("availablePaymentMethods", []))
    methods = (
        set(country_methods.get(country, [])) & configured_methods
        if country in country_methods
        else configured_methods
    )
    method_counts: dict[str, int] = {}
    method_history = completed if len(completed) >= 3 else []
    for item in method_history:
        method = item.get("paymentMethod")
        if method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
    best_method = max(method_counts, key=method_counts.get) if method_counts else None
    payment_method = {"status": "RECOMMENDED" if best_method else "INSUFFICIENT_DATA", "method": best_method, "basis": "COUNTRY_ELIGIBILITY_AND_COMPLETED_PAYMENT_HISTORY" if best_method and country_methods.get(country) else "COMPLETED_PAYMENT_HISTORY" if best_method else None}

    campaigns = payload.get("campaigns", [])
    deposit_amount = median([float(item.get("totalBudget", 0)) for item in campaigns if isinstance(item.get("totalBudget"), (int, float)) and item.get("totalBudget", 0) > 0]) if len(campaigns) >= 3 else None
    deposit = {"status": "ESTIMATED" if deposit_amount else "INSUFFICIENT_DATA", "amount": round(deposit_amount, 2) if deposit_amount else None, "sampleSize": len(campaigns) if deposit_amount else 0, "basis": "MEDIAN_OF_OWN_CAMPAIGN_BUDGETS" if deposit_amount else None}

    daily_values = [float(value) for value in payload.get("dailySpending", []) if isinstance(value, (int, float)) and value >= 0]
    anomaly = {"status": "ASSESSED" if len(daily_values) >= 6 else "INSUFFICIENT_DATA", "risk": "UNKNOWN", "reasonCodes": []}
    if len(daily_values) >= 6:
        outlier = robust_outlier(daily_values[-1], daily_values[:-1])
        anomaly = {"status": "ASSESSED", "risk": "HIGH" if outlier else "LOW", "reasonCodes": ["ROBUST_DAILY_SPEND_OUTLIER"] if outlier else []}

    current = payload.get("campaign") or {}
    average_daily = median(daily_values[-7:]) if len(daily_values) >= 3 else None
    exhaustion = {"status": "INSUFFICIENT_DATA", "daysRemaining": None, "estimatedExhaustionAt": None}
    remaining = current.get("remainingBudget")
    if average_daily is not None and average_daily > 0 and isinstance(remaining, (int, float)) and remaining >= 0:
        days = max(0, round(float(remaining) / average_daily, 1))
        exhaustion = {"status": "ESTIMATED", "daysRemaining": days, "estimatedExhaustionAt": (datetime.now(timezone.utc).date() + timedelta(days=int(days))).isoformat()}
    daily_budget = {"status": "INSUFFICIENT_DATA", "recommendedLimit": None, "basis": None}
    if len(daily_values) >= 3:
        daily_budget = {"status": "ESTIMATED", "recommendedLimit": round(median(daily_values[-7:]) * 1.2, 2), "basis": "120_PERCENT_OF_RECENT_MEDIAN_DAILY_SPEND"}

    refunds = payload.get("refunds", [])
    refund_fraud = {"status": "INSUFFICIENT_DATA", "risk": "UNKNOWN", "reasonCodes": []}
    if len(refunds) >= 5:
        rejected = sum(1 for item in refunds if item.get("status") == "REJECTED")
        refund_fraud = {"status": "ASSESSED", "risk": "HIGH" if rejected / len(refunds) >= 0.6 else "LOW", "reasonCodes": ["HIGH_HISTORICAL_REJECTION_RATE"] if rejected / len(refunds) >= 0.6 else []}
    refund_likelihood = {"status": "INSUFFICIENT_DATA", "probability": None, "sampleSize": 0}
    if len(completed) >= 5:
        refund_likelihood = {"status": "ESTIMATED", "probability": round(min(1.0, len(refunds) / len(completed)), 3), "sampleSize": len(completed)}
    refund_original_method = payload.get("refundOriginalMethod")
    refund_method = {"status": "RECOMMENDED", "method": refund_original_method, "basis": "ORIGINAL_PAYMENT_METHOD_REQUIRED_BY_CURRENT_REFUND_POLICY"} if refund_original_method in {"PAYPAL", "BANK_TRANSFER", "MOYASAR"} else {"status": "INSUFFICIENT_DATA", "method": None, "basis": "REFUND_METHOD_MUST_MATCH_ORIGINAL_PAYMENT_METHOD"}
    return {"paymentFraud": fraud, "paymentMethodRecommendation": payment_method, "depositAmountPrediction": deposit, "dailyBudgetRecommendation": daily_budget, "budgetExhaustionPrediction": exhaustion, "spendingAnomaly": anomaly, "refundFraud": refund_fraud, "refundMethodRecommendation": refund_method, "refundLikelihoodPrediction": refund_likelihood}


def normalize_text(value: str) -> str:
    value = value.casefold().replace("ـ", "")
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {normalize_text(text)} "


def analyze_view_history(payload: dict, *, now: Optional[datetime] = None) -> dict:
    """Return evidence-based view and earnings indicators from supplied platform records.

    View predictions use no more than the recent seven-day aggregation. Historical
    earnings are accepted only as matured ledger aggregates supplied by Main Backend.
    Geographic mismatch is reported as a targeting anomaly and never as proof of fraud.
    """
    current = now or datetime.now(timezone.utc)
    cutoff = current - timedelta(days=7)
    events = []
    for raw in payload.get("events", []):
        try:
            captured = datetime.fromisoformat(str(raw.get("capturedAt", "")).replace("Z", "+00:00"))
            if captured.tzinfo is None:
                captured = captured.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if cutoff <= captured <= current:
            events.append({**raw, "_captured": captured})

    by_day: dict[str, dict] = {}
    per_submission: dict[str, dict] = {}
    submission_daily_views: dict[str, dict[str, int]] = {}
    country_totals: dict[str, int] = {}
    target_countries = {str(code).upper() for code in payload.get("targetCountries", [])}
    out_of_target = 0
    for event in events:
        day = event["_captured"].date().isoformat()
        day_bucket = by_day.setdefault(day, {"views": 0, "validatedViews": 0, "pendingValidationViews": 0, "flaggedViews": 0})
        delta = max(0, int(event.get("deltaViews") or 0))
        day_bucket["views"] += delta
        eligible = max(0, int(event.get("eligibleViews") or 0))
        review_status = event.get("reviewStatus")
        if review_status == "CONFIRMED":
            day_bucket["validatedViews"] += eligible
        elif review_status == "CLEAR":
            day_bucket["pendingValidationViews"] += eligible
        elif review_status == "REVIEW_REQUIRED":
            day_bucket["flaggedViews"] += delta

        submission_id = str(event.get("submissionId") or "")
        if submission_id:
            record = per_submission.setdefault(submission_id, {"views": 0, "flags": set(), "reviewRequired": False})
            record["views"] += delta
            record["flags"].update(str(flag) for flag in event.get("flags", []))
            record["reviewRequired"] = record["reviewRequired"] or review_status == "REVIEW_REQUIRED"
            submission_days = submission_daily_views.setdefault(submission_id, {})
            submission_days[day] = submission_days.get(day, 0) + delta

        for item in event.get("countryDeltas", []):
            code = str(item.get("countryCode", "")).upper()
            views = max(0, int(item.get("views") or 0))
            if not re.fullmatch(r"[A-Z]{3}", code):
                continue
            country_totals[code] = country_totals.get(code, 0) + views
            if target_countries and code not in target_countries:
                out_of_target += views

    day_keys = sorted(by_day)
    daily_views = [by_day[key]["views"] for key in day_keys]
    recent_observed = sum(daily_views)
    mature_history = [item for item in payload.get("validatedHistory", []) if int(item.get("validatedViews") or 0) > 0]
    matured_validated_views = sum(int(item.get("validatedViews") or 0) for item in mature_history)
    confirmed_gross_cents = sum(int(item.get("confirmedGrossAmountCents") or 0) for item in mature_history)
    confirmed_net_cents = sum(int(item.get("confirmedCreatorNetAmountCents") or 0) for item in mature_history)
    earnings_yield = [
        int(item.get("confirmedGrossAmountCents") or 0) * 1000 / int(item["validatedViews"])
        for item in mature_history
        if int(item.get("validatedViews") or 0) > 0
    ]
    earnings_anomaly = {
        "status": "INSUFFICIENT_DATA",
        "detected": False,
        "confidence": 0.0,
        "direction": "UNKNOWN",
        "explanation": "At least six matured daily ledger observations are required to assess earnings yield.",
    }
    if len(earnings_yield) >= 6:
        center = median(earnings_yield[:-1])
        detected = robust_two_sided_outlier(earnings_yield[-1], earnings_yield[:-1])
        earnings_anomaly = {
            "status": "ASSESSED",
            "detected": detected,
            "confidence": 0.75 if detected else 0.6,
            "direction": "HIGH" if detected and earnings_yield[-1] > center else "LOW" if detected else "NONE",
            "explanation": "Latest confirmed gross cents per 1,000 matured validated views were compared with prior daily ledger yield; an anomaly is a review signal, not evidence of a ledger error.",
        }

    flags = []
    suspicious = []
    for submission_id, record in per_submission.items():
        # OUT_OF_TARGET_COUNTRY is a delivery anomaly only and must never be
        # used as a bot/fake-view signal by itself.
        reason_codes = sorted(flag for flag in record["flags"] if flag != "OUT_OF_TARGET_COUNTRY")
        submission_days = submission_daily_views.get(submission_id, {})
        ordered_submission_days = sorted(submission_days)
        if len(ordered_submission_days) >= 6 and robust_outlier(
            submission_days[ordered_submission_days[-1]],
            [submission_days[day] for day in ordered_submission_days[:-1]],
        ):
            reason_codes.append("ABNORMAL_DAILY_VIEW_SPIKE")
        if record["reviewRequired"] and not record["flags"]:
            reason_codes.append("PENDING_HUMAN_VALIDATION")
        if reason_codes:
            suspicious.append({
                "submissionId": submission_id,
                "status": "REVIEW_RECOMMENDED",
                "confidence": 0.75 if record["reviewRequired"] else 0.65,
                "reasonCodes": sorted(set(reason_codes)),
                "explanation": "Existing tracking flags or pending validation require review; this signal does not establish bot activity or fraud.",
                "requiresHumanReview": True,
            })

    spike = {"status": "INSUFFICIENT_DATA", "detected": False, "confidence": 0.0, "explanation": "At least six daily buckets are required to assess a seven-day spike."}
    drop = {"status": "INSUFFICIENT_DATA", "detected": False, "confidence": 0.0, "explanation": "At least six daily buckets are required to assess a seven-day decline."}
    prediction = {"status": "INSUFFICIENT_DATA", "predictedNext7DaysViews": None, "confidence": 0.0, "explanation": "Daily view history is insufficient for a forecast."}
    if len(daily_views) >= 6:
        baseline = median(daily_views[:-1])
        last = daily_views[-1]
        is_spike = robust_outlier(last, daily_views[:-1])
        is_drop = baseline > 0 and last <= baseline * 0.5
        spike = {"status": "ASSESSED", "detected": is_spike, "confidence": 0.75 if is_spike else 0.6, "explanation": "The latest daily count is compared with the preceding daily median using a robust outlier check."}
        drop = {"status": "ASSESSED", "detected": is_drop, "confidence": 0.75 if is_drop else 0.6, "explanation": "The latest daily count is at most half the preceding daily median." if is_drop else "No qualifying sudden drop was observed in the available seven-day window."}
    if len(daily_views) >= 4 and any(value > 0 for value in daily_views):
        recent = daily_views[-4:]
        slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        daily_forecast = max(0.0, recent[-1] + slope * 3.5)
        forecast = max(0, round((sum(recent) / len(recent) * 3.5) + daily_forecast * 3.5))
        prediction = {
            "status": "ESTIMATED",
            "predictedNext7DaysViews": forecast,
            "confidence": min(0.8, 0.35 + 0.07 * len(daily_views)),
            "explanation": "Linear trend blended with the recent daily mean; this is an estimate from the supplied seven-day window, not a guarantee.",
        }

    geography = sorted(country_totals.items(), key=lambda item: item[1], reverse=True)
    country_growth = []
    if len(day_keys) >= 4:
        daily_country_counts: dict[tuple[str, str], int] = {}
        for event in events:
            event_day = event["_captured"].date().isoformat()
            for item in event.get("countryDeltas", []):
                country = str(item.get("countryCode", "")).upper()
                if re.fullmatch(r"[A-Z]{3}", country):
                    key = (event_day, country)
                    daily_country_counts[key] = daily_country_counts.get(key, 0) + max(0, int(item.get("views") or 0))
        for country, total in geography[:10]:
            counts = [daily_country_counts.get((day, country), 0) for day in day_keys]
            if sum(count > 0 for count in counts) >= 2:
                growth = (counts[-1] - counts[0]) / max(len(counts) - 1, 1)
                country_growth.append({"countryCode": country, "observedViews": total, "predictedNext7DaysViews": max(0, round(counts[-1] * 7 + growth * 21)), "confidence": min(0.75, 0.3 + 0.08 * len(counts)), "explanation": "Linear per-day trend over available country deltas, extrapolated seven days."})

    earnings_prediction = {"status": "INSUFFICIENT_DATA", "estimatedGrossAmountCents": None, "estimatedCreatorNetAmountCents": None, "currency": "USD", "confidence": 0.0, "explanation": "A positive CPM, matured validated views, and confirmed historical earnings are required. Estimates are separate from the earnings ledger."}
    cpm = payload.get("cpm")
    if isinstance(cpm, (int, float)) and not isinstance(cpm, bool) and cpm > 0 and matured_validated_views > 0 and confirmed_gross_cents > 0 and prediction["predictedNext7DaysViews"] is not None:
        historical_cpm = confirmed_gross_cents * 1000 / matured_validated_views / 100
        effective_cpm = min(float(cpm), historical_cpm) if historical_cpm > 0 else 0
        if effective_cpm > 0:
            estimated_cents = round(prediction["predictedNext7DaysViews"] * effective_cpm / 1000 * 100)
            remaining_budget = payload.get("remainingBudget")
            if isinstance(remaining_budget, (int, float)) and not isinstance(remaining_budget, bool):
                estimated_cents = min(estimated_cents, max(0, round(float(remaining_budget) * 100)))
            net_cpm = confirmed_net_cents * 1000 / matured_validated_views / 100 if confirmed_net_cents > 0 else 0
            estimated_net_cents = min(estimated_cents, round(prediction["predictedNext7DaysViews"] * min(effective_cpm, net_cpm) / 1000 * 100)) if net_cpm > 0 else None
            earnings_prediction = {
                "status": "ESTIMATED",
                "estimatedGrossAmountCents": max(0, estimated_cents),
                "estimatedCreatorNetAmountCents": max(0, estimated_net_cents) if estimated_net_cents is not None else None,
                "currency": "USD",
                "confidence": min(prediction["confidence"], 0.7 if len(mature_history) >= 7 else 0.45),
                "explanation": "Estimated from the seven-day view forecast, the lower of campaign CPM and realized matured gross CPM, matured validated ledger views, and confirmed gross earnings. It is not confirmed or withdrawable.",
            }

    bot_reason_codes = {code for item in suspicious for code in item["reasonCodes"]}
    if spike["status"] == "ASSESSED" and spike["detected"]:
        bot_reason_codes.add("ABNORMAL_DAILY_VIEW_SPIKE")
    bot_assessment = {
        "status": "REVIEW_RECOMMENDED" if bot_reason_codes else "INSUFFICIENT_DATA" if not events else "NO_STRONG_SIGNAL",
        "confidence": max([item["confidence"] for item in suspicious] + ([spike["confidence"]] if "ABNORMAL_DAILY_VIEW_SPIKE" in bot_reason_codes else []), default=0.0),
        "reasonCodes": sorted(bot_reason_codes),
        "explanation": "Signals indicate possible invalid activity and require human validation; no automated result proves a bot or fake view.",
    }
    return {
        "windowDays": 7,
        "observedViews": recent_observed,
        "validatedViewsInWindow": sum(by_day[key]["validatedViews"] for key in by_day),
        "pendingValidationViewsInWindow": sum(by_day[key]["pendingValidationViews"] for key in by_day),
        "flaggedViewsPendingReview": sum(by_day[key]["flaggedViews"] for key in by_day),
        "maturedValidatedViews": matured_validated_views,
        "confirmedGrossEarningsCents": confirmed_gross_cents,
        "confirmedCreatorNetEarningsCents": confirmed_net_cents,
        "suspiciousPatterns": suspicious[:100],
        "botAndFakeViewAssessment": bot_assessment,
        "abnormalViewSpike": spike,
        "suddenViewDrop": drop,
        "futureViewPrediction": prediction,
        "geographicAudience": {"status": "AVAILABLE" if geography else "INSUFFICIENT_DATA", "countries": [{"countryCode": country, "views": views, "targeted": (not target_countries or country in target_countries)} for country, views in geography[:20]], "confidence": min(0.8, 0.35 + 0.07 * len(day_keys)) if geography else 0.0, "explanation": "Aggregated incremental country view deltas only; lifetime country totals are not summed repeatedly."},
        "countryGrowthPrediction": {"status": "ESTIMATED" if country_growth else "INSUFFICIENT_DATA", "countries": country_growth, "explanation": "Country-level growth is estimated only when at least four daily buckets and two non-zero observations are available."},
        "nonTargetGeographicAnomaly": {"status": "ASSESSED" if target_countries and geography else "INSUFFICIENT_DATA", "detected": bool(target_countries and out_of_target > 0), "outOfTargetViews": out_of_target, "isFraud": False, "confidence": 0.65 if out_of_target else 0.0, "explanation": "Geographic targeting mismatch is an analytics anomaly only; audience location alone is not evidence of fraud."},
        "earningsPrediction": earnings_prediction,
        "earningsAnomaly": earnings_anomaly,
    }


def predict_creator_earnings(payload: dict) -> dict:
    """Forecast creator net earnings from matured validated daily views and realized ledger CPM."""
    predictions = []
    for campaign in payload.get("campaigns", []):
        history = sorted(campaign.get("dailyHistory", []), key=lambda item: str(item.get("day", "")))
        views = [max(0, int(item.get("validatedViews", 0))) for item in history]
        total_views = sum(views)
        gross_history = sum(max(0, int(item.get("confirmedGrossAmountCents", 0))) for item in history)
        net_history = sum(max(0, int(item.get("confirmedCreatorNetAmountCents", 0))) for item in history)
        cpm = campaign.get("applicableCpm")
        result = {
            "campaignId": str(campaign.get("campaignId", "")),
            "status": "INSUFFICIENT_DATA",
            "historicalDays": len(history),
            "maturedValidatedViews": total_views,
            "predictedNext7DaysViews": None,
            "estimatedGrossAmountCents": None,
            "estimatedCreatorNetAmountCents": None,
            "confidence": 0.0,
            "explanation": "At least four matured daily view observations, a positive applicable CPM, and confirmed creator ledger earnings are required.",
        }
        if len(history) < 4 or not total_views or not isinstance(cpm, (int, float)) or isinstance(cpm, bool) or cpm <= 0 or not gross_history or not net_history:
            predictions.append(result)
            continue

        recent = views[-4:]
        slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        forecast_views = max(0, round((sum(recent) / len(recent) * 3.5) + max(0.0, recent[-1] + slope * 3.5) * 3.5))
        applicable_cpm_cents = float(cpm) * 100
        realized_gross_cpm_cents = gross_history * 1000 / total_views
        realized_net_cpm_cents = net_history * 1000 / total_views
        effective_gross_cpm_cents = min(applicable_cpm_cents, realized_gross_cpm_cents)
        effective_net_cpm_cents = min(effective_gross_cpm_cents, realized_net_cpm_cents)
        if effective_gross_cpm_cents <= 0 or effective_net_cpm_cents <= 0:
            predictions.append(result)
            continue

        result.update({
            "status": "ESTIMATED",
            "predictedNext7DaysViews": forecast_views,
            "estimatedGrossAmountCents": max(0, round(forecast_views * effective_gross_cpm_cents / 1000)),
            "estimatedCreatorNetAmountCents": max(0, round(forecast_views * effective_net_cpm_cents / 1000)),
            "confidence": min(0.65, 0.35 + 0.04 * len(history)),
            "explanation": "Estimated from matured validated daily views, the applicable campaign CPM capped by realized gross CPM, and the creator's confirmed net-to-view history. This estimate is not confirmed or withdrawable.",
        })
        predictions.append(result)

    estimated = [item for item in predictions if item["status"] == "ESTIMATED"]
    return {
        "status": "ESTIMATED" if estimated else "INSUFFICIENT_DATA",
        "windowDays": 7,
        "currency": "USD",
        "method": "MATURED_VIEW_TREND_AND_REALIZED_CPM",
        "estimatedGrossAmountCents": sum(item["estimatedGrossAmountCents"] or 0 for item in estimated) if estimated else None,
        "estimatedCreatorNetAmountCents": sum(item["estimatedCreatorNetAmountCents"] or 0 for item in estimated) if estimated else None,
        "predictedNext7DaysViews": sum(item["predictedNext7DaysViews"] or 0 for item in estimated) if estimated else None,
        "confidence": round(sum(item["confidence"] for item in estimated) / len(estimated), 4) if estimated else 0.0,
        "campaignsAnalyzed": len(predictions),
        "campaignsEstimated": len(estimated),
        "explanation": "All returned amounts are estimates from the creator's matured validated-view and earnings history. They do not change or represent the confirmed wallet balance.",
        "campaigns": predictions,
    }


def analyze_withdrawal_history(payload: dict, *, now: Optional[datetime] = None) -> dict:
    """Assess withdrawal patterns using only the authenticated creator's history."""
    now = now or datetime.now(timezone.utc)
    candidate = payload.get("candidate") or {}
    candidate_amount = candidate.get("amountCents")
    candidate_method = str(candidate.get("method") or "").upper()
    history = [item for item in payload.get("withdrawals", []) if isinstance(item, dict)]

    def parsed_date(value: Any) -> Optional[datetime]:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    dated = [(item, parsed_date(item.get("createdAt"))) for item in history]
    recent24 = [item for item, created in dated if created and (now - created).total_seconds() <= 86400 and created <= now]
    completed = [item for item in history if item.get("status") == "COMPLETED" and isinstance(item.get("amountCents"), int) and not isinstance(item.get("amountCents"), bool) and item["amountCents"] > 0]
    signals: list[dict[str, str]] = []

    if isinstance(candidate_amount, int) and not isinstance(candidate_amount, bool) and candidate_amount > 0:
        repeats = [
            item for item in recent24
            if item.get("amountCents") == candidate_amount
            and (not candidate_method or str(item.get("method") or "").upper() == candidate_method)
        ]
        if len(repeats) >= 2:
            signals.append({"code": "REPEATED_REQUEST_PATTERN", "severity": "HIGH" if len(repeats) >= 4 else "MEDIUM", "explanation": f"{len(repeats)} matching requests were recorded in the last 24 hours."})

        prior_amounts = [int(item["amountCents"]) for item in completed]
        if len(prior_amounts) >= 5:
            typical = median(prior_amounts)
            mad = median([abs(value - typical) for value in prior_amounts])
            threshold = max(typical + max(3 * mad, typical), typical * 3)
            if candidate_amount > threshold:
                signals.append({"code": "UNUSUAL_WITHDRAWAL_AMOUNT", "severity": "MEDIUM", "explanation": "The requested amount is materially above this account's completed-withdrawal history."})

    if len(recent24) >= 5:
        signals.append({"code": "HIGH_REQUEST_FREQUENCY", "severity": "MEDIUM", "explanation": "At least five withdrawal requests were recorded in the last 24 hours."})

    # A candidate request alone is not enough to establish a normal baseline.
    risk = "UNKNOWN" if not history else "LOW"
    if any(signal["severity"] == "HIGH" for signal in signals):
        risk = "HIGH"
    elif signals:
        risk = "MEDIUM"

    durations = []
    for item in completed:
        created = parsed_date(item.get("createdAt"))
        finished = parsed_date(item.get("completedAt"))
        if created and finished and finished >= created:
            durations.append((finished - created).total_seconds() / 3600)
    processing = {"status": "INSUFFICIENT_DATA", "medianHours": None, "sampleSize": len(durations)}
    if len(durations) >= 3:
        processing = {"status": "ESTIMATED", "medianHours": round(median(durations), 2), "sampleSize": len(durations)}

    country = str(payload.get("country") or "").upper()
    regional = (payload.get("paymentMethodsByCountry") or {}).get(country, []) if country else []
    owned = {str(method).upper() for method in payload.get("availableMethods", []) if isinstance(method, str)}
    recommended = [method.upper() for method in regional if isinstance(method, str) and method.upper() in owned]
    return {
        "status": "ASSESSED" if history or candidate_amount is not None else "INSUFFICIENT_DATA",
        "risk": risk,
        "confidence": "MEDIUM" if len(completed) >= 5 else "LOW",
        "requiresHumanReview": risk == "HIGH",
        "signals": signals,
        "historySampleSize": len(history),
        "requestsLast24Hours": len(recent24),
        "processingTimePrediction": processing,
        "paymentMethodRecommendation": {
            "status": "RECOMMENDED" if recommended else "INSUFFICIENT_DATA",
            "country": country or None,
            "methods": recommended,
            "basis": "Configured country support intersected with the creator's active payment methods." if recommended else None,
        },
    }
