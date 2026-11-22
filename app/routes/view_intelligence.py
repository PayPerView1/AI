"""Service-key protected, statistical intelligence over the existing view ledger."""

from datetime import datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.routes.reviews import require_service_key
from app.services.intelligence_service import analyze_view_history


router = APIRouter(prefix="/api/v1/ai/intelligence/views", tags=["AI Views and Earnings Intelligence"])


class CountryDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    countryCode: str = Field(pattern="^[A-Z]{3}$")
    views: int = Field(ge=0)


class ViewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    submissionId: str = Field(min_length=1, max_length=100)
    capturedAt: datetime
    deltaViews: int = Field(ge=0)
    eligibleViews: int = Field(ge=0)
    reviewStatus: Literal["CLEAR", "REVIEW_REQUIRED", "CONFIRMED", "DISMISSED"]
    flags: List[str] = Field(default_factory=list, max_length=20)
    countryDeltas: List[CountryDelta] = Field(default_factory=list, max_length=250)


class ValidatedHistoryDay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    day: str = Field(pattern="^\\d{4}-\\d{2}-\\d{2}$")
    validatedViews: int = Field(ge=0)
    confirmedGrossAmountCents: int = Field(ge=0)
    confirmedCreatorNetAmountCents: int = Field(ge=0)


class ViewsAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "campaignId": "66a111111111111111111111", "cpm": 5.0, "remainingBudget": 100, "targetCountries": ["JOR"],
        "events": [{"submissionId": "66b111111111111111111111", "capturedAt": "2026-11-09T12:00:00Z", "deltaViews": 120, "eligibleViews": 120, "reviewStatus": "CLEAR", "flags": [], "countryDeltas": [{"countryCode": "JOR", "views": 118}, {"countryCode": "USA", "views": 2}]}],
        "validatedHistory": [{"day": "2026-11-01", "validatedViews": 1000, "confirmedGrossAmountCents": 500, "confirmedCreatorNetAmountCents": 475}],
    }]})
    campaignId: str = Field(min_length=1, max_length=100)
    cpm: Optional[float] = Field(default=None, ge=0, le=100000)
    remainingBudget: Optional[float] = Field(default=None, ge=0, le=1000000000)
    targetCountries: List[str] = Field(default_factory=list, max_length=250)
    events: List[ViewEvent] = Field(default_factory=list, max_length=10000)
    validatedHistory: List[ValidatedHistoryDay] = Field(default_factory=list, max_length=90)


class SignalAssessment(BaseModel):
    status: Literal["INSUFFICIENT_DATA", "ASSESSED"]
    detected: bool
    confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=1, max_length=500)


class ViewPrediction(BaseModel):
    status: Literal["INSUFFICIENT_DATA", "ESTIMATED"]
    predictedNext7DaysViews: Optional[int] = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=1, max_length=500)


class CountryAudience(BaseModel):
    countryCode: str
    views: int = Field(ge=0)
    targeted: bool


class CountryGrowth(BaseModel):
    countryCode: str
    observedViews: int = Field(ge=0)
    predictedNext7DaysViews: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    explanation: str


class EarningsPrediction(BaseModel):
    status: Literal["INSUFFICIENT_DATA", "ESTIMATED"]
    estimatedGrossAmountCents: Optional[int] = Field(default=None, ge=0)
    estimatedCreatorNetAmountCents: Optional[int] = Field(default=None, ge=0)
    currency: Literal["USD"]
    confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=1, max_length=500)


class EarningsAnomaly(BaseModel):
    status: Literal["INSUFFICIENT_DATA", "ASSESSED"]
    detected: bool
    confidence: float = Field(ge=0, le=1)
    direction: Literal["HIGH", "LOW", "NONE", "UNKNOWN"]
    explanation: str = Field(min_length=1, max_length=500)


class SuspiciousPattern(BaseModel):
    submissionId: str
    status: Literal["REVIEW_RECOMMENDED"]
    confidence: float = Field(ge=0, le=1)
    reasonCodes: List[str]
    explanation: str
    requiresHumanReview: Literal[True]


class BotFakeViewAssessment(BaseModel):
    status: Literal["REVIEW_RECOMMENDED", "INSUFFICIENT_DATA", "NO_STRONG_SIGNAL"]
    confidence: float = Field(ge=0, le=1)
    reasonCodes: List[str]
    explanation: str


class GeographicAudience(BaseModel):
    status: Literal["AVAILABLE", "INSUFFICIENT_DATA"]
    countries: List[CountryAudience]
    confidence: float = Field(ge=0, le=1)
    explanation: str


class CountryGrowthPrediction(BaseModel):
    status: Literal["ESTIMATED", "INSUFFICIENT_DATA"]
    countries: List[CountryGrowth]
    explanation: str


class NonTargetGeographicAnomaly(BaseModel):
    status: Literal["ASSESSED", "INSUFFICIENT_DATA"]
    detected: bool
    outOfTargetViews: int = Field(ge=0)
    isFraud: Literal[False]
    confidence: float = Field(ge=0, le=1)
    explanation: str


class ViewsAnalysisResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "campaignId": "66a111111111111111111111", "windowDays": 7,
        "observedViews": 500, "validatedViewsInWindow": 0, "pendingValidationViewsInWindow": 480,
        "flaggedViewsPendingReview": 20, "maturedValidatedViews": 4000, "confirmedGrossEarningsCents": 2000, "confirmedCreatorNetEarningsCents": 1900,
        "suspiciousPatterns": [{"submissionId": "66b111111111111111111111", "status": "REVIEW_RECOMMENDED", "confidence": 0.65, "reasonCodes": ["RAPID_VIEW_GROWTH"], "explanation": "A rapid-growth tracking flag requires a human review and does not prove fraud.", "requiresHumanReview": True}],
        "botAndFakeViewAssessment": {"status": "REVIEW_RECOMMENDED", "confidence": 0.65, "reasonCodes": ["RAPID_VIEW_GROWTH"], "explanation": "Signals suggest review; they do not establish bot activity."},
        "abnormalViewSpike": {"status": "ASSESSED", "detected": False, "confidence": 0.6, "explanation": "No robust daily spike was found."},
        "suddenViewDrop": {"status": "ASSESSED", "detected": False, "confidence": 0.6, "explanation": "No qualifying daily decline was observed."},
        "futureViewPrediction": {"status": "ESTIMATED", "predictedNext7DaysViews": 700, "confidence": 0.63, "explanation": "Trend and recent daily mean estimate."},
        "geographicAudience": {"status": "AVAILABLE", "countries": [{"countryCode": "JOR", "views": 450, "targeted": True}, {"countryCode": "USA", "views": 50, "targeted": False}], "confidence": 0.7, "explanation": "Incremental country deltas only."},
        "countryGrowthPrediction": {"status": "INSUFFICIENT_DATA", "countries": [], "explanation": "At least four daily buckets are required."},
        "nonTargetGeographicAnomaly": {"status": "ASSESSED", "detected": True, "outOfTargetViews": 50, "isFraud": False, "confidence": 0.65, "explanation": "Targeting mismatch is not proof of fraud."},
        "earningsPrediction": {"status": "ESTIMATED", "estimatedGrossAmountCents": 350, "estimatedCreatorNetAmountCents": 332, "currency": "USD", "confidence": 0.45, "explanation": "Advisory estimate from matured ledger history and the seven-day forecast; not withdrawable."},
        "earningsAnomaly": {"status": "INSUFFICIENT_DATA", "detected": False, "confidence": 0.0, "direction": "UNKNOWN", "explanation": "At least six matured daily ledger observations are required."},
    }]})
    campaignId: str
    windowDays: Literal[7]
    observedViews: int = Field(ge=0)
    validatedViewsInWindow: int = Field(ge=0)
    pendingValidationViewsInWindow: int = Field(ge=0)
    flaggedViewsPendingReview: int = Field(ge=0)
    maturedValidatedViews: int = Field(ge=0)
    confirmedGrossEarningsCents: int = Field(ge=0)
    confirmedCreatorNetEarningsCents: int = Field(ge=0)
    suspiciousPatterns: List[SuspiciousPattern]
    botAndFakeViewAssessment: BotFakeViewAssessment
    abnormalViewSpike: SignalAssessment
    suddenViewDrop: SignalAssessment
    futureViewPrediction: ViewPrediction
    geographicAudience: GeographicAudience
    countryGrowthPrediction: CountryGrowthPrediction
    nonTargetGeographicAnomaly: NonTargetGeographicAnomaly
    earningsPrediction: EarningsPrediction
    earningsAnomaly: EarningsAnomaly


@router.post(
    "/analyze",
    response_model=ViewsAnalysisResponse,
    status_code=200,
    dependencies=[Depends(require_service_key)],
    summary="Analyze seven-day views, geography, and earnings history",
    description=("Consumes only real view-tracking snapshots from the Main Backend. View trend inputs are constrained to the most recent seven days. "
                 "Matured validated view and earnings aggregates come from ledger records after the existing seven-day hold. Geo mismatch is an anomaly, never a fraud finding; estimates are not confirmed earnings."),
    responses={401: {"description": "Missing or invalid service key."}, 422: {"description": "Invalid snapshot or history payload."}},
)
async def analyze_views(body: ViewsAnalysisRequest):
    payload = body.model_dump(mode="json")
    result = analyze_view_history(payload)
    return {"campaignId": body.campaignId, **result}
