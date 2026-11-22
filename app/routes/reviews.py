import hmac
import logging
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.services.chat_service import get_llm


router = APIRouter(tags=["AI Review"])
logger = logging.getLogger(__name__)
service_key_header = APIKeyHeader(
    name="X-Service-Key",
    scheme_name="ServiceKey",
    description="Server-to-server credential configured in AI_SERVICE_API_KEY. Never send this credential from a browser.",
    auto_error=False,
)

REVIEW_AUTH_ERRORS = {
    401: {"description": "The X-Service-Key header is missing or invalid.", "content": {"application/json": {"example": {"detail": "Invalid service credentials."}}}},
    503: {"description": "The AI review service key is not configured.", "content": {"application/json": {"example": {"detail": "AI review service authentication is not configured."}}}},
}


class ReviewResult(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


class CampaignBrief(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={
        "examples": [{
            "mainIdea": "Show how a reusable bottle fits into a daily routine.",
            "tone": "Practical and friendly",
            "keyMessages": "Reusable; easy to clean",
            "keywords": ["reusable", "daily routine"],
            "visualReferences": "Bright kitchen and outdoor scenes",
        }]
    })
    mainIdea: str = Field(..., min_length=1, max_length=5000)
    tone: str = Field(default="", max_length=500)
    keyMessages: str = Field(default="", max_length=3000)
    keywords: List[str] = Field(default_factory=list, max_length=50)
    visualReferences: str = Field(default="", max_length=3000)


class HalalDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{
            "noGambling": True,
            "noSexualContent": True,
            "noExplicitMusic": True,
            "noAlcohol": True,
            "noSuspiciousCurrencies": True,
            "noUnrealisticProfit": True,
        }]
    })
    noGambling: bool
    noSexualContent: bool
    noExplicitMusic: bool
    noAlcohol: bool
    noSuspiciousCurrencies: bool
    noUnrealisticProfit: bool


class CampaignReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{
            "campaignName": "Reusable bottle launch",
            "brief": {"mainIdea": "Show how a reusable bottle fits into a daily routine.", "tone": "Practical and friendly"},
            "targetCountries": ["JOR"],
            "halalDeclaration": {
                "noGambling": True, "noSexualContent": True, "noExplicitMusic": True,
                "noAlcohol": True, "noSuspiciousCurrencies": True, "noUnrealisticProfit": True,
            },
        }]
    })
    campaignName: str = Field(..., min_length=1, max_length=255)
    brief: CampaignBrief
    targetCountries: List[str] = Field(default_factory=list, max_length=100)
    halalDeclaration: HalalDeclaration


class CampaignReviewResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "result": "MANUAL_REVIEW_REQUIRED", "score": 70,
            "feedback": "The brief is understandable, but a reviewer should confirm the remaining policy details.",
        }]
    })
    result: ReviewResult
    score: int = Field(..., ge=0, le=100)
    feedback: str = Field(..., min_length=1, max_length=1000)


class SubmissionMediaFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mimeType: Literal["image/jpeg", "image/png", "image/webp"]
    dataBase64: str = Field(min_length=100, max_length=2_800_000)
    timestampSec: float = Field(ge=0, le=3600)


class SubmissionPostData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    caption: str = Field(default="", max_length=5000)
    hashtags: List[str] = Field(default_factory=list, max_length=50)
    mentions: List[str] = Field(default_factory=list, max_length=50)
    authorHandle: str = Field(default="", max_length=255)
    durationSec: Optional[float] = Field(default=None, ge=0, le=7200)
    thumbnailUrl: str = Field(default="", max_length=2048)
    fetchStatus: Literal["OK", "UNAVAILABLE", "NOT_FOUND", "PRIVATE"] = "UNAVAILABLE"
    aspectRatio: Optional[str] = Field(default=None, max_length=20)


class SubmissionCampaignBrief(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mainIdea: str = Field(default="", max_length=5000)
    tone: str = Field(default="", max_length=500)
    keyMessages: str = Field(default="", max_length=3000)
    keywords: List[str] = Field(default_factory=list, max_length=50)
    visualReferences: str = Field(default="", max_length=3000)


class SubmissionRequirements(BaseModel):
    model_config = ConfigDict(extra="ignore")
    requiredKeywords: List[str] = Field(default_factory=list, max_length=50)
    requiredHashtags: List[str] = Field(default_factory=list, max_length=50)
    requiredMentions: List[str] = Field(default_factory=list, max_length=50)
    aspectRatio: Optional[str] = Field(default=None, max_length=20)
    minDurationSec: Optional[float] = Field(default=None, ge=0, le=7200)
    maxDurationSec: Optional[float] = Field(default=None, ge=0, le=7200)
    shariaNote: str = Field(default="", max_length=1000)


class SubmissionCampaignData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(default="", max_length=255)
    brief: SubmissionCampaignBrief = Field(default_factory=SubmissionCampaignBrief)
    requirements: SubmissionRequirements = Field(default_factory=SubmissionRequirements)
    halalDeclaration: Optional[Dict[str, Any]] = None


class SubmissionReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{
            "submissionId": "66a222222222222222222222",
            "platform": "YOUTUBE",
            "postUrl": "https://www.youtube.com/shorts/abcdefghijk",
            "post": {"caption": "A short product demonstration"},
            "campaign": {"name": "Reusable bottle launch", "requirements": ["Show the product in use"]},
            "backendChecks": [],
        }]
    })
    submissionId: str = Field(..., min_length=1, max_length=100)
    platform: str = Field(..., min_length=1, max_length=30)
    postUrl: str = Field(..., max_length=2048)
    post: SubmissionPostData
    campaign: SubmissionCampaignData
    backendChecks: List[Dict[str, Any]] = Field(default_factory=list, max_length=30)
    mediaFrames: List[SubmissionMediaFrame] = Field(default_factory=list, max_length=6, description="Trusted sampled frames. The Main Backend does not currently populate this field.")
    brandName: Optional[str] = Field(default=None, max_length=255)


class ReviewCheck(BaseModel):
    code: str
    status: str
    message: str
    fixHint: str = ""
    score: Optional[int] = Field(None, ge=0, le=100)


class SubmissionReviewResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "result": "MANUAL_REVIEW_REQUIRED",
            "score": None,
            "feedback": "Media evidence is not available to the review service; a human must verify the submitted content.",
            "checks": [
                {"code": "VIDEO_FORMAT", "status": "NEEDS_REVIEW", "message": "Video media was not supplied for inspection.", "fixHint": ""},
                {"code": "AUDIO_COMPLIANCE", "status": "NEEDS_REVIEW", "message": "Audio media was not supplied for inspection.", "fixHint": ""},
            ],
            "manualReasonCode": "MEDIA_EVIDENCE_UNAVAILABLE",
        }]
    })
    result: ReviewResult
    score: Optional[int] = Field(None, ge=0, le=100)
    feedback: str = Field(..., min_length=1, max_length=1000)
    checks: List[ReviewCheck]
    manualReasonCode: Optional[str] = None


async def require_service_key(x_service_key: Optional[str] = Security(service_key_header)) -> None:
    expected = get_settings().ai_service_api_key
    if not expected:
        raise HTTPException(status_code=503, detail="AI review service authentication is not configured.")
    if not x_service_key or not hmac.compare_digest(x_service_key, expected):
        raise HTTPException(status_code=401, detail="Invalid service credentials.")


@router.post(
    "/review",
    response_model=CampaignReviewResponse,
    dependencies=[Depends(require_service_key)],
    summary="Review campaign brief text",
    description="Synchronous server-to-server Gemini review. It evaluates supplied text and declarations only; uncertain or unverifiable checks must be handled by a human reviewer.",
    responses={**REVIEW_AUTH_ERRORS, 422: {"description": "The request body failed schema validation."}, 502: {"description": "Gemini failed or returned output that did not match the response schema.", "content": {"application/json": {"example": {"detail": "Campaign review provider is unavailable."}}}}},
)
async def review_campaign(body: CampaignReviewRequest):
    """Assess supplied campaign text; uncertain or unverifiable material goes to a human."""
    required_flags = (
        "noGambling", "noSexualContent", "noExplicitMusic", "noAlcohol",
        "noSuspiciousCurrencies", "noUnrealisticProfit",
    )
    if not all(getattr(body.halalDeclaration, flag) is True for flag in required_flags):
        return CampaignReviewResponse(
            result=ReviewResult.MANUAL_REVIEW_REQUIRED,
            score=0,
            feedback="All required Halal declarations must be explicitly confirmed before approval.",
        )
    structured_llm = get_llm().with_structured_output(CampaignReviewResponse)
    try:
        result = await structured_llm.ainvoke([
            SystemMessage(content=(
                "Review a PayPerView campaign brief for explicit platform-policy concerns. "
                "Use only the provided text and declarations. APPROVED requires no evident concern. "
                "All six halal declaration flags must be true before approval; if any flag is false, return MANUAL_REVIEW_REQUIRED. "
                "Use REJECTED only for a clear, explicit violation in supplied content. "
                "Use MANUAL_REVIEW_REQUIRED whenever evidence is missing or ambiguous. "
                "Do not infer video/audio content from a brief. Keep feedback concise and factual."
            )),
            HumanMessage(content=body.model_dump_json()),
        ])
        if isinstance(result, CampaignReviewResponse):
            return result
        return CampaignReviewResponse.model_validate(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Campaign review provider failed (type=%s)", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Campaign review provider is unavailable.") from None


@router.post(
    "/api/v1/ai/submissions/review",
    response_model=SubmissionReviewResponse,
    dependencies=[Depends(require_service_key)],
    summary="Request AI-assisted submission checks",
    description="Checks required keywords and hashtags deterministically, uses Gemini for brief alignment, and Gemini Vision for supplied video frames. With no frames, the result is MANUAL_REVIEW_REQUIRED. The Main Backend currently supplies no frames, so media-dependent checks remain human reviewed.",
    responses={**REVIEW_AUTH_ERRORS, 422: {"description": "The request body failed schema validation."}, 502: {"description": "AI analysis failed."}, 503: {"description": "Gemini is not configured."}},
)
async def review_submission(body: SubmissionReviewRequest):
    """Use deterministic text checks and conservative vision checks when trusted frames exist."""
    from app.routes.intelligence import review_submission_content

    return await review_submission_content(body)
