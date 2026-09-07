from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional

from app.services.review_service import review_campaign

router = APIRouter(tags=["Campaign Review"])


# ─── Pydantic Schemas (mirror AI_SERVICE_CONTRACT exactly) ───────────────────

class CampaignBrief(BaseModel):
    mainIdea: str = Field(..., description="الفكرة الرئيسية للحملة (مطلوبة)")
    tone: Optional[str] = Field("", description="نبرة الحملة وأسلوبها")
    keyMessages: Optional[str] = Field("", description="الرسائل المحورية")
    keywords: Optional[List[str]] = Field(default_factory=list, description="الكلمات المفتاحية")
    visualReferences: Optional[str] = Field("", description="المراجع البصرية")


class HalalDeclaration(BaseModel):
    noGambling: bool = Field(True, description="لا قمار")
    noSexualContent: bool = Field(True, description="لا محتوى جنسي")
    noExplicitMusic: bool = Field(True, description="لا موسيقى صريحة")
    noAlcohol: bool = Field(True, description="لا كحول")
    noSuspiciousCurrencies: bool = Field(True, description="لا عملات مشبوهة")
    noUnrealisticProfit: bool = Field(True, description="لا أرباح غير واقعية")


class ReviewRequest(BaseModel):
    campaignName: str = Field(..., description="اسم الحملة")
    brief: CampaignBrief
    targetCountries: Optional[List[str]] = Field(default_factory=list, description="أكواد الدول ISO-3")
    halalDeclaration: HalalDeclaration = Field(default_factory=HalalDeclaration)


class ReviewResponse(BaseModel):
    result: str = Field(..., description="APPROVED | REJECTED | MANUAL_REVIEW_REQUIRED")
    score: int = Field(..., ge=0, le=100, description="درجة التوافق مع الحلال (0-100)")
    feedback: str = Field(..., description="تعليق توضيحي على القرار")


# ─── POST /review ─────────────────────────────────────────────────────────────

@router.post(
    "/review",
    response_model=ReviewResponse,
    summary="مراجعة حملة إعلانية وفق معايير الحلال",
    description=(
        "يستقبل بيانات الحملة من الـ Backend ويُعيد قراراً فورياً بالموافقة أو الرفض "
        "أو الإحالة للمراجعة اليدوية، مع درجة توافق ومبرر مفصّل."
    ),
)
async def review_campaign_endpoint(body: ReviewRequest) -> ReviewResponse:
    """
    POST /review — Internal endpoint called by the Node.js backend.
    No JWT required: requests originate from a trusted internal service.
    """
    payload = body.model_dump()
    result = await review_campaign(payload)
    return ReviewResponse(**result)
