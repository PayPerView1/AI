import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from app.core.config import get_settings
from app.db.database import init_db
from app.rag.vector_store import check_vector_store
from app.routes.chat import router as chat_router
from app.routes.reviews import router as reviews_router
from app.routes.intelligence import router as intelligence_router
from app.routes.view_intelligence import router as view_intelligence_router
from app.routes.video_analysis import router as video_analysis_router
from app.services.video_analysis_service import recover_interrupted_video_jobs


settings = get_settings()
logger = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {
        "name": "AI Chatbot",
        "description": "Authenticated PayPerView chat. Thread and message data are scoped to the caller's Express user ID; message generation is synchronous and rate limited.",
    },
    {
        "name": "AI Review",
        "description": "Server-to-server review endpoints secured by X-Service-Key. Campaign brief review is synchronous. Submission text checks use deterministic validation and Gemini; media review needs trusted sampled frames, which the current Main Backend integration does not send, so media-dependent decisions remain manual.",
    },
    {"name": "AI Campaign Intelligence", "description": "Service-key protected campaign draft suggestions, compliance checks, classification, similarity, and historical performance analysis."},
    {"name": "AI Budget Intelligence", "description": "Service-key protected statistical payment, budget, spending, and refund recommendations based only on supplied account history."},
    {"name": "AI Wallet and Withdrawal Intelligence", "description": "Service-key protected advisory withdrawal behavior and country-method recommendations using creator-scoped history supplied by Main Backend."},
    {"name": "AI Language Services", "description": "Service-key protected bounded Arabic-English translation and no-model-call language detection."},
    {"name": "AI Recommendations", "description": "Service-key protected content similarity, audience preference, and aggregate collaborative ranking."},
    {"name": "AI Report Intelligence", "description": "Service-key protected report categorization, review urgency, same-target duplicate candidates, and non-binding reviewer suggestions. Every result remains human-reviewed."},
    {"name": "AI Content Review", "description": "Service-key protected deterministic keyword and hashtag checks, brief alignment, and optional visual analysis of trusted frames."},
    {"name": "AI Views and Earnings Intelligence", "description": "Statistical analysis of seven-day Main Backend view snapshots and matured confirmed earnings; geo mismatches are not fraud findings and estimates are not ledger earnings."},
    {"name": "AI Video Analysis", "description": "Tenant-scoped asynchronous video upload, sampled-frame Computer Vision and Gemini analysis, deduplication, and persisted result polling. Suggestions do not auto-approve or auto-reject content."},
    {
        "name": "Service checks",
        "description": "Liveness and readiness checks. Readiness reports configuration and Chroma availability without a paid model call.",
    },
]


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"status": "ok", "service": "PayPerView AI Chatbot Service", "version": "1.0.0"}]
    })
    status: Literal["ok"]
    service: str
    version: str


class ReadinessChecks(BaseModel):
    gemini_configured: bool
    jwt_configured: bool
    backend_configured: bool
    service_key_configured: bool
    chroma_ready: bool


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "status": "configuration_ready",
            "checks": {
                "gemini_configured": True,
                "jwt_configured": True,
                "backend_configured": True,
                "service_key_configured": True,
                "chroma_ready": True,
            },
            "collection_documents": 42,
        }]
    })
    status: Literal["configuration_ready"]
    checks: ReadinessChecks
    collection_documents: Optional[int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    interrupted = recover_interrupted_video_jobs()
    if interrupted:
        logger.warning("Marked %s interrupted video analysis job(s) as failed", interrupted)
    logger.info("AI service database initialized")
    yield
    logger.info("AI service stopped")


app = FastAPI(
    title="PayPerView Platform - AI Chatbot Service",
    description="Authenticated PayPerView AI chat with Gemini, user-owned context, role-scoped Main Backend tools, feedback-adapted responses, suggested actions, and Chroma retrieval. Internal intelligence APIs use X-Service-Key.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(chat_router)
app.include_router(reviews_router)
app.include_router(intelligence_router)
app.include_router(view_intelligence_router)
app.include_router(video_analysis_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service checks"],
    summary="Check AI Service liveness",
    responses={200: {"description": "The process is running."}},
)
async def health_check():
    """Liveness probe; dependency readiness is reported separately."""
    return {
        "status": "ok",
        "service": "PayPerView AI Chatbot Service",
        "version": "1.0.0",
    }


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["Service checks"],
    summary="Check AI Service readiness",
    responses={503: {
        "description": "One or more required settings or Chroma are not ready.",
        "content": {"application/json": {"example": {
            "detail": {
                "status": "not_configured",
                "checks": {
                    "gemini_configured": False,
                    "jwt_configured": True,
                    "backend_configured": True,
                    "service_key_configured": True,
                    "chroma_ready": False,
                },
                "collection_documents": None,
            }
        }}},
    }},
)
async def readiness_check():
    """Check required configuration and local collection availability without a paid model call."""
    collection_ready = False
    collection_count = None
    if settings.gemini_api_key.strip():
        collection_ready, collection_count = check_vector_store()

    checks = {
        "gemini_configured": bool(settings.gemini_api_key.strip()),
        "jwt_configured": bool(settings.jwt_secret.strip()),
        "backend_configured": bool(settings.backend_url.strip()),
        "service_key_configured": bool(settings.ai_service_api_key.strip()),
        "chroma_ready": collection_ready,
    }
    ready = all(checks.values())
    response = {
        "status": "configuration_ready" if ready else "not_configured",
        "checks": checks,
        "collection_documents": collection_count,
    }
    if not ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=response)
    return response
