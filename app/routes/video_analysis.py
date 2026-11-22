"""Internal asynchronous video upload and analysis endpoints."""

import hashlib
import logging
import os
import tempfile
from typing import List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlmodel import Session, select
from sqlalchemy.exc import IntegrityError

from app.db.database import engine
from app.db.models import VideoAnalysisJob
from app.routes.reviews import require_service_key
from app.services.video_analysis_service import MAX_VIDEO_BYTES, process_video_job, public_job


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ai/intelligence/videos", tags=["AI Video Analysis"])


class VideoAnalysisAccepted(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "analysisId": "f1744f70-698e-4ab8-a0e9-f40147461d8f", "submissionId": "66b111111111111111111111", "status": "QUEUED",
        "result": None, "errorCode": None, "createdAt": "2026-11-10T10:00:00Z", "updatedAt": "2026-11-10T10:00:00Z",
    }]})
    analysisId: str
    submissionId: str
    status: str
    result: Optional[dict] = None
    errorCode: Optional[str] = None
    createdAt: str
    updatedAt: str


class VideoQualityResult(BaseModel):
    score: int
    rank: Literal["POOR", "FAIR", "GOOD", "EXCELLENT"]
    resolutionScore: int
    sharpnessScore: int
    brightnessScore: int
    explanation: str


class VideoSceneResult(BaseModel):
    timestampSec: float
    endSec: float
    description: str
    importance: Literal["IMPORTANT"]


class SimilarVideoResult(BaseModel):
    analysisId: str
    submissionId: str
    similarity: float
    type: Literal["SIMILAR_FRAMES"]


class VideoAnalysisResult(BaseModel):
    submissionId: str
    durationSec: float
    dimensions: dict
    videoQuality: VideoQualityResult
    similarVideos: List[SimilarVideoResult]
    summary: Optional[str] = None
    importantScenes: List[VideoSceneResult]
    editRequired: bool
    editReasons: List[str]
    rejectionReasonSuggestions: List[str]
    confidence: float
    explanation: str


class VideoAnalysisJobResponse(VideoAnalysisAccepted):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"analysisId": "f1744f70-698e-4ab8-a0e9-f40147461d8f", "submissionId": "66b111111111111111111111", "status": "QUEUED", "result": None, "errorCode": None, "createdAt": "2026-11-10T10:00:00Z", "updatedAt": "2026-11-10T10:00:00Z"},
        {"analysisId": "f1744f70-698e-4ab8-a0e9-f40147461d8f", "submissionId": "66b111111111111111111111", "status": "COMPLETED", "errorCode": None, "createdAt": "2026-11-10T10:00:00Z", "updatedAt": "2026-11-10T10:00:12Z", "result": {
            "submissionId": "66b111111111111111111111", "durationSec": 24.5, "dimensions": {"width": 720, "height": 1280},
            "videoQuality": {"score": 84, "rank": "GOOD", "resolutionScore": 100, "sharpnessScore": 70, "brightnessScore": 80, "explanation": "Computer-vision score combines sampled frame resolution, Laplacian sharpness, and exposure. It does not assess audio or campaign suitability."},
            "similarVideos": [], "summary": "A creator demonstrates the product and its main use.",
            "importantScenes": [{"timestampSec": 2.0, "endSec": 5.0, "description": "The product is shown in use.", "importance": "IMPORTANT"}],
            "editRequired": False, "editReasons": [], "rejectionReasonSuggestions": [], "confidence": 0.82,
            "explanation": "The sampled frames clearly show the product and the primary scene transition.",
        }},
    ]})
    status: Literal["QUEUED", "PROCESSING", "COMPLETED", "NEEDS_REVIEW", "FAILED"]
    result: Optional[VideoAnalysisResult] = None


def _iso_job(job: VideoAnalysisJob) -> dict:
    value = public_job(job)
    value["createdAt"] = value["createdAt"].isoformat()
    value["updatedAt"] = value["updatedAt"].isoformat()
    return value


@router.post(
    "/analyze",
    response_model=VideoAnalysisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_service_key)],
    summary="Queue a video for asynchronous analysis",
    description=("Accepts a bounded video upload from Main Backend, extracts up to eight evenly spaced frames, computes technical quality and frame similarity, "
                 "then makes one structured Gemini multimodal call for summary, scenes, editing advice, and non-binding rejection reason suggestions. "
                 "Exact duplicates in the same tenant reuse the stored job. The uploaded source is deleted after processing. This endpoint never makes an automatic review decision."),
    openapi_extra={"requestBody": {"content": {"multipart/form-data": {"examples": {"queueVideo": {"summary": "Queue a creator video", "value": {"tenantId": "creator-1", "submissionId": "66b111111111111111111111", "file": "[binary video file, maximum 100 MiB]"}}}}}}},
    responses={401: {"description": "Missing or invalid service key."}, 413: {"description": "Video exceeds the 100 MiB limit."}, 415: {"description": "Unsupported media type."}, 422: {"description": "Invalid tenant, submission, or undecodable media."}},
)
async def queue_video_analysis(
    background_tasks: BackgroundTasks,
    tenantId: str = Form(min_length=1, max_length=100, description="Main Backend owner ID used for job isolation."),
    submissionId: str = Form(min_length=1, max_length=100, description="Main Backend submission ID."),
    file: UploadFile = File(..., description="Video file, maximum 100 MiB."),
):
    if not (tenantId.replace("-", "").replace("_", "").isalnum() and submissionId.replace("-", "").replace("_", "").isalnum()):
        raise HTTPException(status_code=422, detail="Tenant and submission identifiers must be alphanumeric identifiers.")
    if file.content_type and not (file.content_type.startswith("video/") or file.content_type == "application/octet-stream"):
        raise HTTPException(status_code=415, detail="A video media type is required.")

    digest = hashlib.sha256()
    file_bytes = 0
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="ppv-video-", suffix=".upload", delete=False) as destination:
            temp_path = destination.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_bytes += len(chunk)
                if file_bytes > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the 100 MiB limit.")
                digest.update(chunk)
                destination.write(chunk)
        if file_bytes == 0:
            raise HTTPException(status_code=422, detail="An empty video cannot be analyzed.")
        sha256 = digest.hexdigest()
        with Session(engine) as session:
            existing = session.exec(select(VideoAnalysisJob).where(
                VideoAnalysisJob.tenant_id == tenantId,
                VideoAnalysisJob.content_sha256 == sha256,
            )).first()
            if existing:
                os.unlink(temp_path)
                temp_path = None
                return _iso_job(existing)
            job = VideoAnalysisJob(tenant_id=tenantId, submission_id=submissionId, content_sha256=sha256)
            session.add(job)
            try:
                session.commit()
                session.refresh(job)
            except IntegrityError:
                session.rollback()
                existing = session.exec(select(VideoAnalysisJob).where(
                    VideoAnalysisJob.tenant_id == tenantId,
                    VideoAnalysisJob.content_sha256 == sha256,
                )).first()
                if existing:
                    os.unlink(temp_path)
                    temp_path = None
                    return _iso_job(existing)
                raise
        background_tasks.add_task(process_video_job, job.id, temp_path)
        temp_path = None
        return _iso_job(job)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Video analysis queue failed (type=%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Video analysis could not be queued.") from None
    finally:
        await file.close()
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


@router.get(
    "/{analysis_id}",
    response_model=VideoAnalysisJobResponse,
    dependencies=[Depends(require_service_key)],
    summary="Get asynchronous video analysis status",
    description="Returns a job only when the supplied tenantId owns it. Status is QUEUED, PROCESSING, COMPLETED, NEEDS_REVIEW, or FAILED.",
    responses={401: {"description": "Missing or invalid service key."}, 404: {"description": "Analysis job not found for this tenant."}},
)
def get_video_analysis(analysis_id: str, tenantId: str):
    with Session(engine) as session:
        job = session.exec(select(VideoAnalysisJob).where(
            VideoAnalysisJob.id == analysis_id,
            VideoAnalysisJob.tenant_id == tenantId,
        )).first()
        if not job:
            raise HTTPException(status_code=404, detail="Video analysis job not found.")
        return _iso_job(job)
