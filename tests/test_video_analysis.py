import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine, select, Session

import app.main as main
from app.db.models import VideoAnalysisJob
from app.services import video_analysis_service as video_service
from app.services.video_analysis_service import VideoNarrativeAnalysis, ImportantScene, hash_similarity


class VideoAnalysisUnitTests(unittest.TestCase):
    def test_perceptual_hash_similarity_uses_sampled_frames(self):
        hashes = ["0" * 64, "1" * 64]
        near = ["0" * 63 + "1", "1" * 63 + "0"]
        self.assertGreater(hash_similarity(hashes, near), 0.98)
        self.assertLess(hash_similarity(hashes, ["0" * 32 + "1" * 32]), 0.8)

    def test_openapi_documents_async_video_and_view_workflows(self):
        spec = main.app.openapi()
        self.assertIn("/api/v1/ai/intelligence/videos/analyze", spec["paths"])
        self.assertIn("/api/v1/ai/intelligence/videos/{analysis_id}", spec["paths"])
        self.assertIn("/api/v1/ai/intelligence/views/analyze", spec["paths"])
        operation = spec["paths"]["/api/v1/ai/intelligence/videos/analyze"]["post"]
        self.assertEqual(operation["responses"]["202"]["description"], "Successful Response")
        self.assertIn("multipart/form-data", operation["requestBody"]["content"])
        self.assertTrue(any("ServiceKey" in item for item in operation["security"]))


class VideoAnalysisRouteTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self):
        self.engine.dispose()

    def test_upload_queues_once_and_job_is_tenant_scoped(self):
        with patch("app.main.init_db", return_value=None), \
             patch("app.main.recover_interrupted_video_jobs", return_value=0), \
             patch("app.routes.video_analysis.engine", self.engine), \
             patch("app.services.video_analysis_service.engine", self.engine), \
             patch("app.routes.reviews.get_settings", return_value=SimpleNamespace(ai_service_api_key="test-service-key")), \
             patch("app.routes.video_analysis.process_video_job", new_callable=AsyncMock) as worker:
            with TestClient(main.app) as client:
                first = client.post(
                    "/api/v1/ai/intelligence/videos/analyze",
                    data={"tenantId": "creator-1", "submissionId": "submission-1"},
                    files={"file": ("sample.mp4", b"bounded-test-video-content", "video/mp4")},
                    headers={"X-Service-Key": "test-service-key"},
                )
                self.assertEqual(first.status_code, 202, first.text)
                self.assertEqual(first.json()["status"], "QUEUED")
                first_id = first.json()["analysisId"]

                repeated = client.post(
                    "/api/v1/ai/intelligence/videos/analyze",
                    data={"tenantId": "creator-1", "submissionId": "submission-1"},
                    files={"file": ("renamed.mp4", b"bounded-test-video-content", "video/mp4")},
                    headers={"X-Service-Key": "test-service-key"},
                )
                self.assertEqual(repeated.status_code, 202, repeated.text)
                self.assertEqual(repeated.json()["analysisId"], first_id)
                self.assertEqual(worker.await_count, 1)

                owned = client.get(
                    f"/api/v1/ai/intelligence/videos/{first_id}",
                    params={"tenantId": "creator-1"},
                    headers={"X-Service-Key": "test-service-key"},
                )
                self.assertEqual(owned.status_code, 200, owned.text)
                hidden = client.get(
                    f"/api/v1/ai/intelligence/videos/{first_id}",
                    params={"tenantId": "creator-2"},
                    headers={"X-Service-Key": "test-service-key"},
                )
                self.assertEqual(hidden.status_code, 404)
                os.unlink(worker.await_args.args[1])

        with Session(self.engine) as session:
            jobs = session.exec(select(VideoAnalysisJob)).all()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, "QUEUED")

    def test_background_worker_persists_validated_gemini_and_computer_vision_result(self):
        import asyncio

        with Session(self.engine) as session:
            job = VideoAnalysisJob(tenant_id="creator-1", submission_id="submission-1", content_sha256="a" * 64)
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id
        with tempfile.NamedTemporaryFile(delete=False) as source:
            source.write(b"test source")
            source_path = source.name
        extracted = {
            "durationSec": 10.0, "width": 720, "height": 1280,
            "frames": [{"timestampSec": 0.0, "bytes": b"frame-a"}, {"timestampSec": 8.0, "bytes": b"frame-b"}],
            "frameHashes": ["0" * 64, "1" * 64],
            "technicalQuality": {"score": 84, "rank": "GOOD", "resolutionScore": 100, "sharpnessScore": 70, "brightnessScore": 80, "explanation": "Measured frame quality."},
        }
        narrative = VideoNarrativeAnalysis(
            summary="A creator demonstrates a bottle.",
            importantScenes=[ImportantScene(timestampSec=2, endSec=4, description="Bottle is shown clearly.")],
            editRequired=False, editReasons=[], rejectionReasonSuggestions=[], confidence=0.82,
            explanation="The sampled frames show the product and the scene transition clearly.",
        )
        try:
            with patch("app.services.video_analysis_service.engine", self.engine), \
                 patch("app.services.video_analysis_service.extract_video_frames", return_value=extracted), \
                 patch("app.services.video_analysis_service._find_similar", return_value=[]), \
                 patch("app.services.video_analysis_service.get_settings", return_value=SimpleNamespace(gemini_api_key="configured")), \
                 patch("app.services.video_analysis_service._narrative", new_callable=AsyncMock, return_value=narrative):
                asyncio.run(video_service.process_video_job(job_id, source_path))
            with Session(self.engine) as session:
                saved = session.get(VideoAnalysisJob, job_id)
            self.assertEqual(saved.status, "COMPLETED")
            self.assertEqual(saved.error_code, None)
            self.assertEqual(__import__("json").loads(saved.result_json)["summary"], "A creator demonstrates a bottle.")
            self.assertEqual(__import__("json").loads(saved.result_json)["videoQuality"]["rank"], "GOOD")
        finally:
            if os.path.exists(source_path):
                os.unlink(source_path)


if __name__ == "__main__":
    unittest.main()
