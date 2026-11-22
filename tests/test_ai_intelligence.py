import base64
import struct
import unittest
import zlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

import app.main as main
from app.services import intelligence_service as service


class IntelligenceHelpersTests(unittest.TestCase):
    def test_seven_day_view_analysis_separates_anomaly_from_fraud_and_earnings_estimate(self):
        now = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)
        counts = [100, 110, 90, 100, 100, 100, 1000]
        events = []
        for offset, views in enumerate(counts):
            day = now - timedelta(days=6 - offset)
            events.append({
                "submissionId": "submission-1", "capturedAt": day.isoformat(), "deltaViews": views,
                "eligibleViews": views if offset < 6 else 0,
                "reviewStatus": "REVIEW_REQUIRED" if offset == 6 else "CLEAR",
                "flags": ["RAPID_VIEW_GROWTH", "OUT_OF_TARGET_COUNTRY"] if offset == 6 else [],
                "countryDeltas": [{"countryCode": "JOR", "views": max(0, views - 10)}, {"countryCode": "USA", "views": min(10, views)}],
            })
        validated_history = [{
            "day": (now - timedelta(days=14 - offset)).date().isoformat(),
            "validatedViews": 1000,
            "confirmedGrossAmountCents": 500,
            "confirmedCreatorNetAmountCents": 450,
        } for offset in range(7)]

        result = service.analyze_view_history({
            "campaignId": "campaign-1", "cpm": 5, "remainingBudget": 1.23, "targetCountries": ["JOR"],
            "events": events, "validatedHistory": validated_history,
        }, now=now)

        self.assertEqual(result["windowDays"], 7)
        self.assertEqual(result["observedViews"], sum(counts))
        self.assertTrue(result["abnormalViewSpike"]["detected"])
        self.assertEqual(result["botAndFakeViewAssessment"]["status"], "REVIEW_RECOMMENDED")
        self.assertEqual(result["nonTargetGeographicAnomaly"]["isFraud"], False)
        self.assertGreater(result["nonTargetGeographicAnomaly"]["outOfTargetViews"], 0)
        self.assertEqual(result["earningsPrediction"]["status"], "ESTIMATED")
        self.assertLessEqual(result["earningsPrediction"]["estimatedGrossAmountCents"], 123)
        self.assertLessEqual(result["earningsPrediction"]["estimatedCreatorNetAmountCents"], result["earningsPrediction"]["estimatedGrossAmountCents"])
        self.assertIn("not confirmed", result["earningsPrediction"]["explanation"])

    def test_view_analysis_does_not_invent_estimates_without_matured_ledger_history(self):
        result = service.analyze_view_history({"events": [], "validatedHistory": [], "cpm": 5, "targetCountries": []})
        self.assertEqual(result["futureViewPrediction"]["status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["earningsPrediction"]["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(result["earningsPrediction"]["estimatedGrossAmountCents"])
        self.assertEqual(result["earningsAnomaly"]["status"], "INSUFFICIENT_DATA")

    def test_matured_earnings_yield_anomaly_is_advisory_and_two_sided(self):
        stable_days = [
            {"validatedViews": 1000, "confirmedGrossAmountCents": 500}
            for _ in range(5)
        ]
        high = service.analyze_view_history({
            "events": [],
            "validatedHistory": stable_days + [{"validatedViews": 1000, "confirmedGrossAmountCents": 2000}],
        })["earningsAnomaly"]
        low = service.analyze_view_history({
            "events": [],
            "validatedHistory": stable_days + [{"validatedViews": 1000, "confirmedGrossAmountCents": 100}],
        })["earningsAnomaly"]

        self.assertEqual((high["status"], high["detected"], high["direction"]), ("ASSESSED", True, "HIGH"))
        self.assertEqual((low["status"], low["detected"], low["direction"]), ("ASSESSED", True, "LOW"))
        self.assertLessEqual(high["confidence"], 0.75)
        self.assertIn("not evidence", high["explanation"])

    def test_geographic_mismatch_alone_does_not_trigger_bot_or_fake_view_suspicion(self):
        now = datetime.now(timezone.utc)
        result = service.analyze_view_history({
            "targetCountries": ["JOR"],
            "events": [{
                "submissionId": "submission-geo", "capturedAt": now.isoformat(), "deltaViews": 50,
                "eligibleViews": 0, "reviewStatus": "REVIEW_REQUIRED", "flags": ["OUT_OF_TARGET_COUNTRY"],
                "countryDeltas": [{"countryCode": "USA", "views": 50}],
            }],
        }, now=now)
        self.assertEqual(result["suspiciousPatterns"], [])
        self.assertEqual(result["botAndFakeViewAssessment"]["status"], "NO_STRONG_SIGNAL")
        self.assertTrue(result["nonTargetGeographicAnomaly"]["detected"])
        self.assertFalse(result["nonTargetGeographicAnomaly"]["isFraud"])

    def test_campaign_prediction_and_category_analytics_require_real_samples(self):
        history = [
            {"id": str(index), "category": "UGC", "stats": {"totalViews": views, "totalSpent": 10}}
            for index, views in enumerate([100, 120, 80])
        ]
        self.assertEqual(service.campaign_performance({"totalBudget": 100}, history, [] )["status"], "INSUFFICIENT_DATA")
        matches = [{"id": item["id"]} for item in history]
        result = service.campaign_performance({"totalBudget": 100}, history, matches)
        self.assertEqual(result["status"], "ESTIMATED")
        self.assertEqual(result["predictedViews"], 1000)
        self.assertEqual(service.category_analytics(history)["categories"][0]["category"], "UGC")

    def test_payment_recommendation_respects_configured_country_methods_and_original_refund_method(self):
        transactions = [{"type": "CREDIT", "status": "COMPLETED", "amount": 100, "paymentMethod": "PAYPAL"}] * 5
        result = service.budget_analysis({
            "transactions": transactions,
            "availablePaymentMethods": ["PAYPAL"],
            "paymentMethodsByCountry": {"JO": ["MOYASAR"]},
            "country": "JO",
            "refunds": [{"refundMethod": "MOYASAR"}] * 6,
            "refundOriginalMethod": None,
        })
        self.assertEqual(result["paymentMethodRecommendation"]["status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["refundMethodRecommendation"]["status"], "INSUFFICIENT_DATA")

        result = service.budget_analysis({"refundOriginalMethod": "PAYPAL"})
        self.assertEqual(result["refundMethodRecommendation"]["method"], "PAYPAL")

    def test_budget_features_use_only_supplied_history_and_return_statistical_estimates(self):
        history = [{"type": "CREDIT", "status": "COMPLETED", "amount": 100, "paymentMethod": "PAYPAL"}] * 5
        result = service.budget_analysis({
            "transactions": history,
            "availablePaymentMethods": ["PAYPAL"], "paymentMethodsByCountry": {"JO": ["PAYPAL"]}, "country": "JO",
            "candidateDeposit": {"amount": 1000}, "refundOriginalMethod": "PAYPAL",
            "campaigns": [{"totalBudget": amount} for amount in [100, 200, 300]],
            "dailySpending": [10, 10, 10, 10, 10, 1000], "campaign": {"remainingBudget": 25},
            "refunds": [{"status": "REJECTED", "refundMethod": "PAYPAL"}] * 4 + [{"status": "APPROVED", "refundMethod": "PAYPAL"}],
        })
        self.assertEqual(result["paymentFraud"]["risk"], "MEDIUM")
        self.assertEqual(result["paymentMethodRecommendation"]["method"], "PAYPAL")
        self.assertEqual(result["depositAmountPrediction"]["amount"], 200)
        self.assertEqual(result["dailyBudgetRecommendation"]["recommendedLimit"], 12)
        self.assertEqual(result["budgetExhaustionPrediction"]["daysRemaining"], 2.5)
        self.assertEqual(result["spendingAnomaly"]["risk"], "HIGH")
        self.assertEqual(result["refundFraud"]["risk"], "HIGH")
        self.assertEqual(result["refundLikelihoodPrediction"]["probability"], 1.0)
        self.assertEqual(result["refundMethodRecommendation"]["method"], "PAYPAL")

    def test_embedding_duplicate_detection_returns_only_threshold_matches(self):
        candidates = [{"id": "similar"}, {"id": "different"}]
        matches = service.embedding_matches("draft", candidates, [[1, 0], [0.9, 0.1], [0, 1]], threshold=0.78)
        self.assertEqual([item["id"] for item in matches], ["similar"])

    def test_daily_spend_anomaly_requires_five_prior_complete_day_observations(self):
        five = service.budget_analysis({"dailySpending": [10, 10, 10, 10, 1000]})
        six = service.budget_analysis({"dailySpending": [10, 10, 10, 10, 10, 1000]})
        self.assertEqual(five["spendingAnomaly"]["status"], "INSUFFICIENT_DATA")
        self.assertEqual(six["spendingAnomaly"]["risk"], "HIGH")

    def test_keyword_matching_uses_word_boundaries_and_normalizes_arabic_diacritics(self):
        self.assertFalse(service.contains_phrase("partial agreement", "art"))
        self.assertTrue(service.contains_phrase("show the reusable bottle", "reusable bottle"))
        self.assertTrue(service.contains_phrase("كِتاب جديد", "كتاب"))

    def test_withdrawal_anomalies_recommendations_and_processing_estimate_use_owner_history(self):
        now = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)
        history = [{
            "amountCents": 10000, "status": "COMPLETED", "method": "PAYPAL",
            "createdAt": (now - timedelta(days=days, hours=24)).isoformat(),
            "completedAt": (now - timedelta(days=days, hours=20)).isoformat(),
        } for days in range(1, 6)]
        result = service.analyze_withdrawal_history({
            "country": "JO", "candidate": {"amountCents": 100000, "method": "PAYPAL"},
            "availableMethods": ["PAYPAL", "BANK_TRANSFER"],
            "paymentMethodsByCountry": {"JO": ["PAYPAL"]}, "withdrawals": history,
        }, now=now)
        self.assertEqual(result["risk"], "MEDIUM")
        self.assertIn("UNUSUAL_WITHDRAWAL_AMOUNT", [signal["code"] for signal in result["signals"]])
        self.assertEqual(result["processingTimePrediction"], {"status": "ESTIMATED", "medianHours": 4.0, "sampleSize": 5})
        self.assertEqual(result["paymentMethodRecommendation"]["methods"], ["PAYPAL"])
        self.assertFalse(result["requiresHumanReview"])

        no_history = service.analyze_withdrawal_history({
            "candidate": {"amountCents": 10000, "method": "PAYPAL"},
            "availableMethods": ["PAYPAL"], "paymentMethodsByCountry": {}, "withdrawals": [],
        }, now=now)
        self.assertEqual(no_history["status"], "ASSESSED")
        self.assertEqual(no_history["risk"], "UNKNOWN")
        self.assertEqual(no_history["processingTimePrediction"]["status"], "INSUFFICIENT_DATA")

    def test_campaign_business_analysis_reports_insufficient_data_and_owner_relative_patterns(self):
        history = [
            *[{"category": "UGC", "isArchived": True, "stats": {"totalViews": views, "totalSpent": 10}} for views in [300, 350, 400]],
            *[{"category": "AUDIO", "isArchived": True, "stats": {"totalViews": views, "totalSpent": 10}} for views in [100, 120, 140]],
        ]
        result = service.campaign_business_analysis(history, {"totalSubmissions": 5, "acceptedSubmissions": 3})
        self.assertEqual(result["archivedCampaignAnalysis"]["archivedCampaigns"], 6)
        self.assertEqual(result["successfulCampaignPatterns"]["patterns"][0]["category"], "UGC")
        self.assertEqual(result["policyAcceptanceAnalytics"]["acceptanceRate"], 0.6)
        self.assertTrue(result["policyAcceptanceAnalytics"]["recommendations"])
        empty = service.campaign_business_analysis([], {"totalSubmissions": 0, "acceptedSubmissions": 0})
        self.assertEqual(empty["archivedCampaignAnalysis"]["status"], "INSUFFICIENT_DATA")


class IntelligenceRouteTests(unittest.TestCase):
    def setUp(self):
        self.client_context = patch("app.main.init_db", return_value=None)
        self.client_context.start()
        self.recovery_patch = patch("app.main.recover_interrupted_video_jobs", return_value=0)
        self.recovery_patch.start()
        self.client = TestClient(main.app)
        self.client.__enter__()
        self.settings_patch = patch(
            "app.routes.reviews.get_settings",
            return_value=SimpleNamespace(ai_service_api_key="test-service-key", gemini_api_key=""),
        )
        self.settings_patch.start()
        self.intelligence_settings_patch = patch(
            "app.routes.intelligence.get_settings",
            return_value=SimpleNamespace(ai_service_api_key="test-service-key", gemini_api_key=""),
        )
        self.intelligence_settings_patch.start()

    def tearDown(self):
        self.intelligence_settings_patch.stop()
        self.settings_patch.stop()
        self.client.__exit__(None, None, None)
        self.recovery_patch.stop()
        self.client_context.stop()

    def test_budget_route_returns_validated_insufficient_data_fallback(self):
        response = self.client.post(
            "/api/v1/ai/intelligence/budget/analyze",
            json={},
            headers={"X-Service-Key": "test-service-key"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["depositAmountPrediction"]["status"], "INSUFFICIENT_DATA")
        self.assertEqual(response.json()["refundMethodRecommendation"]["method"], None)

    def test_draft_analysis_cannot_approve_when_halal_flags_are_incomplete(self):
        from app.routes import intelligence as routes
        from app.services.intelligence_service import DraftLanguageAnalysis

        result = DraftLanguageAnalysis(
            decision="APPROVED", qualityScore=80, category="UGC", subCategories=[],
            feedback="The brief is clear.", checks=[], suggestions=[],
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=result)
        with patch.object(routes, "get_settings", return_value=SimpleNamespace(ai_service_api_key="test-service-key", gemini_api_key="test-gemini-key")), patch.object(routes, "get_llm", return_value=llm):
            response = self.client.post(
                "/api/v1/ai/intelligence/campaign/draft-analysis",
                json={"draft": {"name": "Product demo", "brief": {"mainIdea": "Show product use."}}},
                headers={"X-Service-Key": "test-service-key"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["policyDecision"], "MANUAL_REVIEW_REQUIRED")
        self.assertEqual(response.json()["category"], "UGC")
        self.assertEqual(response.json()["qualityScore"], 80)

    def test_draft_analysis_rejects_failed_validated_checks_even_if_model_says_approve(self):
        from app.routes import intelligence as routes
        from app.services.intelligence_service import Check, DraftLanguageAnalysis

        result = DraftLanguageAnalysis(
            decision="APPROVED", qualityScore=80, category="UGC", subCategories=[],
            feedback="A prohibited policy check failed.",
            checks=[Check(code="POLICY", status="FAILED", message="Explicit prohibited content was found.")], suggestions=[],
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=result)
        draft = {
            "name": "Product demo", "brief": {"mainIdea": "Show product use."},
            "halalDeclaration": {
                "noGambling": True, "noSexualContent": True, "noExplicitMusic": True,
                "noAlcohol": True, "noSuspiciousCurrencies": True, "noUnrealisticProfit": True,
            },
        }
        with patch.object(routes, "get_settings", return_value=SimpleNamespace(ai_service_api_key="test-service-key", gemini_api_key="test-gemini-key")), patch.object(routes, "get_llm", return_value=llm):
            response = self.client.post(
                "/api/v1/ai/intelligence/campaign/draft-analysis", json={"draft": draft},
                headers={"X-Service-Key": "test-service-key"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["policyDecision"], "REJECTED")

    def test_content_review_fails_missing_keywords_and_rejects_spoofed_frame_mime(self):
        payload = {
            "submissionId": "submission-1", "platform": "YOUTUBE", "postUrl": "https://example.invalid/post",
            "post": {"caption": "A short caption", "hashtags": [], "fetchStatus": "OK"},
            "campaign": {"name": "Campaign", "brief": {"mainIdea": "A short caption"}, "requirements": {"requiredKeywords": ["missing"], "requiredHashtags": []}},
        }
        headers = {"X-Service-Key": "test-service-key"}
        rejected = self.client.post("/api/v1/ai/intelligence/content/review", json=payload, headers=headers)
        self.assertEqual(rejected.status_code, 200, rejected.text)
        self.assertEqual(rejected.json()["result"], "REJECTED")

        routes_settings = SimpleNamespace(ai_service_api_key="test-service-key", gemini_api_key="configured")
        payload["post"]["caption"] = "A short caption missing"
        payload["mediaFrames"] = [{"mimeType": "image/png", "dataBase64": base64.b64encode(b"x" * 120).decode(), "timestampSec": 0}]
        with patch("app.routes.intelligence.get_settings", return_value=routes_settings), patch(
            "app.routes.intelligence.get_llm"
        ) as llm_factory:
            from app.services.intelligence_service import BriefAlignment
            llm_factory.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=BriefAlignment(status="PASSED", score=80, message="The text aligns.")
            )
            invalid_frame = self.client.post("/api/v1/ai/intelligence/content/review", json=payload, headers=headers)
        self.assertEqual(invalid_frame.status_code, 422, invalid_frame.text)

    def test_content_review_approves_only_with_valid_text_and_two_vision_frames(self):
        from app.services.intelligence_service import BriefAlignment, VisualAnalysis

        def png_chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

        width = height = 16
        pixels = b"".join(b"\x00" + bytes(channel for x in range(width) for channel in (x * 11 % 256, y * 13 % 256, (x + y) * 7 % 256, 255)) for y in range(height))
        png = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0)) + png_chunk(b"IDAT", zlib.compress(pixels)) + png_chunk(b"IEND", b"")
        encoded = base64.b64encode(png).decode()
        payload = {
            "submissionId": "submission-2", "platform": "YOUTUBE", "postUrl": "https://example.invalid/post",
            "post": {"caption": "Brand bottle caption", "hashtags": ["brand"], "fetchStatus": "OK", "durationSec": 8},
            "campaign": {"name": "Brand bottle", "brief": {"mainIdea": "Brand bottle caption"}, "requirements": {"requiredKeywords": ["caption"], "requiredHashtags": ["brand"], "minDurationSec": 5, "maxDurationSec": 10}},
            "mediaFrames": [
                {"mimeType": "image/png", "dataBase64": encoded, "timestampSec": 1},
                {"mimeType": "image/png", "dataBase64": encoded, "timestampSec": 2},
            ],
            "brandName": "Brand",
        }
        align_llm = MagicMock()
        align_llm.ainvoke = AsyncMock(return_value=BriefAlignment(status="PASSED", score=90, message="The text aligns."))
        visual_llm = MagicMock()
        visual_llm.ainvoke = AsyncMock(return_value=VisualAnalysis(
            decision="APPROVED", score=88, logoDetected=True, commercialElementsDetected=True,
            videoQuality="GOOD", feedback="The sampled frames are clear and show the brand.",
        ))
        llm_factory = MagicMock()
        llm_factory.return_value.with_structured_output.side_effect = [align_llm, visual_llm]
        settings = SimpleNamespace(ai_service_api_key="test-service-key", gemini_api_key="configured")
        with patch("app.routes.intelligence.get_settings", return_value=settings), patch("app.routes.intelligence.get_llm", llm_factory):
            response = self.client.post(
                "/api/v1/ai/intelligence/content/review", json=payload,
                headers={"X-Service-Key": "test-service-key"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["result"], "APPROVED")
        self.assertIn("VIDEO_DURATION", [check["code"] for check in response.json()["checks"]])

    def test_recommendation_rank_combines_embedding_audience_and_peer_signals(self):
        fake_embeddings = MagicMock()
        fake_embeddings.embed_documents.return_value = [[1.0, 0.0], [1.0, 0.0]]
        with patch("app.rag.embeddings.get_embeddings", return_value=fake_embeddings):
            response = self.client.post(
                "/api/v1/ai/intelligence/recommendations/rank",
                json={
                    "creatorProfile": {"preferredContentTypes": ["UGC"], "preferredCountries": ["JOR"]},
                    "campaigns": [{"id": "campaign-a", "content": "UGC product story", "contentType": "UGC", "targetCountries": ["JOR"]}],
                    "collaborativeSignals": {"campaign-a": 4},
                },
                headers={"X-Service-Key": "test-service-key"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["data"][0]["score"], 0.91)
        self.assertEqual(set(response.json()["data"][0]["reasons"]), {
            "CONTENT_SIMILARITY", "AUDIENCE_COUNTRY_MATCH", "PREFERRED_CONTENT_TYPE", "SIMILAR_CREATORS_INTERACTED",
        })

    def test_view_analysis_route_validates_and_returns_seven_day_analytics(self):
        response = self.client.post(
            "/api/v1/ai/intelligence/views/analyze",
            json={"campaignId": "campaign-1", "cpm": 5, "targetCountries": ["JOR"], "events": [], "validatedHistory": []},
            headers={"X-Service-Key": "test-service-key"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["windowDays"], 7)
        self.assertEqual(response.json()["earningsPrediction"]["status"], "INSUFFICIENT_DATA")
        self.assertEqual(response.json()["earningsAnomaly"]["status"], "INSUFFICIENT_DATA")
        self.assertFalse(response.json()["nonTargetGeographicAnomaly"]["isFraud"])

    def test_withdrawal_intelligence_route_and_language_detection_are_typed_and_service_key_protected(self):
        headers = {"X-Service-Key": "test-service-key"}
        response = self.client.post("/api/v1/ai/intelligence/withdrawals/analyze", json={
            "country": "JO", "candidate": {"amountCents": 5000, "method": "PAYPAL"},
            "availableMethods": ["PAYPAL"], "paymentMethodsByCountry": {"JO": ["PAYPAL"]}, "withdrawals": [],
        }, headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["paymentMethodRecommendation"]["methods"], ["PAYPAL"])
        self.assertEqual(response.json()["processingTimePrediction"]["status"], "INSUFFICIENT_DATA")
        detected = self.client.post("/api/v1/ai/intelligence/language/detect", json={"text": "مرحبا كيف حالك"}, headers=headers)
        self.assertEqual(detected.status_code, 200, detected.text)
        self.assertEqual(detected.json()["language"], "ar")
        self.assertEqual(self.client.post("/api/v1/ai/intelligence/language/detect", json={"text": "English"}).status_code, 401)

    def test_campaign_performance_route_returns_archive_and_pledge_aggregates(self):
        history = [
            *[{"id": f"ugc-{index}", "name": f"UGC {index}", "category": "UGC", "isArchived": True, "stats": {"totalViews": views, "totalSpent": 10}} for index, views in enumerate([300, 350, 400])],
            *[{"id": f"audio-{index}", "name": f"Audio {index}", "category": "AUDIO", "isArchived": True, "stats": {"totalViews": views, "totalSpent": 10}} for index, views in enumerate([100, 120, 140])],
        ]
        embeddings = MagicMock()
        embeddings.embed_documents.return_value = [[1.0, 0.0]] * (len(history) + 1)
        with patch("app.rag.embeddings.get_embeddings", return_value=embeddings):
            response = self.client.post("/api/v1/ai/intelligence/campaign/performance", headers={"X-Service-Key": "test-service-key"}, json={
                "campaign": {"id": "current", "name": "Current UGC", "category": "UGC", "totalBudget": 100},
                "history": history,
                "policyAcceptance": {"totalSubmissions": 10, "acceptedSubmissions": 7},
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["archivedCampaignAnalysis"]["archivedCampaigns"], 6)
        self.assertEqual(response.json()["successfulCampaignPatterns"]["patterns"][0]["category"], "UGC")
        self.assertEqual(response.json()["policyAcceptanceAnalytics"]["acceptanceRate"], 0.7)

    def test_translation_uses_validated_gemini_output_without_exposing_provider_errors(self):
        from app.routes.intelligence import TranslationOutput
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=TranslationOutput(translation="حملتك جاهزة.", detectedSourceLanguage="en"))
        settings = SimpleNamespace(ai_service_api_key="test-service-key", gemini_api_key="configured", gemini_request_timeout=5)
        with patch("app.routes.intelligence.get_settings", return_value=settings), patch("app.routes.intelligence.get_llm", return_value=fake_llm):
            response = self.client.post("/api/v1/ai/intelligence/language/translate", headers={"X-Service-Key": "test-service-key"}, json={"text": "Campaign is ready.", "targetLanguage": "ar"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"translation": "حملتك جاهزة.", "sourceLanguage": "en", "targetLanguage": "ar"})


class OptionalIntelligenceFeatureTests(unittest.TestCase):
    def setUp(self):
        self.client_context = patch("app.main.init_db", return_value=None)
        self.client_context.start()
        self.recovery_patch = patch("app.main.recover_interrupted_video_jobs", return_value=0)
        self.recovery_patch.start()
        self.client = TestClient(main.app)
        self.client.__enter__()
        self.settings_patch = patch(
            "app.routes.reviews.get_settings",
            return_value=SimpleNamespace(ai_service_api_key="test-service-key", gemini_api_key=""),
        )
        self.settings_patch.start()

    def tearDown(self):
        self.settings_patch.stop()
        self.client.__exit__(None, None, None)
        self.recovery_patch.stop()
        self.client_context.stop()

    def test_applicant_rank_uses_embedding_similarity_and_validates_country_codes(self):
        payload = {
            "campaign": {
                "id": "campaign-1", "content": "UGC product bottle daily routine",
                "categories": ["UGC"], "targetCountries": ["JOR"],
            },
            "applicants": [
                {"id": "app-1", "content": "UGC product bottle daily routine", "categories": ["UGC"], "audienceCountries": ["JOR"], "historicalPerformance": {"sampleSize": 1, "validatedViews": 700}},
                {"id": "app-2", "content": "gaming streams", "categories": ["GAMING"], "audienceCountries": ["USA"], "historicalPerformance": {"sampleSize": 0, "validatedViews": 0}},
            ],
        }
        embeddings = MagicMock()
        embeddings.embed_documents.return_value = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        with patch("app.rag.embeddings.get_embeddings", return_value=embeddings):
            response = self.client.post(
                "/api/v1/ai/intelligence/campaign/applicants/rank",
                json=payload,
                headers={"X-Service-Key": "test-service-key"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["method"], "GEMINI_EMBEDDINGS")
        self.assertEqual(data["data"][0]["applicantId"], "app-1")
        self.assertEqual(data["data"][0]["rank"], 1)
        self.assertEqual(data["data"][0]["breakdown"]["audienceCountryMatch"], 1.0)
        self.assertEqual(data["data"][1]["dataStatus"], "AVAILABLE")

        payload["campaign"]["targetCountries"] = ["JO"]
        invalid = self.client.post(
            "/api/v1/ai/intelligence/campaign/applicants/rank",
            json=payload,
            headers={"X-Service-Key": "test-service-key"},
        )
        self.assertEqual(invalid.status_code, 422)

    def test_applicant_rank_falls_back_to_token_overlap_and_requires_service_key(self):
        payload = {
            "campaign": {"id": "campaign-1", "content": "reusable bottle product", "categories": ["UGC"], "targetCountries": []},
            "applicants": [{"id": "app-1", "content": "reusable bottle product video", "categories": ["UGC"]}],
        }
        with patch("app.rag.embeddings.get_embeddings", side_effect=RuntimeError("provider unavailable")):
            response = self.client.post(
                "/api/v1/ai/intelligence/campaign/applicants/rank",
                json=payload,
                headers={"X-Service-Key": "test-service-key"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["method"], "TOKEN_OVERLAP_FALLBACK")
        self.assertGreater(response.json()["data"][0]["compatibilityScore"], 0)
        unauthorized = self.client.post("/api/v1/ai/intelligence/campaign/applicants/rank", json=payload)
        self.assertEqual(unauthorized.status_code, 401)

    def test_report_triage_finds_same_target_duplicate_and_falls_back_to_human_review(self):
        settings = SimpleNamespace(ai_service_api_key="test-service-key", gemini_api_key="", gemini_request_timeout=5)
        payload = {
            "targetType": "CAMPAIGN", "reason": "FALSE_CLAIMS",
            "details": "The campaign promises a guaranteed return without evidence.",
            "candidateReports": [{
                "id": "report-old", "reason": "FALSE_CLAIMS",
                "details": "The campaign promises a guaranteed return without evidence.",
            }],
        }
        with patch("app.services.report_intelligence_service.get_settings", return_value=settings), patch(
            "app.services.report_intelligence_service.get_embeddings", side_effect=RuntimeError("provider unavailable")
        ):
            response = self.client.post(
                "/api/v1/ai/intelligence/reports/classify",
                json=payload,
                headers={"X-Service-Key": "test-service-key"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "NEEDS_REVIEW")
        self.assertEqual(data["severity"], "UNKNOWN")
        self.assertTrue(data["needsHumanReview"])
        self.assertTrue(data["duplicateDetected"])
        self.assertEqual(data["duplicateMethod"], "TOKEN_OVERLAP")
        self.assertEqual(data["similarReports"][0]["reportId"], "report-old")

    def test_report_triage_validates_model_output_and_keeps_human_decision_required(self):
        from app.services.report_intelligence_service import ReportClassificationOutput

        settings = SimpleNamespace(ai_service_api_key="test-service-key", gemini_api_key="configured", gemini_request_timeout=5)
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=ReportClassificationOutput(
            category="GAMBLING", severity="HIGH", confidence=0.91,
            explanation="The supplied report warrants an evidence review.",
            suggestedActions=["REVIEW_EVIDENCE"], needsHumanReview=False,
        ))
        with patch("app.services.report_intelligence_service.get_settings", return_value=settings), patch(
            "app.services.report_intelligence_service.get_llm", return_value=fake_llm
        ):
            response = self.client.post(
                "/api/v1/ai/intelligence/reports/classify",
                json={"targetType": "CLIP", "reason": "OTHER", "details": "A specific report details this concern."},
                headers={"X-Service-Key": "test-service-key"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["category"], "GAMBLING")
        self.assertEqual(response.json()["severity"], "HIGH")
        self.assertTrue(response.json()["needsHumanReview"])

    def test_openapi_documents_optional_feature_contracts_and_security(self):
        schema = self.client.get("/openapi.json").json()
        ranking = schema["paths"]["/api/v1/ai/intelligence/campaign/applicants/rank"]["post"]
        triage = schema["paths"]["/api/v1/ai/intelligence/reports/classify"]["post"]
        earnings = schema["paths"]["/api/v1/ai/intelligence/creator-earnings/predict"]["post"]
        self.assertIn("AI Recommendations", ranking["tags"])
        self.assertIn("AI Report Intelligence", triage["tags"])
        self.assertIn("ServiceKey", str(ranking["security"]))
        self.assertEqual(triage["requestBody"]["content"]["application/json"]["schema"]["$ref"], "#/components/schemas/ReportClassificationRequest")
        self.assertIn("AI Views and Earnings Intelligence", earnings["tags"])
        self.assertIn("matured daily validated", earnings["description"])
        self.assertIn("ServiceKey", str(earnings["security"]))

    def test_creator_earnings_prediction_uses_matured_validated_history_and_reports_estimate_separately(self):
        payload = {"asOfDate": "2026-11-10", "campaigns": [{
            "campaignId": "campaign-1", "applicableCpm": 5,
            "dailyHistory": [
                {"day": "2026-10-28", "validatedViews": 900, "confirmedGrossAmountCents": 450, "confirmedCreatorNetAmountCents": 405},
                {"day": "2026-10-29", "validatedViews": 950, "confirmedGrossAmountCents": 475, "confirmedCreatorNetAmountCents": 428},
                {"day": "2026-10-30", "validatedViews": 1000, "confirmedGrossAmountCents": 500, "confirmedCreatorNetAmountCents": 450},
                {"day": "2026-10-31", "validatedViews": 1050, "confirmedGrossAmountCents": 525, "confirmedCreatorNetAmountCents": 473},
            ],
        }]}
        response = self.client.post(
            "/api/v1/ai/intelligence/creator-earnings/predict",
            json=payload,
            headers={"X-Service-Key": "test-service-key"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["status"], "ESTIMATED")
        self.assertGreater(result["estimatedCreatorNetAmountCents"], 0)
        self.assertLessEqual(result["estimatedCreatorNetAmountCents"], result["estimatedGrossAmountCents"])
        self.assertIn("not confirmed or withdrawable", result["campaigns"][0]["explanation"])

        insufficient = self.client.post(
            "/api/v1/ai/intelligence/creator-earnings/predict",
            json={"asOfDate": "2026-11-10", "campaigns": [{"campaignId": "campaign-2", "applicableCpm": 5, "dailyHistory": []}]},
            headers={"X-Service-Key": "test-service-key"},
        )
        self.assertEqual(insufficient.status_code, 200, insufficient.text)
        self.assertEqual(insufficient.json()["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(insufficient.json()["estimatedCreatorNetAmountCents"])

        held_views = self.client.post(
            "/api/v1/ai/intelligence/creator-earnings/predict",
            json={"asOfDate": "2026-11-10", "campaigns": [{"campaignId": "campaign-2", "applicableCpm": 5, "dailyHistory": [{"day": "2026-11-04", "validatedViews": 100}]}]},
            headers={"X-Service-Key": "test-service-key"},
        )
        self.assertEqual(held_views.status_code, 422)

    def test_creator_earnings_response_rejects_inconsistent_campaign_totals(self):
        from pydantic import ValidationError
        from app.routes.intelligence import CreatorEarningsPredictionResponse

        with self.assertRaises(ValidationError):
            CreatorEarningsPredictionResponse.model_validate({
                "status": "ESTIMATED", "windowDays": 7, "currency": "USD",
                "method": "MATURED_VIEW_TREND_AND_REALIZED_CPM",
                "estimatedGrossAmountCents": 100, "estimatedCreatorNetAmountCents": 90,
                "predictedNext7DaysViews": 500, "confidence": 0.5,
                "campaignsAnalyzed": 1, "campaignsEstimated": 1,
                "explanation": "Estimate.",
                "campaigns": [{
                    "campaignId": "campaign-1", "status": "ESTIMATED", "historicalDays": 4,
                    "maturedValidatedViews": 1000, "predictedNext7DaysViews": 400,
                    "estimatedGrossAmountCents": 100, "estimatedCreatorNetAmountCents": 90,
                    "confidence": 0.5, "explanation": "Estimate.",
                }],
            })


if __name__ == "__main__":
    unittest.main()
