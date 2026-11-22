import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
import httpx
import jwt
from sqlalchemy import inspect, text
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool

import app.main as main
from app.core import security
from app.core.backend_client import backend_get
from app.core.config import Settings
from app.core.security import get_current_user
from app.db.database import get_session, migrate_legacy_chat_message_metadata
from app.db.database import resolve_database_url
from app.db.models import ChatMessage
from app.rag.prompts import build_system_prompt
from app.rag.retriever import retrieve_relevant_docs
from app.rag.tools import TOOL_REGISTRY, execute_tool, tool_definitions_for_role
from app.rag.vector_store import expected_embedding_metadata, has_compatible_embedding_index
from app.services import chat_service
from ingest import reconcile_index, stable_chunk_ids


class FakeCollection:
    def __init__(self):
        self.records = {}
        self.metadata = {}

    def get(self, where=None, include=None):
        sources = where["source"]["$in"]
        selected = {
            record_id: metadata
            for record_id, metadata in self.records.items()
            if metadata.get("source") in sources
        }
        return {"ids": list(selected), "metadatas": list(selected.values())}

    def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)

    def modify(self, metadata):
        self.metadata = metadata


class FakeStore:
    def __init__(self):
        self._collection = FakeCollection()

    def add_documents(self, documents, ids):
        for document, record_id in zip(documents, ids):
            self._collection.records[record_id] = dict(document.metadata)


class FakeLLM:
    def __init__(self, answer="Test response"):
        self.answer = answer
        self.bound_tools = None
        self.tool_choices = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        self.tool_choices.append(kwargs.get("tool_choice"))
        return self

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self.answer)


class FakeToolLLM(FakeLLM):
    def __init__(self):
        super().__init__("Tool grounded answer")
        self.invocations = 0

    async def ainvoke(self, messages, **kwargs):
        self.invocations += 1
        if self.invocations == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_submissions",
                    "args": {"limit": 2},
                    "id": "call-1",
                    "type": "tool_call",
                }],
            )
        return AIMessage(content=self.answer)


class AIServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        def override_session():
            with Session(self.engine) as session:
                yield session

        self.user = {"user_id": "owner-1", "role": "CLIPPER", "raw_token": "test-token"}

        async def override_user():
            return dict(self.user)

        main.app.dependency_overrides[get_session] = override_session
        main.app.dependency_overrides[get_current_user] = override_user
        main.app.dependency_overrides.pop(get_current_user, None)
        self.override_user = override_user
        main.app.dependency_overrides[get_current_user] = self.override_user
        self.client_context = patch("app.main.init_db", return_value=None)
        self.client_context.start()
        self.recovery_context = patch("app.main.recover_interrupted_video_jobs", return_value=0)
        self.recovery_context.start()
        self.client = TestClient(main.app)
        self.client.__enter__()
        import app.routes.chat as chat_routes
        chat_routes._rate_windows.clear()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.recovery_context.stop()
        self.client_context.stop()
        main.app.dependency_overrides.clear()
        self.engine.dispose()

    def test_legacy_chat_requires_authentication(self):
        main.app.dependency_overrides.pop(get_current_user, None)
        response = self.client.post("/api/v1/chat", json={"message": "hello"})
        self.assertEqual(response.status_code, 401)

    def test_swagger_redoc_and_openapi_document_all_ai_routes_and_security(self):
        swagger = self.client.get("/docs")
        redoc = self.client.get("/redoc")
        response = self.client.get("/openapi.json")

        self.assertEqual(swagger.status_code, 200)
        self.assertIn("/openapi.json", swagger.text)
        self.assertEqual(redoc.status_code, 200)
        self.assertEqual(response.status_code, 200)
        spec = response.json()
        from fastapi.routing import APIRoute
        from app.routes.chat import router as chat_router
        from app.routes.reviews import router as reviews_router
        from app.routes.intelligence import router as intelligence_router
        from app.routes.view_intelligence import router as view_intelligence_router
        from app.routes.video_analysis import router as video_analysis_router

        documented = {
            f"{method.upper()} {path}"
            for path, path_item in spec["paths"].items()
            for method in ("get", "post", "put", "patch", "delete")
            if method in path_item
        }
        registered_routes = [
            *chat_router.routes,
            *reviews_router.routes,
            *intelligence_router.routes,
            *view_intelligence_router.routes,
            *video_analysis_router.routes,
            *(route for route in main.app.routes if isinstance(route, APIRoute)),
        ]
        mounted = {
            f"{method} {route.path}"
            for route in registered_routes
            if getattr(route, "include_in_schema", False)
            for method in (getattr(route, "methods", None) or set())
            if method in {"GET", "POST", "PUT", "PATCH", "DELETE"}
        }

        self.assertEqual(documented, mounted)
        self.assertEqual(spec["components"]["securitySchemes"]["ServiceKey"]["in"], "header")
        self.assertEqual(spec["components"]["securitySchemes"]["ServiceKey"]["name"], "X-Service-Key")
        self.assertEqual(spec["paths"]["/review"]["post"]["security"], [{"ServiceKey": []}])
        self.assertEqual(spec["paths"]["/api/v1/ai/submissions/review"]["post"]["security"], [{"ServiceKey": []}])
        self.assertEqual(spec["paths"]["/api/v1/ai/intelligence/campaign/draft-analysis"]["post"]["security"], [{"ServiceKey": []}])
        self.assertIn("mediaFrames", spec["components"]["schemas"]["SubmissionReviewRequest"]["properties"])
        self.assertIn("/api/v1/ai/intelligence/views/analyze", spec["paths"])
        self.assertIn("/api/v1/ai/intelligence/videos/analyze", spec["paths"])
        self.assertIn("multipart/form-data", spec["paths"]["/api/v1/ai/intelligence/videos/analyze"]["post"]["requestBody"]["content"])
        self.assertEqual(
            spec["components"]["securitySchemes"]["HTTPBearer"]["scheme"],
            "bearer",
        )
        self.assertEqual(
            spec["paths"]["/api/v1/ai/threads"]["post"]["security"],
            [{"HTTPBearer": []}],
        )
        self.assertIn("examples", spec["components"]["schemas"]["CampaignReviewRequest"])
        self.assertIn("examples", spec["components"]["schemas"]["SendMessageRequest"])
        self.assertIn("503", spec["paths"]["/ready"]["get"]["responses"])

    def test_submission_review_requires_service_key_and_fails_closed_to_manual(self):
        import app.routes.reviews as reviews
        from unittest.mock import patch

        payload = {
            "submissionId": "submission-1",
            "platform": "TIKTOK",
            "postUrl": "https://example.invalid/post",
            "post": {"caption": "caption"},
            "campaign": {"name": "campaign"},
            "backendChecks": [],
        }
        with patch.object(reviews, "get_settings", return_value=SimpleNamespace(ai_service_api_key="review-secret")), patch(
            "app.routes.intelligence.get_settings",
            return_value=SimpleNamespace(ai_service_api_key="review-secret", gemini_api_key=""),
        ):
            denied = self.client.post("/api/v1/ai/submissions/review", json=payload)
            self.assertEqual(denied.status_code, 401)
            response = self.client.post(
                "/api/v1/ai/submissions/review",
                json=payload,
                headers={"X-Service-Key": "review-secret"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"], "MANUAL_REVIEW_REQUIRED")
        self.assertEqual(response.json()["manualReasonCode"], "MEDIA_EVIDENCE_UNAVAILABLE")

    def test_campaign_review_requires_service_key_and_validates_gemini_result(self):
        import app.routes.reviews as reviews

        payload = {
            "campaignName": "Campaign",
            "brief": {"mainIdea": "Explain the product"},
            "targetCountries": ["USA"],
            "halalDeclaration": {
                "noGambling": True,
                "noSexualContent": True,
                "noExplicitMusic": True,
                "noAlcohol": True,
                "noSuspiciousCurrencies": True,
                "noUnrealisticProfit": True,
            },
        }
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value={
            "result": "MANUAL_REVIEW_REQUIRED", "score": 70, "feedback": "Needs a human check."
        })
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        with patch.object(reviews, "get_settings", return_value=SimpleNamespace(ai_service_api_key="review-secret")), patch.object(
            reviews, "get_llm", return_value=llm
        ):
            response = self.client.post(
                "/review", json=payload, headers={"X-Service-Key": "review-secret"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"], "MANUAL_REVIEW_REQUIRED")
        self.assertEqual(response.json()["score"], 70)

    def test_campaign_review_sends_incomplete_halal_declarations_to_manual_without_llm_call(self):
        import app.routes.reviews as reviews

        payload = {
            "campaignName": "Campaign",
            "brief": {"mainIdea": "Explain the product"},
            "targetCountries": ["USA"],
            "halalDeclaration": {
                "noGambling": True, "noSexualContent": True, "noExplicitMusic": False,
                "noAlcohol": True, "noSuspiciousCurrencies": True, "noUnrealisticProfit": True,
            },
        }
        with patch.object(reviews, "get_settings", return_value=SimpleNamespace(ai_service_api_key="review-secret")), patch.object(
            reviews, "get_llm"
        ) as llm:
            response = self.client.post("/review", json=payload, headers={"X-Service-Key": "review-secret"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"], "MANUAL_REVIEW_REQUIRED")
        llm.assert_not_called()

    def test_thread_route_accepts_a_signed_express_style_bearer_token(self):
        test_secret = "isolated-route-test-secret-key-with-more-than-32-bytes"
        now = datetime.now(timezone.utc)
        token = jwt.encode(
            {
                "id": "backend-user-3",
                "role": "CLIPPER",
                "exp": int((now + timedelta(minutes=5)).timestamp()),
                "iat": int(now.timestamp()),
            },
            test_secret,
            algorithm="HS256",
        )
        main.app.dependency_overrides.pop(get_current_user, None)
        with patch.object(security.settings, "jwt_secret", test_secret), patch(
            "app.core.security.backend_get",
            new=AsyncMock(return_value={"user": {"_id": "backend-user-3", "role": "CLIPPER"}}),
        ):
            response = self.client.post(
                "/api/v1/ai/threads",
                json={"title": "Signed route test"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["data"]["title"], "Signed route test")

    def test_signed_but_revoked_token_is_rejected_by_express_session_check(self):
        test_secret = "isolated-route-test-secret-key-with-more-than-32-bytes"
        now = datetime.now(timezone.utc)
        token = jwt.encode(
            {
                "id": "backend-user-4", "role": "CLIPPER",
                "exp": int((now + timedelta(minutes=5)).timestamp()),
                "iat": int(now.timestamp()),
            },
            test_secret,
            algorithm="HS256",
        )
        main.app.dependency_overrides.pop(get_current_user, None)
        with patch.object(security.settings, "jwt_secret", test_secret), patch(
            "app.core.security.backend_get",
            new=AsyncMock(return_value={"error": "Not authorized", "code": "BACKEND_HTTP_401"}),
        ):
            response = self.client.get(
                "/api/v1/ai/threads", headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 401)

    def test_backend_outage_does_not_bypass_session_validation(self):
        test_secret = "isolated-route-test-secret-key-with-more-than-32-bytes"
        now = datetime.now(timezone.utc)
        token = jwt.encode(
            {
                "id": "backend-user-5", "role": "CLIPPER",
                "exp": int((now + timedelta(minutes=5)).timestamp()),
                "iat": int(now.timestamp()),
            },
            test_secret,
            algorithm="HS256",
        )
        main.app.dependency_overrides.pop(get_current_user, None)
        with patch.object(security.settings, "jwt_secret", test_secret), patch(
            "app.core.security.backend_get",
            new=AsyncMock(return_value={"error": "unavailable", "code": "BACKEND_UNAVAILABLE"}),
        ):
            response = self.client.get(
                "/api/v1/ai/threads", headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("unavailable", response.text)

    def test_thread_reads_are_scoped_to_authenticated_user(self):
        created = self.client.post("/api/v1/ai/threads", json={"title": "Account A"})
        self.assertEqual(created.status_code, 201)
        thread_id = created.json()["data"]["threadId"]

        self.user = {"user_id": "owner-2", "role": "CLIPPER", "raw_token": "other-token"}
        response = self.client.get(f"/api/v1/ai/threads/{thread_id}/messages")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get("/api/v1/ai/threads").json()["data"], [])

    def test_thread_message_uses_gemini_response_and_persists_valid_schema(self):
        created = self.client.post("/api/v1/ai/threads", json={})
        thread_id = created.json()["data"]["threadId"]
        fake_llm = FakeLLM("A grounded answer")
        with patch.object(chat_service, "retrieve_relevant_docs", return_value=[]), patch.object(
            chat_service, "get_llm", return_value=fake_llm
        ):
            response = self.client.post(
                f"/api/v1/ai/threads/{thread_id}/messages",
                json={"content": "How do I submit a clip?"},
            )

        self.assertEqual(response.status_code, 200)
        message = response.json()["data"]["message"]
        self.assertEqual(message["content"], "A grounded answer")
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["language"], "en")
        self.assertEqual(message["suggestedActions"][0], "Show my submissions")
        self.assertTrue(fake_llm.bound_tools)
        history = self.client.get(f"/api/v1/ai/threads/{thread_id}/messages").json()["data"]
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertEqual(history[-1]["suggestedActions"], message["suggestedActions"])

    def test_tool_cycle_runs_scoped_tool_and_disables_more_calls_for_final_answer(self):
        created = self.client.post("/api/v1/ai/threads", json={})
        thread_id = created.json()["data"]["threadId"]
        fake_llm = FakeToolLLM()
        with patch.object(chat_service, "retrieve_relevant_docs", return_value=[]), patch.object(
            chat_service, "get_llm", return_value=fake_llm
        ), patch.object(
            chat_service, "execute_tool", new=AsyncMock(return_value={"success": True, "data": []})
        ) as execute:
            response = self.client.post(
                f"/api/v1/ai/threads/{thread_id}/messages",
                json={"content": "Show my submissions"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["message"]["content"], "Tool grounded answer")
        execute.assert_awaited_once_with(
            "get_submissions", "owner-1", {"limit": 2}, user_token="test-token", role="CLIPPER"
        )
        self.assertEqual(fake_llm.invocations, 2)
        self.assertEqual(fake_llm.tool_choices[-1], "none")

    def test_provider_failure_returns_error_without_fabricated_reply(self):
        created = self.client.post("/api/v1/ai/threads", json={})
        thread_id = created.json()["data"]["threadId"]
        with patch.object(chat_service, "retrieve_relevant_docs", return_value=[]), patch.object(
            chat_service, "get_llm", side_effect=RuntimeError("provider unavailable")
        ):
            response = self.client.post(
                f"/api/v1/ai/threads/{thread_id}/messages",
                json={"content": "What did I earn?"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("240", response.text)
        self.assertEqual(self.client.get(f"/api/v1/ai/threads/{thread_id}/messages").json()["data"], [])

    def test_chat_request_schema_rejects_extra_fields_and_oversized_input(self):
        extra = self.client.post("/api/v1/chat", json={"message": "hello", "user_id": "other"})
        long_message = self.client.post("/api/v1/chat", json={"message": "x" * 4001})
        self.assertEqual(extra.status_code, 422)
        self.assertEqual(long_message.status_code, 422)


class ToolAndRetrievalTests(unittest.TestCase):
    def test_campaign_budget_and_report_tools_are_role_scoped(self):
        brand_tools = {item["name"] for item in tool_definitions_for_role("BRAND")}
        clipper_tools = {item["name"] for item in tool_definitions_for_role("CLIPPER")}
        self.assertIn("get_campaign_ai_insights", brand_tools)
        self.assertIn("get_campaign_report_summary", brand_tools)
        self.assertIn("get_brand_budget_fraud_intelligence", brand_tools)
        self.assertNotIn("get_campaign_report_summary", clipper_tools)
        self.assertIn("get_creator_earnings_summary", clipper_tools)

    def test_brand_campaign_status_schema_matches_express_contract(self):
        status_property = TOOL_REGISTRY["get_brand_campaigns"]["definition"]["parameters"]["properties"]["status"]
        self.assertEqual(
            set(status_property["enum"]),
            {
                "ALL",
                "DRAFT",
                "PENDING_REVIEW",
                "ACTIVE",
                "REJECTED",
                "MANUAL_REVIEW",
                "COMPLETED",
                "CANCELLED",
                "ARCHIVED",
                "EXPIRED",
                "PAUSED",
                "MANUALLY_PAUSED",
            },
        )

    def test_brand_campaign_status_matches_express_contract(self):
        async def check():
            with patch("app.rag.tools.backend_get", new=AsyncMock(return_value={"success": True})) as get:
                result = await execute_tool(
                    "get_brand_campaigns",
                    "brand-1",
                    {"status": "active", "limit": 5},
                    user_token="signed-token",
                    role="BRAND",
                )

                self.assertEqual(result, {"success": True})
                get.assert_awaited_once_with(
                    "/api/v1/campaigns",
                    "signed-token",
                    params={"page": 1, "limit": 5, "status": "ACTIVE"},
                )

            with patch("app.rag.tools.backend_get", new=AsyncMock()) as get:
                invalid = await execute_tool(
                    "get_brand_campaigns",
                    "brand-1",
                    {"status": "made-up"},
                    user_token="signed-token",
                    role="BRAND",
                )
                self.assertEqual(invalid["code"], "INVALID_TOOL_ARGUMENTS")
                get.assert_not_awaited()

        asyncio.run(check())

    def test_gemini_38_client_uses_supported_low_thinking_config(self):
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            api_key="isolated-local-test-key",
            max_tokens=1200,
            thinking_level="low",
            retries=2,
        )
        config = llm._prepare_params(stop=None, candidate_count=None).model_dump(exclude_none=True)
        self.assertEqual(llm.temperature, None)
        self.assertEqual(config["max_output_tokens"], 1200)
        self.assertNotIn("candidate_count", config)

    def test_gemini_default_is_current_stable_flash_model(self):
        self.assertEqual(Settings(_env_file=None).gemini_model, "gemini-3.8-flash")

    def test_cors_configuration_rejects_wildcard_origins(self):
        with self.assertRaises(ValueError):
            _ = Settings(frontend_origins="*").allowed_origins

    def test_conversation_history_is_chronological_and_role_typed(self):
        records = [
            ChatMessage(thread_id="t", role="user", content="Earlier question"),
            ChatMessage(thread_id="t", role="assistant", content="Earlier answer"),
        ]
        messages = chat_service._history_messages(records)
        self.assertEqual([message.type for message in messages], ["human", "ai"])
        self.assertEqual([message.content for message in messages], ["Earlier question", "Earlier answer"])

    def test_database_url_reads_config_and_has_service_local_fallback(self):
        self.assertEqual(resolve_database_url(" postgresql://db.example/app "), "postgresql://db.example/app")
        self.assertTrue(resolve_database_url("", Path("history.db")).startswith("sqlite:///"))

    def test_startup_migrates_legacy_chat_message_metadata_column(self):
        legacy_engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        try:
            with legacy_engine.begin() as connection:
                connection.execute(text(
                    "CREATE TABLE chat_messages (id VARCHAR PRIMARY KEY, thread_id VARCHAR NOT NULL, "
                    "role VARCHAR NOT NULL, content VARCHAR NOT NULL, created_at DATETIME NOT NULL)"
                ))
                connection.execute(text(
                    "INSERT INTO chat_messages (id, thread_id, role, content, created_at) "
                    "VALUES ('m1', 't1', 'assistant', 'Existing answer', '2026-01-01T00:00:00')"
                ))

            migrate_legacy_chat_message_metadata(legacy_engine)
            migrate_legacy_chat_message_metadata(legacy_engine)

            columns = {column["name"] for column in inspect(legacy_engine).get_columns("chat_messages")}
            with legacy_engine.connect() as connection:
                migrated = connection.execute(text(
                    "SELECT content, metadata_json FROM chat_messages WHERE id = 'm1'"
                )).one()
            self.assertIn("metadata_json", columns)
            self.assertEqual(migrated.content, "Existing answer")
            self.assertEqual(migrated.metadata_json, "{}")
        finally:
            legacy_engine.dispose()

    def test_missing_jwt_secret_fails_closed(self):
        with patch.object(security.settings, "jwt_secret", ""):
            with self.assertRaises(HTTPException) as raised:
                security.decode_jwt_token("untrusted")
        self.assertEqual(raised.exception.status_code, 503)

    def test_authentication_secrets_have_no_development_fallbacks(self):
        settings = Settings(_env_file=None)
        self.assertEqual(settings.jwt_secret, "")
        self.assertEqual(settings.ai_service_api_key, "")

    def test_express_style_id_and_role_claims_are_accepted(self):
        test_secret = "isolated-test-secret-key-with-more-than-32-bytes"
        now = datetime.now(timezone.utc)
        token = jwt.encode(
            {
                "id": "backend-user-1",
                "role": "BRAND",
                "exp": int((now + timedelta(minutes=5)).timestamp()),
                "iat": int(now.timestamp()),
            },
            test_secret,
            algorithm="HS256",
        )
        with patch.object(security.settings, "jwt_secret", test_secret):
            user = security.decode_jwt_token(token)
        self.assertEqual(user["user_id"], "backend-user-1")
        self.assertEqual(user["role"], "BRAND")

    def test_missing_gemini_key_does_not_create_a_placeholder_client(self):
        chat_service.reset_llm_cache()
        with patch("app.services.chat_service.get_settings", return_value=SimpleNamespace(gemini_api_key="")):
            with self.assertRaises(HTTPException) as raised:
                chat_service.get_llm()
        self.assertEqual(raised.exception.status_code, 503)

    def test_llm_factory_uses_model_appropriate_sampling_parameters(self):
        settings = SimpleNamespace(
            gemini_api_key="isolated-test-key",
            gemini_model="gemini-3.8-flash",
            gemini_request_timeout=30,
        )
        chat_service.reset_llm_cache()
        with patch("app.services.chat_service.get_settings", return_value=settings), patch(
            "app.services.chat_service.ChatGoogleGenerativeAI", return_value=object()
        ) as factory:
            chat_service.get_llm()
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["thinking_level"], "low")
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs["model"], "gemini-3.8-flash")

        chat_service.reset_llm_cache()
        settings.gemini_model = "gemini-2.5-flash"
        with patch("app.services.chat_service.get_settings", return_value=settings), patch(
            "app.services.chat_service.ChatGoogleGenerativeAI", return_value=object()
        ) as factory:
            chat_service.get_llm()
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertNotIn("thinking_level", kwargs)
        chat_service.reset_llm_cache()

    def test_registry_is_role_scoped_and_matches_dispatch_names(self):
        clipper_names = {tool["name"] for tool in tool_definitions_for_role("CLIPPER")}
        brand_names = {tool["name"] for tool in tool_definitions_for_role("BRAND")}
        self.assertEqual(clipper_names, {"get_submissions", "get_available_campaigns_for_clipper", "get_campaign_categories", "get_sub_categories", "get_creator_earnings_summary"})
        self.assertIn("get_submissions_for_campaign", brand_names)
        self.assertNotIn("get_weekly_earnings", clipper_names)
        self.assertEqual(clipper_names | brand_names, set(TOOL_REGISTRY))

        from langchain_core.utils.function_calling import convert_to_openai_tool
        for role in ("CLIPPER", "BRAND"):
            for definition in tool_definitions_for_role(role):
                self.assertEqual(convert_to_openai_tool(definition)["function"]["name"], definition["name"])

    def test_dispatch_rejects_wrong_role_and_invalid_campaign_id(self):
        denied = asyncio.run(execute_tool("get_brand_campaigns", "u1", {}, role="CLIPPER"))
        invalid = asyncio.run(execute_tool(
            "get_campaign_full_details", "u1", {"campaignId": "../../other"}, role="BRAND"
        ))
        self.assertEqual(denied["code"], "TOOL_NOT_ALLOWED")
        self.assertEqual(invalid["code"], "INVALID_TOOL_ARGUMENTS")

    def test_prompt_has_no_default_or_sample_account_metrics(self):
        prompt = build_system_prompt({}, "CLIPPER", "")
        self.assertIn("Never invent amounts, counts, names, statuses, or dates", prompt)
        self.assertNotIn("240.00", prompt)
        self.assertNotIn("September", prompt)
        refined = build_system_prompt({}, "CLIPPER", "", "MORE_CONCISE")
        self.assertIn("more direct, concise answer", refined)

    def test_rag_failure_returns_service_error_instead_of_ungrounded_answer(self):
        with patch("app.services.chat_service.retrieve_relevant_docs", side_effect=RuntimeError("stale index")), patch(
            "app.services.chat_service.get_llm"
        ) as llm:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(chat_service.process_chat("Platform policy?", "u1", "CLIPPER", "token"))
        self.assertEqual(raised.exception.status_code, 503)
        llm.assert_not_called()

    def test_gemini_rate_limit_is_reported_with_retry_hint(self):
        fake_llm = FakeLLM()

        async def rate_limited(messages, **kwargs):
            raise GoogleRateLimitError("upstream details are not returned")

        fake_llm.ainvoke = rate_limited
        with patch("app.services.chat_service.retrieve_relevant_docs", return_value=[]), patch(
            "app.services.chat_service.get_llm", return_value=fake_llm
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(chat_service.process_chat("How do I begin?", "u1", "CLIPPER", "token"))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.headers["Retry-After"], "5")
        self.assertNotIn("upstream details", raised.exception.detail)

    def test_retriever_filters_documents_by_role_and_relevance(self):
        docs = [
            Document(page_content="Brand only", metadata={"role": "brand", "source": "brands.md"}),
            Document(page_content="Campaigns", metadata={"role": "public", "source": "campaigns.md"}),
        ]
        fake_store = type("Store", (), {})()
        fake_store.similarity_search_with_relevance_scores = MagicMock(return_value=[
            (docs[0], 0.8), (docs[0], 0.8), (docs[0], 0.8), (docs[1], 0.7),
        ])
        with patch("app.rag.retriever.get_vector_store", return_value=fake_store), patch(
            "app.rag.retriever.has_compatible_embedding_index", return_value=True
        ):
            result = retrieve_relevant_docs("campaign budget", role="BRAND")
        self.assertEqual(result, docs)
        self.assertEqual(fake_store.similarity_search_with_relevance_scores.call_args.kwargs["k"], 9)
        self.assertEqual(
            fake_store.similarity_search_with_relevance_scores.call_args.kwargs["filter"]["$or"],
            [{"role": "public"}, {"role": "brand"}],
        )

    def test_ingestion_uses_stable_ids_and_removes_only_stale_owned_chunks(self):
        store = FakeStore()
        old_chunk = Document(page_content="old", metadata={"source": "clippers.md"})
        stable_old_id = stable_chunk_ids([old_chunk])[0]
        store._collection.records[stable_old_id] = dict(old_chunk.metadata)
        store._collection.records["stale"] = {"source": "clippers.md"}
        store._collection.records["unrelated"] = {"source": "other.md"}

        new_chunks = [Document(page_content="new", metadata={"source": "clippers.md"})]
        index_metadata = {"embedding_model": "models/gemini-embedding-001", "embedding_profile": "test-profile"}
        removed = reconcile_index(store, new_chunks, ["clippers.md"], index_metadata)
        self.assertEqual(removed, 1)
        self.assertEqual(len(store._collection.records), 2)
        self.assertIn("unrelated", store._collection.records)
        self.assertEqual(store._collection.metadata, index_metadata)
        self.assertEqual(stable_chunk_ids(new_chunks), stable_chunk_ids(new_chunks))

    def test_rag_refuses_an_unmarked_or_incompatible_embedding_index(self):
        store = FakeStore()
        self.assertFalse(has_compatible_embedding_index(store))
        store._collection.metadata = expected_embedding_metadata()
        self.assertTrue(has_compatible_embedding_index(store))
        store._collection.metadata["embedding_profile"] = "stale-profile"
        store._collection.count = MagicMock(return_value=22)
        stale_metadata = dict(store._collection.metadata)
        self.assertFalse(has_compatible_embedding_index(store))
        self.assertEqual(store._collection.metadata, stale_metadata)

    def test_backend_get_retries_transient_errors_and_never_returns_upstream_body(self):
        class FakeClient:
            calls = []
            responses = []

            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def request(self, method, url, headers, params=None, json=None):
                self.calls.append((method, url, headers))
                return self.responses.pop(0)

        request = httpx.Request("GET", "https://api.example.test/account")
        FakeClient.calls = []
        FakeClient.responses = [
            httpx.Response(503, json={"message": "private upstream detail"}, request=request),
            httpx.Response(200, json={"success": True, "data": []}, request=request),
        ]
        settings = SimpleNamespace(backend_url="https://api.example.test", backend_timeout_seconds=2)
        with patch("app.core.backend_client.get_settings", return_value=settings), patch(
            "app.core.backend_client.httpx.AsyncClient", FakeClient
        ), patch("app.core.backend_client.asyncio.sleep", new=AsyncMock()):
            result = asyncio.run(backend_get("/account", "test-token"))
        self.assertEqual(result, {"success": True, "data": []})
        self.assertEqual(len(FakeClient.calls), 2)
        self.assertEqual(FakeClient.calls[0][2]["Authorization"], "Bearer test-token")

        FakeClient.calls = []
        FakeClient.responses = [httpx.Response(401, json={"message": "private upstream detail"}, request=request)]
        with patch("app.core.backend_client.get_settings", return_value=settings), patch(
            "app.core.backend_client.httpx.AsyncClient", FakeClient
        ):
            result = asyncio.run(backend_get("/account", "test-token"))
        self.assertEqual(result["code"], "BACKEND_HTTP_401")
        self.assertNotIn("private upstream detail", str(result))
        self.assertEqual(len(FakeClient.calls), 1)


if __name__ == "__main__":
    unittest.main()
