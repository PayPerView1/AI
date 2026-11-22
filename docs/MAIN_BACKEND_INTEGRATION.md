# Main Backend integration contract

## Scope and verification

This map was checked against the sibling PayPerView `Back-End` source tree on 2026-11-22, including its mounted Express routes, AI clients, and the controllers that build AI payloads. This is source-level contract verification; the live services, MongoDB data, provider credentials, and deployed environment were not exercised.

The AI Service does not connect to MongoDB or read Main Backend records directly. The Main Backend authenticates the caller, applies role and ownership checks, then sends only the data needed for the requested analysis. Intelligence endpoints use the shared `X-Service-Key`. User-scoped chatbot tools use the caller's bearer token.

## Main Backend calls into the AI Service

| AI Service endpoint | Main Backend caller | Contract notes |
| --- | --- | --- |
| `POST /review` | Campaign review service | Brief text and declarations only. A missing or incomplete Halal declaration requires manual review. |
| `POST /api/v1/ai/submissions/review` | Submission review client | Sends normalized caption, hashtags, requirements, declarations, backend checks, and `fetchStatus`. It does not send sampled video frames. The AI result therefore remains manual when media evidence is required. |
| `POST /api/v1/ai/intelligence/campaign/draft-analysis` | Owned campaign-draft assistance route | Sends the advertiser's draft and up to 50 of that advertiser's peer drafts. |
| `POST /api/v1/ai/intelligence/campaign/performance` | Owned campaign insights route | Sends the advertiser's campaign plus their completed/archived campaigns and aggregate Halal pledge acceptance counts. |
| `POST /api/v1/ai/intelligence/budget/analyze` | Authenticated brand wallet intelligence route | Sends only the brand's transactions, refunds, campaign history, daily spend, payment methods, and optional owned campaign context. |
| `POST /api/v1/ai/intelligence/recommendations/rank` | Campaign recommendation service | Sends visible candidates, the creator profile, and aggregate collaborative interaction counts. |
| `POST /api/v1/ai/intelligence/campaign/applicants/rank` | Application ranking service | Sends one owned campaign and its applicant profiles, including available audience and matured performance signals. |
| `POST /api/v1/ai/intelligence/reports/classify` | Report triage service | Sends the report and same-target candidates selected by the backend. AI output remains advisory and human-reviewed. |
| `POST /api/v1/ai/intelligence/views/analyze` | Campaign view analytics | Sends seven-day snapshot deltas and matured ledger aggregates. The AI result is produced after snapshots exist. |
| `POST /api/v1/ai/intelligence/creator-earnings/predict` | Creator earnings prediction service | Sends matured validated daily views, applicable CPM, and confirmed ledger history. |
| `POST /api/v1/ai/intelligence/withdrawals/analyze` | Creator wallet intelligence | Sends the authenticated creator's withdrawal history and available methods. |
| `POST /api/v1/ai/intelligence/language/detect` and `/translate` | Authenticated language routes | Arabic/English/unknown detection and bounded Arabic-English translation. |

All of these requests use `X-Service-Key: AI_SERVICE_API_KEY`. The receiving AI routes validate request models and validate structured Gemini output where a model is used. The Main Backend also validates important response shapes before returning them to the frontend or persisting results.

The Main Backend does not currently call `POST /api/v1/ai/intelligence/content/review` or the asynchronous `/api/v1/ai/intelligence/videos/*` workflow. Its submission review payload has no sampled frame bytes or durable media adapter. Those routes are available for a trusted internal caller but are not part of the current submission pipeline.

View snapshots are held for review when existing backend checks flag them; the backend accrues ledger earnings only after an unflagged snapshot or a human confirmation. AI view analysis currently summarizes stored snapshots after that workflow and does not participate in the pre-accrual decision.

## AI Service calls into the Main Backend

The role-scoped registry in `app/rag/tools.py` uses these read-only API routes with the caller's bearer token:

| Main Backend route | AI tool | Caller role |
| --- | --- | --- |
| `GET /api/v1/profile` | Session revalidation | BRAND or CLIPPER |
| `GET /api/v1/submissions/my` | `get_submissions` | CLIPPER |
| `GET /api/v1/submissions/eligible-campaigns` | `get_available_campaigns_for_clipper` | CLIPPER |
| `GET /api/v1/creator/earnings` | `get_creator_earnings_summary` | CLIPPER |
| `GET /api/v1/campaigns` | `get_brand_campaigns` | BRAND |
| `GET /api/v1/campaigns/{campaignId}` | `get_campaign_full_details` | BRAND |
| `GET /api/v1/campaigns/{campaignId}/ai-review` | `get_campaign_ai_review` | BRAND |
| `GET /api/v1/campaigns/{campaignId}/submissions` | `get_submissions_for_campaign` | BRAND |
| `GET /api/v1/campaigns/{campaignId}/ai-insights` | `get_campaign_ai_insights` | BRAND |
| `GET /api/v1/campaigns/{campaignId}/report-summary` | `get_campaign_report_summary` | BRAND |
| `GET /api/v1/campaigns/statistics` | `get_brand_dashboard_stats` | BRAND |
| `GET /api/v1/campaigns/categories` and `GET /api/v1/campaigns/categories/sub-categories` | Category tools | BRAND or CLIPPER |
| `POST /api/v1/wallet/intelligence` | `get_brand_budget_fraud_intelligence` | BRAND |

These paths were matched to the mounted Express route tree. Campaign IDs are validated before use, tool definitions are role-scoped, GET calls retry only transient failures, and upstream response bodies are not copied into error responses. Main Backend ownership middleware remains the authority for protected records.

## Authentication and chat history

- The Main Backend signs user JWTs. The AI Service accepts the Express `id`/`user_id`/`sub` identity claims and `BRAND` or `CLIPPER` role, then confirms the user and role through `GET /api/v1/profile`.
- A revoked or mismatched session is rejected. If the profile service is unavailable, the AI Service now returns `503` instead of treating the local JWT signature as sufficient session validation.
- The Main Backend owns chat thread mapping and feedback storage. It submits `responsePreference` (`NEUTRAL` or `MORE_CONCISE`) with each AI message based on the caller's stored `HELPFUL` / `NOT_HELPFUL` ratings. The AI Service applies that style hint; it does not duplicate the backend feedback store.
- Main Backend chat routes are mounted below `/api/v1/chat`. They proxy to the AI Service's `/api/v1/ai/threads` routes and persist their own user-visible conversation records.

## Remaining integration work

- Add a trusted frame or video-byte adapter before using visual review in the integrated submission pipeline. Do not make the AI Service fetch arbitrary creator URLs.
- If AI signals are to affect view eligibility before ledger accrual, agree on a separate request contract and call point with the Main Backend. Current AI analytics are post-snapshot and advisory; the backend's existing flags and human review remain authoritative.
- The recommendation route consumes aggregate collaborative signals. The current AI Service has no periodic model-training or refresh job; any event aggregation and refresh cadence must be defined by the Main Backend product/data contract.
- Backup health, storage capacity, and actual policy-acceptance telemetry are not present in the inputs used by these AI routes. The campaign endpoint's pledge acceptance aggregate is not proof of policy compliance.
- The local `.env` values and production secrets were not read or tested. Configure the same `AI_SERVICE_API_KEY` in both deployments and the Express `JWT_SECRET` in the AI Service before live calls.
