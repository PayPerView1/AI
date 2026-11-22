# PayPerView AI Service

FastAPI service for authenticated PayPerView chat and internal AI engineering workflows. It uses Gemini for responses, structured language analysis, vision review, and embeddings; ChromaDB stores platform documentation, and the Express API supplies authenticated account data. Chat history is stored in SQLModel/SQLite.

## Local setup

From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` and the same `JWT_SECRET` used by the Express backend in `.env`. For the supported monorepo setup, `npm run dev` from the workspace root passes the local backend URL and shared JWT secret to this service. Never commit a populated `.env` file.

The current collection is under `chroma_db/`. The checked-in collection predates the retrieval-task embedding profile and currently contains 66 records for 22 source chunks. The service rejects an unmarked or mismatched index for retrieval, so run the re-index before deploying this change or `/ready` will return HTTP 503:

```powershell
python ingest.py
```

Ingestion uses stable document and chunk IDs, updates changed chunks, and removes stale chunks for the Markdown sources it owns. If Gemini is not configured, ingestion exits without changing the existing collection.

Start the API with:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

`GET /health` reports process liveness. `GET /ready` checks Gemini, JWT and backend configuration plus configured Chroma availability and document count; it returns HTTP 503 until configured. It does not make a paid Gemini call or contact the Express API, so it cannot verify that those external credentials are accepted or reachable.

## API documentation

With the service running on its default local port, use:

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`
- OpenAPI JSON: `http://localhost:8001/openapi.json`

FastAPI generates the specification from mounted route definitions and Pydantic models. Endpoint groups include chat, review, campaign intelligence, budget intelligence, recommendations, report intelligence, content review, views and earnings intelligence, video analysis, and service checks. `ServiceKey` is shown as an `X-Service-Key` header scheme for backend integrations; never expose that credential in browser code. Campaign, view, earnings, and content intelligence calls are synchronous. Video processing is asynchronous and its job/result records use `DATABASE_URL`.

See [the feature completion report](docs/FEATURE_COMPLETION.md) for a status of each requested sprint capability, and [the Main Backend integration contract](docs/MAIN_BACKEND_INTEGRATION.md) for verified route mappings and remaining integration work.

### Video analysis workflow

- `POST /api/v1/ai/intelligence/videos/analyze` accepts `multipart/form-data` fields `tenantId`, `submissionId`, and `file` (up to 100 MiB). It returns HTTP 202 with an analysis ID and `QUEUED` status. The service samples up to eight frames, computes frame hashes and technical image quality, then uses one Gemini multimodal call for a summary, scenes, editing advice, and non-binding rejection suggestions.
- `GET /api/v1/ai/intelligence/videos/{analysis_id}?tenantId=...` returns `QUEUED`, `PROCESSING`, `COMPLETED`, `NEEDS_REVIEW`, or `FAILED`, with result, confidence, explanations, and similar-video matches scoped to the same tenant. Exact duplicate uploads within a tenant reuse the job. The uploaded source file is removed after processing.
- If Gemini is not configured, measured Computer Vision quality is retained while narrative fields remain unavailable and the job requires review. A service restart marks interrupted in-process jobs failed; the job record remains available for diagnosis.

The current Main Backend submission record stores a public platform URL and metadata, not video bytes or a durable frame source. Therefore its existing submission-review worker cannot yet queue this file-upload workflow automatically; a trusted video-byte ingestion/source adapter is needed to complete that handoff. The standalone internal API accepts real bytes and does not fetch creator-supplied URLs.

### Views and earnings analysis

- `POST /api/v1/ai/intelligence/views/analyze` accepts actual Main Backend view snapshots and matured ledger aggregates. It restricts view trend analysis to seven days, separates clear views still inside the seven-day hold from validated and matured ledger views, and returns spike/drop/suspicious, forecast, geography, and earnings analysis. Estimated gross and creator-net amounts use actual CPM, matured ledger realization, and remaining campaign budget when supplied.
- Country mismatch is reported as a targeting anomaly and is never treated as bot/fraud proof. Earnings estimates use forecast views, campaign CPM, matured validated ledger views, and confirmed historical gross earnings; they are advisory and separate from confirmed wallet balances.

## Configuration

| Variable | Required for readiness | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | Gemini chat and embedding access. No placeholder key or canned answer is used. |
| `GEMINI_MODEL` | No | Defaults to `gemini-3.8-flash`; low thinking is selected and unsupported sampling parameters are omitted. |
| `GEMINI_EMBEDDING_MODEL` | No | Defaults to `models/gemini-embedding-001`; document and query embeddings use matching retrieval task types. |
| `JWT_SECRET` | Yes | Must match the Express backend signing secret. The service has no fallback secret. |
| `AI_SERVICE_API_KEY` | Yes for review and intelligence endpoints | Shared server-to-server credential used by Express to call campaign, budget, recommendation, and submission review routes. Keep it identical in both services; never send it to a browser. |
| `JWT_ALGORITHM` | No | `HS256`, matching the Express token issuer. |
| `BACKEND_URL` | Yes | Express API base URL for authenticated tools; there is no hard-coded production host. |
| `FRONTEND_ORIGINS` | No | Comma-separated browser origins allowed by CORS; defaults to local frontend development. |
| `BACKEND_TIMEOUT_SECONDS` | No | Express API request timeout in seconds, default `10`. |
| `CHAT_RATE_LIMIT` / `CHAT_RATE_WINDOW_SECONDS` | No | Per-user process-local chat limit; defaults to `30` requests per `60` seconds. |
| `MAX_MESSAGE_CHARS` | No | Maximum accepted message length; default and API hard cap are `4000` characters. |
| `MAX_HISTORY_MESSAGES` / `MAX_HISTORY_CHARS` | No | Conversation context limits; defaults to `12` messages and `12000` characters. |
| `CHROMA_URL` | No | Optional remote Chroma HTTP(S) URL; used only when `CHROMA_PERSIST_DIRECTORY` is blank. |
| `CHROMA_PERSIST_DIRECTORY` | No | Local store path, default `./chroma_db`; this takes precedence over `CHROMA_URL`. |
| `CHROMA_COLLECTION` | No | Collection name, default `brands_clippers_kb`. |
| `DATABASE_URL` | No | SQLModel database URL; defaults to `chat_history.db` under this service directory. |
| `GEMINI_REQUEST_TIMEOUT` | No | Provider timeout in seconds, default `45`. |

For Render, set `JWT_SECRET` to the value shared with Express, `AI_SERVICE_API_KEY` to the value shared with Express, plus `BACKEND_URL` and `FRONTEND_ORIGINS`. The blueprint marks these settings as manually configured and uses `/ready` as its health check. A generated per-service JWT secret will not validate Express-issued tokens.

## Database and migrations

At startup, `init_db()` creates missing SQLModel tables and applies an idempotent migration that adds `chat_messages.metadata_json` to existing chat databases. `video_analysis_jobs` is created as a new table. Set `DATABASE_URL` to a persistent database URL in deployments; the default SQLite file is local to the service. There is no general Alembic revision chain yet, so future schema changes need an explicit migration before rollout.

## API and data access

The thread routes require `Authorization: Bearer <Express JWT>`:

- `POST /api/v1/ai/threads`
- `GET /api/v1/ai/threads`
- `GET /api/v1/ai/threads/{thread_id}/messages`
- `POST /api/v1/ai/threads/{thread_id}/messages`
- `DELETE /api/v1/ai/threads/{thread_id}`
- `POST /api/v1/chat` (authenticated compatibility endpoint)

The backend-only review routes require `X-Service-Key: <AI_SERVICE_API_KEY>`:

- `POST /review` evaluates campaign brief text with Gemini and returns a validated decision. Ambiguous or unverifiable policy questions should be sent to manual review.
- `POST /api/v1/ai/submissions/review` checks required keywords and hashtags deterministically, compares post text to the brief, and optionally reviews trusted sampled image frames using Gemini Vision. Without at least two frames it returns `MANUAL_REVIEW_REQUIRED` for media-dependent review. The Main Backend currently sends text and metadata only, so visual approval/rejection is not active in the integrated workflow. Audio compliance always requires a human review.
- `POST /api/v1/ai/intelligence/campaign/draft-analysis` returns draft completion suggestions, text-based checks and quality score, category/subcategory classification, owner-supplied embedding duplicate matches, and expiry status. It never changes or approves a stored draft.
- `POST /api/v1/ai/intelligence/campaign/performance` finds semantically similar supplied campaign outcomes and estimates views only when at least three similar completed campaigns have actual spend and view data; category analytics use actual supplied history.
- `POST /api/v1/ai/intelligence/budget/analyze` returns advisory payment/refund fraud signals, payment method and deposit recommendations, budget and exhaustion estimates, spend anomalies, refund method, and refund likelihood. Samples below configured minimums return `INSUFFICIENT_DATA`.
- `POST /api/v1/ai/intelligence/recommendations/rank` ranks only supplied visible campaigns using embedding similarity, creator audience preferences, and aggregate collaborative signals.
- `POST /api/v1/ai/intelligence/campaign/applicants/rank` ranks at most 200 applicants supplied by Main Backend against the advertiser's campaign brief, campaign categories and target countries, creator content categories, country audience signals, and matured validated-view history. It uses Gemini embeddings when configured and deterministic token overlap otherwise; absent signals are excluded and surfaced as insufficient data. Rankings are advisory and never make an application decision.
- `POST /api/v1/ai/intelligence/reports/classify` classifies an unverified report into the existing report taxonomy, assesses review urgency, compares only same-target candidate reports supplied by Main Backend, and suggests reviewer actions. Every response requires human review; provider failure preserves the submitted category with `UNKNOWN` severity. It does not approve, dismiss, or otherwise change report state.
- `POST /api/v1/ai/intelligence/creator-earnings/predict` estimates next-seven-day creator earnings from up to 100 campaign histories supplied by Main Backend. `asOfDate` anchors the complete UTC history window from 14 to 7 days ago, so views inside the seven-day hold are rejected. Each campaign needs at least four matured validated daily view observations, a positive active CPM, and confirmed gross/net ledger history; otherwise that campaign returns `INSUFFICIENT_DATA`. It uses a bounded linear trend and realized CPM, makes no LLM call, and always labels output as estimate-only rather than a confirmed or withdrawable balance.
- `POST /api/v1/ai/intelligence/withdrawals/analyze` returns advisory withdrawal behavior/frequency and amount anomaly signals, a country-method intersection, and a processing-time median only when the backend supplies sufficient creator-scoped history.
- `POST /api/v1/ai/intelligence/language/detect` identifies Arabic/English/unknown text using script detection without a model call. `POST /api/v1/ai/intelligence/language/translate` returns validated Arabic/English translation with bounded text and provider timeout.
- `POST /api/v1/ai/intelligence/campaign/performance` also returns owner-supplied archived campaign analysis, category patterns based on actual outcomes, and pledge acceptance analytics. Pledge acceptance is not evidence that campaign/content complies with policy.

All intelligence routes require the server-to-server `X-Service-Key`. The Main Backend scopes and sanitizes its database data before calling them; the AI Service does not read Main Backend database records directly. Campaign and finance outputs are advisory and do not execute policy, payment, refund, or budget actions. These endpoints are documented in this service's `/docs` and `/openapi.json` specification.

Thread and message reads are scoped to the token's user ID. The legacy chat route is also authenticated and rate limited. Request and response bodies are validated, and provider errors return an HTTP error rather than an invented assistant answer. The Main Backend stores chat feedback and passes a `NEUTRAL` or `MORE_CONCISE` response preference to this service; feedback storage remains in the Main Backend.

Tools are defined in one role-scoped registry and call the Express API with the caller's bearer token. Creators can query their submissions, eligible campaigns, and their own earnings summary. Brands can query their own campaigns, campaign insights, campaign submissions, review status, dashboard statistics, budget/fraud intelligence, and aggregate campaign report summaries. Both roles can fetch public campaign categories. Tool output is restricted by the backend to the authenticated user's role and ownership.

## Sprint 8–10 behavior and dependencies

- **Sprint 8:** The withdrawal analysis endpoint only consumes the authenticated creator's withdrawal records and active methods supplied by Main Backend. Empty history yields `UNKNOWN` risk; fewer than three completed processing samples yields `INSUFFICIENT_DATA`. It is advisory and cannot execute or block a withdrawal.
- **Sprint 9:** Assistant conversations are user-owned, use bounded history, support Arabic and English, return role-specific suggested actions, and adapt toward concise answers when stored negative feedback outweighs positive feedback. Campaign/budget/fraud/report/earnings answers use authenticated Main Backend tools, not fabricated profile data. Campaign performance notifications are generated by the backend from validated view history.
- **Sprint 10:** Campaign archive and category-pattern analysis uses only actual owner campaign outcomes. Pledge acceptance analytics are explicitly not interpreted as proof of policy compliance. Language detection and translation are exposed to authenticated Main Backend callers.
- **Unavailable infrastructure source:** Backup anomaly detection and storage issue prediction are not emitted because this repository has no real backup completion/failure feed, storage-capacity metrics, or object-store health interface. Supply those data sources before implementing those predictions.
- The Main Backend exposes translation APIs, but no existing frontend text-entry workflow was identified for a translation action. The UI remains unchanged rather than adding a new page.
- Unit and route tests use mocked provider/backend integrations. They do not validate live Gemini credentials, production traffic, payment-provider availability, or deployed database history.

Conversation history is limited by message count and character budget. Retrieval filters documents to `public` and the authenticated user's role. Tool results, retrieved text, and user messages are treated as untrusted data in the system prompt.

## Deployment and operational limits

- The per-user AI rate limiter is process local. Multi-instance deployments should also apply a shared gateway or distributed rate limit.
- The SQLite history database is local by default; deployments that need durable history must provide a persistent `DATABASE_URL`.
- Submission review remains human-gated for checks that require video/audio media. Automated media review requires a trustworthy media retrieval/inspection pipeline before approval or rejection can be enabled.
- Country-specific payment method recommendations require a truthful `PAYMENT_METHODS_BY_COUNTRY` JSON mapping in the Main Backend deployment. Methods are not recommended unless configured and supported by actual payment history.
- Daily budget insights depend on real UTC daily snapshots written by the Main Backend reset job. Initial recommendations can return `INSUFFICIENT_DATA` until enough complete days have accumulated. The Sprint 3 specification calls for deposit amount prediction from campaign history; it does not define shipping amount prediction.
## Testing

Install the development requirements and run the isolated service suite with Pytest:

```powershell
pip install -r requirements-dev.txt
python -m pytest -q -p no:cacheprovider tests
```

The tests use mocked providers/backend responses and temporary databases. They do not verify a live Gemini key, production traffic, or the deployed Express service.
