from collections import deque
from time import monotonic
from typing import Any, Deque, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Session

from app.core.config import get_settings
from app.core.security import get_current_user
from app.db.database import get_session
from app.services import chat_service


router = APIRouter(prefix="/api/v1", tags=["AI Chatbot"])
_rate_windows: Dict[str, Deque[float]] = {}

CHAT_ERRORS = {
    401: {"description": "Missing, invalid, or revoked Express bearer token.", "content": {"application/json": {"example": {"detail": "A bearer token is required."}}}},
    403: {"description": "The bearer token's role is not permitted to use this AI workflow.", "content": {"application/json": {"example": {"detail": "This account role cannot use the assistant."}}}},
    404: {"description": "The thread does not exist or belongs to another user.", "content": {"application/json": {"example": {"detail": "Chat thread not found."}}}},
    429: {"description": "Per-user chat rate limit exceeded.", "headers": {"Retry-After": {"description": "Seconds until another chat request is accepted.", "schema": {"type": "integer"}}}, "content": {"application/json": {"example": {"detail": "Chat request limit reached. Please try again shortly."}}}},
    502: {"description": "Gemini rejected the generated request or returned an invalid tool request.", "content": {"application/json": {"example": {"detail": "The AI provider rejected this request."}}}},
    503: {"description": "Backend session validation or Gemini is unavailable.", "content": {"application/json": {"example": {"detail": "The AI provider is temporarily unavailable."}}}},
    504: {"description": "Gemini did not respond before the configured request timeout.", "content": {"application/json": {"example": {"detail": "The AI provider timed out. Please retry shortly."}}}},
}


class CreateThreadRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"title": "Campaign setup questions"}]},
    )
    title: Optional[str] = Field(None, max_length=255)


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"content": "How do I submit a clip?", "responsePreference": "NEUTRAL"}]},
    )
    content: str = Field(..., min_length=1, max_length=4000)
    responsePreference: Literal["NEUTRAL", "MORE_CONCISE"] = "NEUTRAL"


class SimpleChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"message": "How do campaign applications work?"}]},
    )
    message: str = Field(..., min_length=1, max_length=4000)


class ThreadCreatedData(BaseModel):
    threadId: str
    title: str
    createdAt: str


class ThreadCreatedResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "success": True,
            "data": {"threadId": "thread_example_001", "title": "Campaign setup questions", "createdAt": "2026-11-10T09:30:00+00:00"},
        }]
    })
    success: bool = True
    data: ThreadCreatedData


class ThreadSummary(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "id": "thread_example_001", "title": "Campaign setup questions",
            "createdAt": "2026-11-10T09:30:00+00:00", "updatedAt": "2026-11-10T09:32:00+00:00",
            "lastMessage": "How do I submit a clip?",
        }]
    })
    id: str
    title: str
    createdAt: str
    updatedAt: str
    lastMessage: str


class ThreadListResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"success": True, "data": []}]
    })
    success: bool = True
    data: List[ThreadSummary]


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "id": "message_example_001", "role": "assistant",
            "content": "Open the campaign page and use the submission link after your application is accepted.",
            "createdAt": "2026-11-10T09:32:00+00:00",
            "language": "en", "suggestedActions": ["Show my submissions", "Find eligible campaigns"],
        }]
    })
    id: str
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)
    createdAt: str
    language: Optional[Literal["ar", "en", "unknown"]] = None
    suggestedActions: List[str] = Field(default_factory=list, max_length=5)


class ThreadMessagesResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"success": True, "data": []}]
    })
    success: bool = True
    data: List[ChatMessageResponse]


class SendMessageData(BaseModel):
    message: ChatMessageResponse


class SendMessageResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "success": True,
            "data": {"message": {
                "id": "message_example_001", "role": "assistant",
                "content": "Open the campaign page and use the submission link after your application is accepted.",
                "createdAt": "2026-11-10T09:32:00+00:00",
                "language": "en", "suggestedActions": ["Show my submissions", "Find eligible campaigns"],
            }},
        }]
    })
    success: bool = True
    data: SendMessageData


class DeleteThreadResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"success": True, "message": "Thread deleted successfully."}]
    })
    success: bool = True
    message: str


class ChatSource(BaseModel):
    source: str
    category: str


class SimpleChatResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "answer": "Creators can apply to open campaigns and submit a post link after acceptance.",
            "sources": [{"source": "campaign_submission_guide.md", "category": "campaigns"}],
            "language": "en", "suggestedActions": ["Show my submissions", "Find eligible campaigns"],
        }]
    })
    answer: str = Field(min_length=1, max_length=12000)
    sources: List[ChatSource]
    language: Literal["ar", "en", "unknown"] = "unknown"
    suggestedActions: List[str] = Field(default_factory=list, max_length=5)


def enforce_chat_rate_limit(user_id: str) -> None:
    """Apply a bounded per-user limit to expensive model requests."""
    settings = get_settings()
    now = monotonic()
    window_start = now - settings.chat_rate_window_seconds
    requests = _rate_windows.setdefault(user_id, deque())
    while requests and requests[0] <= window_start:
        requests.popleft()
    if len(requests) >= settings.chat_rate_limit:
        retry_after = max(1, int(requests[0] + settings.chat_rate_window_seconds - now))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Chat request limit reached. Please try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )
    requests.append(now)
    if len(_rate_windows) > 10000:
        for key, window in list(_rate_windows.items()):
            while window and window[0] <= window_start:
                window.popleft()
            if not window:
                _rate_windows.pop(key, None)


@router.post(
    "/ai/threads",
    response_model=ThreadCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a chat thread",
    description="Creates an empty thread for the authenticated user. The title is optional; no Gemini request is made.",
    responses={401: CHAT_ERRORS[401], 503: CHAT_ERRORS[503]},
)
async def create_chat_thread(
    body: Optional[CreateThreadRequest] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    thread = chat_service.create_thread(session, current_user["user_id"], body.title if body else None)
    return {
        "success": True,
        "data": {
            "threadId": thread.id,
            "title": thread.title,
            "createdAt": thread.created_at.isoformat(),
        },
    }


@router.get(
    "/ai/threads",
    response_model=ThreadListResponse,
    summary="List the caller's chat threads",
    description="Returns only threads owned by the bearer token's user ID, ordered by recent activity.",
    responses={401: CHAT_ERRORS[401], 503: CHAT_ERRORS[503]},
)
async def list_chat_threads(
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    return {"success": True, "data": chat_service.get_user_threads(session, current_user["user_id"])}


@router.get(
    "/ai/threads/{thread_id}/messages",
    response_model=ThreadMessagesResponse,
    summary="List messages in an owned thread",
    description="Returns the stored user and assistant messages in chronological order. Threads owned by another user are not disclosed.",
    responses={401: CHAT_ERRORS[401], 403: CHAT_ERRORS[403], 404: CHAT_ERRORS[404], 503: CHAT_ERRORS[503]},
)
async def get_thread_messages(
    thread_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    messages = chat_service.get_thread_messages(session, thread_id, current_user["user_id"])
    return {"success": True, "data": messages}


@router.post(
    "/ai/threads/{thread_id}/messages",
    response_model=SendMessageResponse,
    summary="Send a message to an owned thread",
    description="Runs one synchronous Gemini response with bounded history, role-scoped tools, and retrieved context. The assistant message is persisted only after a valid response. Retry after the supplied Retry-After interval when rate limited.",
    responses={401: CHAT_ERRORS[401], 403: CHAT_ERRORS[403], 404: CHAT_ERRORS[404], 429: CHAT_ERRORS[429], 502: CHAT_ERRORS[502], 503: CHAT_ERRORS[503], 504: CHAT_ERRORS[504]},
)
async def send_thread_message(
    thread_id: str,
    body: SendMessageRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    enforce_chat_rate_limit(current_user["user_id"])
    return await chat_service.process_thread_chat(
        session=session,
        user_id=current_user["user_id"],
        role=current_user["role"],
        thread_id=thread_id,
        user_message=body.content,
        user_token=current_user["raw_token"],
        response_preference=body.responsePreference,
    )


@router.delete(
    "/ai/threads/{thread_id}",
    response_model=DeleteThreadResponse,
    summary="Delete an owned thread",
    description="Deletes the caller's thread and its stored messages. A thread owned by another user is not disclosed.",
    responses={401: CHAT_ERRORS[401], 404: CHAT_ERRORS[404], 503: CHAT_ERRORS[503]},
)
async def delete_chat_thread(
    thread_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    chat_service.delete_thread(session, thread_id, current_user["user_id"])
    return {"success": True, "message": "Thread deleted successfully."}


@router.post(
    "/chat",
    response_model=SimpleChatResponse,
    summary="Ask a one-off RAG question",
    description="Authenticated compatibility endpoint. Returns a Gemini answer and the retrieved source labels without storing a conversation thread.",
    responses={401: CHAT_ERRORS[401], 403: CHAT_ERRORS[403], 429: CHAT_ERRORS[429], 502: CHAT_ERRORS[502], 503: CHAT_ERRORS[503], 504: CHAT_ERRORS[504]},
)
async def legacy_rag_chat(
    request: SimpleChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Authenticated compatibility endpoint; it has no persistent chat history."""
    enforce_chat_rate_limit(current_user["user_id"])
    return await chat_service.process_chat(
        request.message,
        user_id=current_user["user_id"],
        role=current_user["role"],
        user_token=current_user["raw_token"],
    )
