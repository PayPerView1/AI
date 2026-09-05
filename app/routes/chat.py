from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from sqlmodel import Session

from app.core.security import get_current_user
from app.db.database import get_session
from app.services import chat_service
from app.services.chat_service import process_chat

router = APIRouter(prefix="/api/v1", tags=["AI Chatbot"])


# ─── Pydantic Schemas ───────────────────────────────────────────────────────

class CreateThreadRequest(BaseModel):
    title: Optional[str] = Field(None, description="Optional title for the chat thread.")


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, description="User prompt or message content.")


class SimpleChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="RAG chat query.")


# ─── 1. Create Chat Thread (POST /api/v1/ai/threads) ────────────────────────

@router.post("/ai/threads", status_code=status.HTTP_21_CREATED if hasattr(status, "HTTP_21_CREATED") else 201)
async def create_chat_thread(
    body: Optional[CreateThreadRequest] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Create a new isolated chat thread for the authenticated user.
    """
    user_id = current_user["user_id"]
    title = body.title if body else None
    thread = chat_service.create_thread(session, user_id, title)
    return {
        "success": True,
        "data": {
            "threadId": thread.id,
            "title": thread.title,
            "createdAt": thread.created_at.isoformat(),
        }
    }


# ─── 2. List User Chat Threads (GET /api/v1/ai/threads) ─────────────────────

@router.get("/ai/threads")
async def list_chat_threads(
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    List all chat threads belonging to the authenticated user.
    """
    user_id = current_user["user_id"]
    threads = chat_service.get_user_threads(session, user_id)
    return {
        "success": True,
        "data": threads
    }


# ─── 3. Fetch Thread Messages (GET /api/v1/ai/threads/{thread_id}/messages) ──

@router.get("/ai/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Fetch message history for a specific thread with strict ownership verification.
    """
    user_id = current_user["user_id"]
    messages = chat_service.get_thread_messages(session, thread_id, user_id)
    return {
        "success": True,
        "data": messages
    }


# ─── 4. Send Message & Generate AI Response (POST /api/v1/ai/threads/{thread_id}/messages) ⭐

@router.post("/ai/threads/{thread_id}/messages")
async def send_thread_message(
    thread_id: str,
    body: SendMessageRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Send user prompt, inject context, run tool calling LLM, persist messages, and return assistant answer.
    """
    user_id = current_user["user_id"]
    role = current_user["role"]

    response = await chat_service.process_thread_chat(
        session=session,
        user_id=user_id,
        role=role,
        thread_id=thread_id,
        user_message=body.content,
    )
    return response


# ─── 5. Delete Chat Thread (DELETE /api/v1/ai/threads/{thread_id}) ──────────

@router.delete("/ai/threads/{thread_id}")
async def delete_chat_thread(
    thread_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Delete a chat thread and all its messages after ownership verification.
    """
    user_id = current_user["user_id"]
    chat_service.delete_thread(session, thread_id, user_id)
    return {
        "success": True,
        "message": f"Thread '{thread_id}' deleted successfully."
    }


# ─── 6. Legacy/Direct RAG Endpoint (POST /api/v1/chat) ──────────────────────

@router.post("/chat")
async def legacy_rag_chat(request: SimpleChatRequest):
    """
    Direct RAG query endpoint without thread or JWT requirements.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        result = await process_chat(request.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Service error: {str(e)}")
