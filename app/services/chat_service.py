import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    GoogleAuthenticationError,
    GoogleInvalidRequestError,
    GooglePermissionDeniedError,
    GoogleRateLimitError,
)
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import ChatMessage, ChatThread
from app.rag.prompts import build_system_prompt
from app.rag.retriever import retrieve_relevant_docs
from app.rag.tools import execute_tool, tool_definitions_for_role


logger = logging.getLogger(__name__)
_llm: ChatGoogleGenerativeAI | None = None
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_CONTEXT_CHARS = 8000
_MAX_TOOL_CALLS = 3
_MAX_TOOL_RESULT_CHARS = 6000
_MAX_RESPONSE_CHARS = 12000


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the configured Gemini model on demand; never substitute a fake key."""
    global _llm
    settings = get_settings()
    if not settings.gemini_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI provider is not configured.",
        )
    if _llm is None:
        model_kwargs: Dict[str, Any] = {
            "model": settings.gemini_model,
            "api_key": settings.gemini_api_key,
            "max_tokens": 1200,
            "request_timeout": settings.gemini_request_timeout,
            "retries": 2,
        }
        if settings.gemini_model.removeprefix("models/").startswith("gemini-3"):
            model_kwargs["thinking_level"] = "low"
        else:
            model_kwargs["temperature"] = 0.2
        _llm = ChatGoogleGenerativeAI(**model_kwargs)
    return _llm


def reset_llm_cache() -> None:
    global _llm
    _llm = None


def sanitize_user_message(content: str) -> str:
    """Normalize transport noise without pretending to neutralize prompt injection."""
    if not content:
        return ""
    return _CONTROL_CHARS.sub("", content).strip()


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        answer = content.strip()
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        answer = "\n".join(parts).strip()
    else:
        answer = ""

    if not answer or len(answer) > _MAX_RESPONSE_CHARS:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider returned an invalid response.",
        )
    return answer


def _build_rag_context(query: str, role: str) -> tuple[str, list[Dict[str, str]]]:
    documents = retrieve_relevant_docs(query, k=3, role=role)

    sections: list[str] = []
    sources: list[Dict[str, str]] = []
    remaining = _MAX_CONTEXT_CHARS
    for document in documents:
        source = str(document.metadata.get("source") or "platform documentation")[:120]
        content = document.page_content.strip()
        if not content or remaining <= 0:
            continue
        section = f"Source: {source}\n{content[:remaining]}"
        sections.append(section)
        sources.append({
            "source": source,
            "category": str(document.metadata.get("category") or "general")[:80],
        })
        remaining -= len(section)
    return "\n\n".join(sections), sources


def _history_messages(records: List[ChatMessage]) -> list[Any]:
    settings = get_settings()
    selected: list[ChatMessage] = []
    used_chars = 0
    for record in reversed(records):
        if record.role not in {"user", "assistant"}:
            continue
        content = record.content.strip()
        if not content:
            continue
        if selected and used_chars + len(content) > settings.max_history_chars:
            break
        if not selected and len(content) > settings.max_history_chars:
            content = content[-settings.max_history_chars:]
        selected.append(record.model_copy(update={"content": content}))
        used_chars += len(content)

    messages: list[Any] = []
    for record in reversed(selected):
        if record.role == "user":
            messages.append(HumanMessage(content=record.content))
        else:
            messages.append(AIMessage(content=record.content))
    return messages


async def _generate_answer(
    user_id: str,
    role: str,
    user_message: str,
    user_token: str,
    history: Optional[List[ChatMessage]] = None,
    response_preference: str = "NEUTRAL",
) -> tuple[str, list[Dict[str, str]]]:
    role = role.upper()
    if role not in {"CLIPPER", "BRAND"}:
        raise HTTPException(status_code=403, detail="This account role cannot use the assistant.")

    try:
        rag_context, sources = await asyncio.to_thread(_build_rag_context, user_message, role)
    except Exception as exc:
        logger.warning("RAG retrieval failed (type=%s role=%s)", type(exc).__name__, role)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Platform knowledge is temporarily unavailable. Please retry shortly.",
        ) from None
    system_prompt = build_system_prompt({}, role, rag_context, response_preference)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(_history_messages(history or []))
    messages.append(HumanMessage(content=user_message))

    try:
        llm = get_llm()
        available_tools = tool_definitions_for_role(role)
        first_model = llm.bind_tools(available_tools) if available_tools else llm
        response = await asyncio.wait_for(
            first_model.ainvoke(messages, candidate_count=None),
            timeout=get_settings().gemini_request_timeout + 5,
        )

        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            messages.append(response)
            calls_to_run = tool_calls[:_MAX_TOOL_CALLS]
            results = await asyncio.gather(
                *[
                    execute_tool(
                        call.get("name", ""),
                        user_id,
                        call.get("args", {}),
                        user_token=user_token,
                        role=role,
                    )
                    for call in calls_to_run
                ]
            )
            for call, result in zip(calls_to_run, results):
                call_id = call.get("id")
                if not call_id:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="The AI provider returned an invalid tool request.",
                    )
                serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                if len(serialized) > _MAX_TOOL_RESULT_CHARS:
                    serialized = json.dumps({
                        "error": "The platform returned more data than can be summarized.",
                        "code": "TOOL_RESULT_TOO_LARGE",
                    }, ensure_ascii=False)
                messages.append(ToolMessage(content=serialized, tool_call_id=call_id))

            for call in tool_calls[_MAX_TOOL_CALLS:]:
                call_id = call.get("id")
                if call_id:
                    messages.append(ToolMessage(
                        content='{"error":"Too many tool calls in one request.","code":"TOOL_LIMIT"}',
                        tool_call_id=call_id,
                    ))

            final_model = llm.bind_tools(available_tools, tool_choice="none")
            response = await asyncio.wait_for(
                final_model.ainvoke(messages, candidate_count=None),
                timeout=get_settings().gemini_request_timeout + 5,
            )

        return _message_text(response), sources
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.warning("Gemini request timed out (role=%s)", role)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The AI provider timed out. Please retry shortly.",
        ) from None
    except GoogleRateLimitError:
        logger.warning("Gemini request was rate limited (role=%s)", role)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI provider is busy. Please retry shortly.",
            headers={"Retry-After": "5"},
        ) from None
    except (GoogleAuthenticationError, GooglePermissionDeniedError):
        logger.error("Gemini provider credentials were rejected")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI provider is not configured correctly.",
        ) from None
    except GoogleInvalidRequestError:
        logger.error("Gemini rejected the generated request (role=%s)", role)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider rejected this request.",
        ) from None
    except Exception as exc:
        logger.error("Gemini request failed (type=%s role=%s)", type(exc).__name__, role)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI provider is temporarily unavailable.",
        ) from None


# ─── THREAD MANAGEMENT ─────────────────────────────────────────────────────

def create_thread(session: Session, user_id: str, title: Optional[str] = None) -> ChatThread:
    clean_title = (title or "").strip()[:255] or "New Conversation"
    thread = ChatThread(user_id=user_id, title=clean_title)
    session.add(thread)
    session.commit()
    session.refresh(thread)
    return thread


def get_user_threads(session: Session, user_id: str) -> List[Dict[str, Any]]:
    statement = (
        select(ChatThread)
        .where(ChatThread.user_id == user_id)
        .order_by(ChatThread.updated_at.desc())
    )
    threads = session.exec(statement).all()
    result = []
    for thread in threads:
        last_message_stmt = (
            select(ChatMessage)
            .where(ChatMessage.thread_id == thread.id)
            .order_by(ChatMessage.created_at.desc())
        )
        last_message = session.exec(last_message_stmt).first()
        result.append({
            "id": thread.id,
            "title": thread.title,
            "createdAt": thread.created_at.isoformat(),
            "updatedAt": thread.updated_at.isoformat(),
            "lastMessage": last_message.content[:300] if last_message else "",
        })
    return result


def verify_thread_ownership(session: Session, thread_id: str, user_id: str) -> ChatThread:
    """Return an existing user-owned thread without disclosing other users' IDs."""
    thread = session.get(ChatThread, thread_id)
    if thread is None or thread.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread not found.",
        )
    return thread


def get_thread_messages(session: Session, thread_id: str, user_id: str) -> List[Dict[str, Any]]:
    verify_thread_ownership(session, thread_id, user_id)
    statement = (
        select(ChatMessage)
        .where(ChatMessage.thread_id == thread_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    )
    messages = session.exec(statement).all()
    result = []
    for message in messages:
        try:
            metadata = json.loads(message.metadata_json or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        result.append({
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "createdAt": message.created_at.isoformat(),
            "language": metadata.get("language"),
            "suggestedActions": metadata.get("suggestedActions", []),
        })
    return result


def delete_thread(session: Session, thread_id: str, user_id: str) -> bool:
    thread = verify_thread_ownership(session, thread_id, user_id)
    session.delete(thread)
    session.commit()
    return True


async def process_thread_chat(
    session: Session,
    user_id: str,
    role: str,
    thread_id: str,
    user_message: str,
    user_token: str = "",
    response_preference: str = "NEUTRAL",
) -> Dict[str, Any]:
    thread = verify_thread_ownership(session, thread_id, user_id)
    safe_content = sanitize_user_message(user_message)
    if not safe_content:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(safe_content) > get_settings().max_message_chars:
        raise HTTPException(status_code=413, detail="Message is too long.")

    history_stmt = (
        select(ChatMessage)
        .where(ChatMessage.thread_id == thread_id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(get_settings().max_history_messages)
    )
    history_records = list(reversed(session.exec(history_stmt).all()))

    answer, _sources = await _generate_answer(
        user_id=user_id,
        role=role,
        user_message=safe_content,
        user_token=user_token,
        history=history_records,
        response_preference=response_preference,
    )

    if not history_records and thread.title == "New Conversation":
        thread.title = safe_content[:30] + ("..." if len(safe_content) > 30 else "")
    user_record = ChatMessage(thread_id=thread_id, role="user", content=safe_content)
    language = detect_chat_language(safe_content)
    actions = suggested_actions(role, language)
    assistant_record = ChatMessage(
        thread_id=thread_id,
        role="assistant",
        content=answer,
        metadata_json=json.dumps({"language": language, "suggestedActions": actions}, ensure_ascii=False),
    )
    thread.updated_at = datetime.now(timezone.utc)
    session.add(user_record)
    session.add(assistant_record)
    session.add(thread)
    session.commit()
    session.refresh(assistant_record)

    return {
        "success": True,
        "data": {
            "message": {
                "id": assistant_record.id,
                "role": "assistant",
                "content": assistant_record.content,
                "createdAt": assistant_record.created_at.isoformat(),
                "language": language,
                "suggestedActions": actions,
            }
        },
    }


async def process_chat(
    message: str,
    user_id: str,
    role: str,
    user_token: str,
) -> Dict[str, Any]:
    safe_content = sanitize_user_message(message)
    if not safe_content:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(safe_content) > get_settings().max_message_chars:
        raise HTTPException(status_code=413, detail="Message is too long.")
    answer, sources = await _generate_answer(user_id, role, safe_content, user_token)
    language = detect_chat_language(safe_content)
    return {"answer": answer, "sources": sources, "language": language, "suggestedActions": suggested_actions(role, language)}


def detect_chat_language(text: str) -> str:
    arabic = sum("\u0600" <= char <= "\u06ff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    if arabic >= max(2, latin):
        return "ar"
    if latin:
        return "en"
    return "unknown"


def suggested_actions(role: str, language: str) -> list[str]:
    if language == "ar":
        return ["اعرض حملاتي أو طلباتي", "راجع الأداء أو الأرباح", "اشرح الخطوة التالية"]
    if role.upper() == "BRAND":
        return ["Review my campaign performance", "Check campaign budgets", "Review campaign submissions"]
    return ["Show my submissions", "Find eligible campaigns", "Review my earnings"]
