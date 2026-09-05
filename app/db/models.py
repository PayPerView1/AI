import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship


class ChatThread(SQLModel, table=True):
    __tablename__ = "chat_threads"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(index=True, nullable=False)
    title: str = Field(default="New Conversation", max_length=255)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    messages: List["ChatMessage"] = Relationship(back_populates="thread", cascade_delete=True)


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    thread_id: str = Field(foreign_key="chat_threads.id", index=True, nullable=False)
    role: str = Field(nullable=False)  # 'user' | 'assistant'
    content: str = Field(nullable=False)
    metadata_json: str = Field(default="{}", nullable=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    thread: Optional[ChatThread] = Relationship(back_populates="messages")
