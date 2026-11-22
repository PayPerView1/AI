from sqlmodel import SQLModel, create_engine, Session
from typing import Generator
from pathlib import Path
from sqlalchemy import inspect, text

from app.core.config import get_settings

BASE_DIR = Path(__file__).parent.parent.parent
DB_PATH = (BASE_DIR / "chat_history.db").resolve()


def resolve_database_url(configured_url: str, default_path: Path = DB_PATH) -> str:
    return configured_url.strip() or f"sqlite:///{default_path.resolve()}"


DATABASE_URL = resolve_database_url(get_settings().database_url)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)


def init_db() -> None:
    """Create tables if they don't exist."""
    from app.db.models import ChatThread, ChatMessage, VideoAnalysisJob  # noqa: F401
    SQLModel.metadata.create_all(engine)
    migrate_legacy_chat_message_metadata(engine)


def migrate_legacy_chat_message_metadata(target_engine=None) -> None:
    """Add the metadata column introduced for chat response metadata to old databases."""
    target_engine = target_engine or engine
    inspector = inspect(target_engine)
    if "chat_messages" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "metadata_json" in columns:
        return
    with target_engine.begin() as connection:
        connection.execute(text(
            "ALTER TABLE chat_messages ADD COLUMN metadata_json VARCHAR NOT NULL DEFAULT '{}'"
        ))


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    with Session(engine) as session:
        yield session
