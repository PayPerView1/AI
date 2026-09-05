from sqlmodel import SQLModel, create_engine, Session
from typing import Generator
from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent.parent
DB_PATH = (BASE_DIR / "chat_history.db").resolve()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)


def init_db() -> None:
    """Create tables if they don't exist."""
    from app.db.models import ChatThread, ChatMessage  # noqa: F401
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    with Session(engine) as session:
        yield session
