import logging
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from app.core.config import get_settings


logger = logging.getLogger(__name__)
_vector_store = None
_store_lock = Lock()


def expected_embedding_metadata(settings=None) -> dict[str, str]:
    settings = settings or get_settings()
    return {
        "embedding_model": settings.gemini_embedding_model,
        "embedding_profile": "gemini_retrieval_document_query_v1",
    }


def has_compatible_embedding_index(store, settings=None) -> bool:
    metadata = store._collection.metadata or {}
    expected = expected_embedding_metadata(settings)
    return all(metadata.get(key) == value for key, value in expected.items())


def _create_chroma_client(settings):
    import chromadb

    if settings.chroma_persist_directory.strip():
        service_root = Path(__file__).resolve().parents[2]
        persist_path = Path(settings.chroma_persist_directory)
        if not persist_path.is_absolute():
            persist_path = (service_root / persist_path).resolve()
        persist_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(persist_path))

    if settings.chroma_url.strip():
        parsed = urlparse(settings.chroma_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("CHROMA_URL must be a valid http(s) URL")
        port = parsed.port or (443 if parsed.scheme == "https" else 8000)
        return chromadb.HttpClient(
            host=parsed.hostname,
            port=port,
            ssl=parsed.scheme == "https",
        )

    raise ValueError("Set CHROMA_PERSIST_DIRECTORY or CHROMA_URL")


def get_vector_store():
    """Return the configured persistent or remote Chroma collection wrapper."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    with _store_lock:
        if _vector_store is not None:
            return _vector_store
        try:
            from langchain_chroma import Chroma
            from app.rag.embeddings import get_embeddings

            settings = get_settings()
            _vector_store = Chroma(
                collection_name=settings.chroma_collection,
                embedding_function=get_embeddings(),
                client=_create_chroma_client(settings),
                collection_metadata=expected_embedding_metadata(settings),
            )
        except Exception as exc:
            logger.warning("Chroma initialization failed (type=%s)", type(exc).__name__)
            return None
    return _vector_store


def check_vector_store() -> tuple[bool, int | None]:
    """Check that the configured collection is reachable and contains documents."""
    store = get_vector_store()
    if store is None:
        return False, None
    try:
        count = int(store._collection.count())
        compatible = has_compatible_embedding_index(store)
        return count > 0 and compatible, count
    except Exception as exc:
        logger.warning("Chroma readiness check failed (type=%s)", type(exc).__name__)
        return False, None


def reset_vector_store_cache() -> None:
    """Reset the cached client for tests and explicit configuration reloads."""
    global _vector_store
    with _store_lock:
        _vector_store = None
