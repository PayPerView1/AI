from pathlib import Path
from app.core.config import get_settings

_settings = get_settings()
_vector_store = None


def get_vector_store():
    """Get or initialize the Chroma vector store instance safely inside ai-service/."""
    global _vector_store
    if _vector_store is None:
        try:
            import chromadb
            from langchain_chroma import Chroma
            from app.rag.embeddings import get_embeddings

            # Always resolve relative paths against service root (ai-service/)
            service_root = Path(__file__).parent.parent.parent
            if _settings.chroma_persist_directory:
                persist_path = Path(_settings.chroma_persist_directory)
                if not persist_path.is_absolute():
                    persist_path = (service_root / persist_path).resolve()
                client = chromadb.PersistentClient(path=str(persist_path))
            else:
                chroma_host = _settings.chroma_url.replace("http://", "").split(":")[0]
                chroma_port = int(_settings.chroma_url.split(":")[-1])
                client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

            _vector_store = Chroma(
                collection_name=_settings.chroma_collection,
                embedding_function=get_embeddings(),
                client=client,
            )
        except Exception as err:
            print(f"[vector_store] Warning: Could not initialize Chroma store ({err}). RAG will return empty context.")
            return None

    return _vector_store
