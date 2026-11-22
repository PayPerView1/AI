from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import get_settings


class GeminiRetrievalEmbeddings(Embeddings):
    """Use Gemini's matching retrieval tasks for stored documents and queries."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.gemini_api_key.strip():
            raise RuntimeError("GEMINI_API_KEY is not configured")

        common = {
            "model": settings.gemini_embedding_model,
            "api_key": settings.gemini_api_key,
        }
        self._document_embeddings = GoogleGenerativeAIEmbeddings(
            **common,
            task_type="RETRIEVAL_DOCUMENT",
        )
        self._query_embeddings = GoogleGenerativeAIEmbeddings(
            **common,
            task_type="RETRIEVAL_QUERY",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._document_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._query_embeddings.embed_query(text)


def get_embeddings() -> GeminiRetrievalEmbeddings:
    return GeminiRetrievalEmbeddings()
