from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import get_settings

_settings = get_settings()


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a configured Gemini embedding model instance."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=_settings.gemini_api_key,
    )
