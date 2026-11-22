from typing import List

from langchain_core.documents import Document

from app.rag.vector_store import get_vector_store, has_compatible_embedding_index


MIN_RELEVANCE_SCORE = 0.15


def retrieve_relevant_docs(query: str, k: int = 3, role: str = "CLIPPER") -> List[Document]:
    """Retrieve only public and role-appropriate knowledge base chunks."""
    normalized_role = role.upper()
    if normalized_role not in {"CLIPPER", "BRAND"}:
        raise ValueError("Unsupported role for knowledge retrieval")
    if not query.strip():
        return []

    store = get_vector_store()
    if store is None:
        raise RuntimeError("Chroma collection is unavailable")
    if not has_compatible_embedding_index(store):
        raise RuntimeError("Chroma collection must be re-indexed for the configured Gemini embedding profile")

    metadata_filter = {
        "$or": [
            {"role": "public"},
            {"role": normalized_role.lower()},
        ]
    }
    result_limit = max(1, min(k, 5))
    scored_docs = store.similarity_search_with_relevance_scores(
        query.strip(),
        # Checked-in collections may contain repeated chunks from older ingestions.
        k=result_limit * 3,
        filter=metadata_filter,
    )
    documents = []
    seen_chunks = set()
    for document, score in scored_docs:
        if not isinstance(score, (int, float)) or score < MIN_RELEVANCE_SCORE:
            continue
        key = (str(document.metadata.get("source", "")), document.page_content)
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        documents.append(document)
        if len(documents) >= result_limit:
            break
    return documents
