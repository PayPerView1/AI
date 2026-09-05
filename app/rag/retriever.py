from typing import List
from langchain_core.documents import Document
from app.rag.vector_store import get_vector_store


def retrieve_relevant_docs(query: str, k: int = 4) -> List[Document]:
    """
    Retrieve the top-k most relevant documents for a given query.
    """
    try:
        store = get_vector_store()
        if store is None:
            return []
        docs = store.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"[retriever] Warning during similarity search: {e}")
        return []
