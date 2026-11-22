"""
Ingestion script: loads Markdown files from the documents/ directory,
splits them into chunks, generates Gemini embeddings, and stores them
in the Chroma vector database.

Run this script on-demand or during build:
    python ingest.py
"""

import glob
import hashlib
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import get_settings
from app.rag.vector_store import _create_chroma_client, expected_embedding_metadata

_settings = get_settings()
logger = logging.getLogger(__name__)

# Path to knowledge markdown files (ai-service/documents/)
DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Chunk configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_markdown_files() -> list[Document]:
    """Load all .md files from the documents directory and parse frontmatter."""
    documents = []
    md_files = sorted(glob.glob(str(DOCUMENTS_DIR / "*.md")))

    if not md_files:
        print(f"Warning: No .md files found in: {DOCUMENTS_DIR}")
        return []

    for file_path in md_files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Parse YAML-style frontmatter (--- ... ---)
        metadata: dict = {"source": filename, "category": "general", "role": "public"}
        content = raw.strip()

        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
        if frontmatter_match:
            fm_block = frontmatter_match.group(1)
            content = raw[frontmatter_match.end():].strip()
            for line in fm_block.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip()

        documents.append(Document(page_content=content, metadata=metadata))
        print(f"  - Loaded: {filename} (category={metadata.get('category')}, role={metadata.get('role')})")

    return documents


def stable_chunk_ids(chunks: list[Document]) -> list[str]:
    """Assign deterministic per-source IDs so document edits do not renumber others."""
    positions: dict[str, int] = defaultdict(int)
    ids = []
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        index = positions[source]
        positions[source] += 1
        chunk.metadata["chunk_index"] = index
        ids.append(hashlib.sha256(f"{source}:{index}".encode("utf-8")).hexdigest())
    return ids


def _provider_status_code(exc: Exception) -> int | None:
    current: BaseException | None = exc
    while current is not None:
        for attribute in ("status_code", "status", "code"):
            value = getattr(current, attribute, None)
            if isinstance(value, int) and 100 <= value <= 599:
                return value
        current = current.__cause__
    return None


def reconcile_index(
    store,
    chunks: list[Document],
    source_names: list[str],
    index_metadata: dict[str, str],
) -> int:
    """Upsert current chunks, then prune stale chunks from only the owned sources."""
    ids = stable_chunk_ids(chunks)
    store.add_documents(chunks, ids=ids)
    existing = store._collection.get(
        where={"source": {"$in": source_names}},
        include=["metadatas"],
    )
    desired_ids = set(ids)
    stale_ids = [record_id for record_id in existing["ids"] if record_id not in desired_ids]
    if stale_ids:
        store._collection.delete(ids=stale_ids)
    metadata = store._collection.metadata or {}
    store._collection.modify(metadata={**metadata, **index_metadata})
    return len(stale_ids)


def run_ingestion():
    """Main ingestion pipeline."""
    print("--- Starting RAG Knowledge Ingestion (Python) ---")
    print(f"Reading from: {DOCUMENTS_DIR}")

    if not _settings.gemini_api_key.strip():
        logger.warning("Gemini key is not configured; ingestion skipped")
        return None

    try:
        from langchain_chroma import Chroma
        from app.rag.embeddings import get_embeddings
    except ImportError as e:
        logger.error("Ingestion dependency is missing (type=%s)", type(e).__name__)
        return False

    docs = load_markdown_files()
    if not docs:
        logger.warning("No platform documents found; ingestion skipped")
        return None

    print(f"\nSplitting {len(docs)} documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Generated {len(chunks)} chunks.")

    try:
        store = Chroma(
            collection_name=_settings.chroma_collection,
            embedding_function=get_embeddings(),
            client=_create_chroma_client(_settings),
            collection_metadata=expected_embedding_metadata(_settings),
        )
        # Embed/upsert first. If Gemini fails, the existing indexed documents stay intact.
        source_names = sorted({str(doc.metadata["source"]) for doc in docs})
        stale_count = reconcile_index(
            store,
            chunks,
            source_names,
            expected_embedding_metadata(_settings),
        )
        logger.info(
            "Knowledge base ingestion completed (documents=%s chunks=%s stale_chunks_removed=%s)",
            len(docs),
            len(chunks),
            stale_count,
        )
        return True
    except Exception as exc:
        logger.error(
            "Knowledge base ingestion failed (type=%s provider_status=%s)",
            type(exc).__name__,
            _provider_status_code(exc) or "unknown",
        )
        return False


if __name__ == "__main__":
    if run_ingestion() is False:
        raise SystemExit(1)
