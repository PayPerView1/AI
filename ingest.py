"""
Ingestion script: loads Markdown files from the documents/ directory,
splits them into chunks, generates Gemini embeddings, and stores them
in the Chroma vector database.

Run this script on-demand or during build:
    python ingest.py
"""

import os
import glob
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import get_settings

_settings = get_settings()

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


def run_ingestion():
    """Main ingestion pipeline."""
    print("--- Starting RAG Knowledge Ingestion (Python) ---")
    print(f"Reading from: {DOCUMENTS_DIR}")

    if not _settings.gemini_api_key:
        print("[ingest] GEMINI_API_KEY is not set. Skipping build-time ingestion. Pre-built chroma_db will be used.")
        return

    try:
        import chromadb
        from langchain_chroma import Chroma
        from app.rag.embeddings import get_embeddings
    except ImportError as e:
        print(f"[ingest] Dependency missing ({e}). Skipping ingestion.")
        return

    docs = load_markdown_files()
    if not docs:
        print("No documents loaded. Ingestion skipped.")
        return

    print(f"\nSplitting {len(docs)} documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Generated {len(chunks)} chunks.")

    if _settings.chroma_persist_directory:
        persist_path = Path(_settings.chroma_persist_directory)
        if not persist_path.is_absolute():
            persist_path = (Path(__file__).parent / persist_path).resolve()
        print(f"\nConnecting to local ChromaDB at {persist_path}...")
        client = chromadb.PersistentClient(path=str(persist_path))
    else:
        print(f"\nConnecting to ChromaDB at {_settings.chroma_url}...")
        chroma_host = _settings.chroma_url.replace("http://", "").split(":")[0]
        chroma_port = int(_settings.chroma_url.split(":")[-1])
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    print("Generating embeddings and saving to Chroma...")
    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            collection_name=_settings.chroma_collection,
            client=client,
        )
        print("--- Ingestion Completed Successfully! ---")
    except Exception as e:
        print(f"[ingest] Warning during embedding generation: {e}. Pre-existing ChromaDB will be preserved.")


if __name__ == "__main__":
    run_ingestion()
