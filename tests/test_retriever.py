"""
Unit tests for RAG retrieval.
Requires the vector store to be seeded first (run ingest.py on docs/).
"""
from pathlib import Path

import pytest

from app.agents.tools.search_docs import search_docs
from app.rag.ingest import chunk_markdown, ingest_directory
from app.settings import settings


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids(tmp_path):
    """search_docs must return chunk IDs and scores in [0, 1]."""
    settings.chroma_persist_dir = str(tmp_path / "chroma")
    await ingest_directory(Path("docs"), chunk_size=512, chunk_overlap=64)

    results = await search_docs("how to rotate a deploy key", k=3)

    assert len(results) > 0
    assert all(result.chunk_id for result in results)
    assert all(0.0 <= result.score <= 1.0 for result in results)
    assert results[0].metadata["source"] == "deploy-keys.md"


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    text = "# Header\n\nSome content.\n\n## Section 2\n\nMore content here."
    chunks = chunk_markdown(text, chunk_size=100, overlap=20)

    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)
