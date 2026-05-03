"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 512 --chunk-overlap 64

Reads markdown files, chunks them, embeds, and writes to the vector store.

TODO for candidate: implement chunking and embedding logic.
"""
import argparse
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
import yaml

from app.settings import settings

COLLECTION_NAME = "helix_docs"
EMBEDDING_DIM = 384


def embed_text(text: str) -> list[float]:
    """Create a deterministic local embedding from token hashes."""
    vector = [0.0] * EMBEDDING_DIM
    tokens = re.findall(r"[a-z0-9_/-]+", text.lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL).strip()


def _split_long_section(section: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", section)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and current_len + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current).strip())
            tail = chunks[-1][-overlap:] if overlap > 0 else ""
            current = [tail, sentence] if tail else [sentence]
            current_len = sum(len(part) for part in current) + max(len(current) - 1, 0)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def make_chunk_id(file_path: Path, chunk_index: int) -> str:
    raw = f"{file_path.as_posix()}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool):
            clean[key] = value
        elif isinstance(value, list):
            clean[key] = ",".join(str(item) for item in value)
    return clean


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split markdown text into overlapping chunks.

    Design considerations:
    - Simple character splitting is fast but breaks mid-sentence.
    - Sentence-aware splitting is better for retrieval quality.
    - Heading-aware splitting (split on ## / ###) keeps sections coherent.
    - Overlap helps preserve context at chunk boundaries.

    Choose an approach and document why in the README.
    """
    body = _strip_frontmatter(text)
    sections = re.split(r"\n(?=#{1,3}\s+)", body)
    chunks: list[str] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(_split_long_section(section, chunk_size, overlap))

    return [chunk for chunk in chunks if chunk.strip()]


def extract_metadata(file_path: Path, text: str) -> dict:
    """
    Extract metadata from a markdown file's frontmatter.

    Expected frontmatter format:
        ---
        title: Deploy Keys
        product_area: security
        tags: [keys, secrets]
        ---

    Returns a dict suitable for vector store metadata filtering.
    """
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    parsed: dict[str, Any] = {}
    if match:
        loaded = yaml.safe_load(match.group(1)) or {}
        if isinstance(loaded, dict):
            parsed = loaded

    parsed.setdefault("title", file_path.stem.replace("-", " ").title())
    parsed.setdefault("product_area", "general")
    parsed["source"] = file_path.name
    return parsed


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """
    Walk docs_path, chunk and embed every .md file, upsert into vector store.

    Design considerations:
    - Generate a stable chunk_id (e.g. sha256(file + chunk_index)) for deduplication.
    - Run embeddings in batches to avoid rate limiting.
    - Print progress so the user can see what's happening.
    """
    md_files = sorted(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")
    collection = get_collection()

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata = extract_metadata(file_path, text)
        chunks = chunk_markdown(text, chunk_size, chunk_overlap)
        print(f"  {file_path.name}: {len(chunks)} chunks")
        if not chunks:
            continue

        relative_path = file_path.relative_to(docs_path)
        ids = [make_chunk_id(relative_path, index) for index in range(len(chunks))]
        metadatas = [
            _metadata_for_chroma(
                {
                    **metadata,
                    "chunk_index": index,
                    "chunk_id": chunk_id,
                }
            )
            for index, chunk_id in enumerate(ids)
        ]
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=[embed_text(chunk) for chunk in chunks],
            metadatas=metadatas,
        )

    print("Ingest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
