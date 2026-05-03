"""
search_docs tool — used by KnowledgeAgent.

Queries the vector store for relevant documentation chunks.
Returns chunk IDs, scores, and content so the agent can cite sources.

TODO for candidate: implement this tool.
Wire it to your chosen vector store (Chroma, LanceDB, FAISS, etc.).
"""
from dataclasses import dataclass

from app.rag.ingest import embed_text, get_collection


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict  # e.g. {"product_area": "security", "source": "deploy-keys.md"}


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """
    Search the vector store for top-k relevant chunks.

    Args:
        query: natural language query from the user
        k: number of chunks to return
        product_area: optional metadata filter (e.g. "security", "ci-cd")

    Returns:
        List of DocChunk ordered by descending similarity score.

    Design considerations:
    - How do you embed the query? Same model as at ingest time.
    - Do you apply a score threshold to filter low-quality results?
    - How do you format chunks for the agent? Include chunk_id so agent can cite.
    """
    collection = get_collection()
    where = {"product_area": product_area} if product_area else None
    result = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=k,
        where=where,
        include=["documents", "distances", "metadatas"],
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    chunks: list[DocChunk] = []
    rows = zip(ids, documents, distances, metadatas, strict=False)
    for chunk_id, document, distance, metadata in rows:
        chunks.append(
            DocChunk(
                chunk_id=str(chunk_id),
                score=_distance_to_score(float(distance)),
                content=str(document),
                metadata=dict(metadata or {}),
            )
        )
    return chunks


def _distance_to_score(distance: float) -> float:
    score = 1.0 - distance
    return max(0.0, min(1.0, score))
