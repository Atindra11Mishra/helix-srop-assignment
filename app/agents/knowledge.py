from google.adk.agents import LlmAgent

from app.agents.tools.search_docs import search_docs
from app.settings import settings

KNOWLEDGE_INSTRUCTION = """
You are Helix KnowledgeAgent.
Answer only from Helix documentation returned by search_helix_docs.
Always cite the chunk IDs you used, for example: "According to [chunk_abc123]...".
If the retrieved chunks do not contain the answer, say you could not find it in the docs.
"""


async def search_helix_docs(query: str, k: int = 5, product_area: str | None = None) -> list[dict]:
    """
    Search Helix product documentation for relevant chunks.

    Use this for how-to questions, feature explanations, troubleshooting, API usage,
    billing documentation, CI/CD docs, security docs, and other product knowledge.
    Returns chunk IDs, scores, content, and metadata for citation.
    """
    results = await search_docs(query=query, k=k, product_area=product_area)
    return [
        {
            "chunk_id": result.chunk_id,
            "score": result.score,
            "content": result.content,
            "metadata": result.metadata,
        }
        for result in results
    ]


knowledge_agent = LlmAgent(
    name="knowledge",
    model=settings.adk_model,
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[search_helix_docs],
)
