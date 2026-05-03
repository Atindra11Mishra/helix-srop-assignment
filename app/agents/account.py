from dataclasses import asdict

from google.adk.agents import LlmAgent

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.settings import settings

ACCOUNT_INSTRUCTION = """
You are Helix AccountAgent.
Use account tools for questions about the user's plan, usage, builds, build failures,
account limits, storage, or current account status.
Do not invent account data. Summarize tool results clearly and mention relevant IDs.
"""


async def recent_builds(user_id: str, limit: int = 5) -> list[dict]:
    """
    Get the user's most recent Helix CI/CD builds, newest first.

    Use this when the user asks about builds, failed builds, pipelines, branches,
    or recent CI/CD activity for their own account.
    """
    builds = await get_recent_builds(user_id=user_id, limit=limit)
    return [
        {
            **asdict(build),
            "started_at": build.started_at.isoformat(),
        }
        for build in builds
    ]


async def account_status(user_id: str) -> dict:
    """
    Get the user's Helix account status, plan tier, build limits, and storage usage.

    Use this when the user asks about their plan, limits, usage, storage, or account status.
    """
    return asdict(await get_account_status(user_id=user_id))


account_agent = LlmAgent(
    name="account",
    model=settings.adk_model,
    instruction=ACCOUNT_INSTRUCTION,
    tools=[recent_builds, account_status],
)
