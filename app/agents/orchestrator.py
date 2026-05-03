from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.account import account_agent
from app.agents.knowledge import knowledge_agent
from app.settings import settings
from app.srop.state import SessionState

ROOT_INSTRUCTION = """
You are the Helix Support Concierge, a routing agent.
Route each user message to the correct specialist tool.

Routing rules:
- Product docs, setup, troubleshooting, how-to, API, security, billing docs, CI/CD docs:
  call the knowledge specialist.
- The user's own account, plan, limits, usage, builds, failed builds, or account status:
  call the account specialist.
- Greetings, thanks, or tiny social turns: answer directly without a tool.

Always call a specialist tool when the intent matches one. Do not answer knowledge or
account questions yourself.
"""


def build_root_agent(state: SessionState) -> LlmAgent:
    """Build a root ADK agent with persisted session state in its instruction."""
    context = f"""
Current persisted user context:
- user_id: {state.user_id}
- plan_tier: {state.plan_tier}
- last_agent: {state.last_agent or "none"}
- turn_count: {state.turn_count}

Use this context for follow-up turns. Do not ask again for values already listed here.
When calling account tools, pass user_id exactly as: {state.user_id}
"""

    return LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        instruction=f"{ROOT_INSTRUCTION}\n{context}",
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
        ],
    )
