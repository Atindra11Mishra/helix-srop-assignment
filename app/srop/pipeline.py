"""
SROP entrypoint called by the chat route.

The pipeline owns app persistence around a single ADK turn: load session state,
run the root orchestrator, persist messages/state, and write a structured trace.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from google.adk.runners import InMemoryRunner
from google.genai import types
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import build_root_agent
from app.api.errors import RateLimitedError, SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message, Session
from app.settings import settings
from app.srop.state import SessionState

APP_NAME = "helix_srop"


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


@dataclass
class AdkTurnResult:
    content: str
    routed_to: str
    tool_calls: list[dict[str, Any]]
    retrieved_chunk_ids: list[str]


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    started = time.perf_counter()
    session = await db.scalar(select(Session).where(Session.session_id == session_id))
    if session is None:
        raise SessionNotFoundError(f"Session {session_id} was not found")

    state = SessionState.from_db_dict(session.state)
    trace_id = str(uuid.uuid4())

    try:
        turn = await asyncio.wait_for(
            _run_adk_turn(user_message=user_message, state=state, session_id=session_id),
            timeout=settings.llm_timeout_seconds,
        )
    except TimeoutError as exc:
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        ) from exc
    except Exception as exc:
        if exc.__class__.__name__ == "_ResourceExhaustedError":
            raise RateLimitedError("Gemini API quota was exhausted or rate limited") from exc
        raise

    state.turn_count += 1
    state.last_agent = _state_agent_name(turn.routed_to)
    session.state = state.to_db_dict()

    db.add(
        Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
            trace_id=trace_id,
        )
    )
    db.add(
        Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=turn.content,
            trace_id=trace_id,
        )
    )
    db.add(
        AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=turn.routed_to,
            tool_calls=turn.tool_calls,
            retrieved_chunk_ids=turn.retrieved_chunk_ids,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    )
    await db.commit()

    return PipelineResult(content=turn.content, routed_to=turn.routed_to, trace_id=trace_id)


async def _run_adk_turn(user_message: str, state: SessionState, session_id: str) -> AdkTurnResult:
    root_agent = build_root_agent(state)
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=state.user_id,
        session_id=session_id,
        state=state.to_db_dict(),
    )

    final_text = ""
    routed_to = "smalltalk"
    tool_calls: list[dict[str, Any]] = []
    pending_calls: dict[str, dict[str, Any]] = {}
    retrieved_chunk_ids: list[str] = []
    message = types.Content(role="user", parts=[types.Part.from_text(text=user_message)])

    async for event in runner.run_async(
        user_id=state.user_id,
        session_id=session_id,
        new_message=message,
    ):
        for function_call in event.get_function_calls():
            call_id = str(function_call.id or uuid.uuid4())
            tool_name = function_call.name or "unknown_tool"
            routed_to = _route_from_tool_name(tool_name)
            record = {
                "tool_name": tool_name,
                "args": dict(function_call.args or {}),
                "result": None,
            }
            pending_calls[call_id] = record
            tool_calls.append(record)

        for function_response in event.get_function_responses():
            response_name = function_response.name or "unknown_tool"
            call_id = str(function_response.id or "")
            response = _json_safe(function_response.response)
            record = pending_calls.get(call_id)
            if record is None:
                record = {"tool_name": response_name, "args": {}, "result": response}
                tool_calls.append(record)
            else:
                record["result"] = response
            retrieved_chunk_ids.extend(_extract_chunk_ids(response))

        if event.is_final_response() and event.content and event.content.parts:
            text = _parts_text(event.content.parts)
            if text:
                final_text = text
            routed_to = _route_from_author(event.author, routed_to)

    return AdkTurnResult(
        content=final_text or "I could not produce a response.",
        routed_to=routed_to,
        tool_calls=tool_calls,
        retrieved_chunk_ids=_dedupe(retrieved_chunk_ids),
    )


def _parts_text(parts: list[types.Part]) -> str:
    return "\n".join(part.text for part in parts if part.text).strip()


def _route_from_tool_name(tool_name: str) -> str:
    lowered = tool_name.lower()
    if "knowledge" in lowered:
        return "knowledge"
    if "account" in lowered:
        return "account"
    return lowered


def _route_from_author(author: str | None, fallback: str) -> str:
    if author in {"knowledge", "account"}:
        return author
    return fallback


def _state_agent_name(routed_to: str) -> str:
    if routed_to in {"knowledge", "account"}:
        return routed_to
    return "smalltalk"


def _extract_chunk_ids(value: Any) -> list[str]:
    chunk_ids: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "chunk_id" and isinstance(item, str):
                chunk_ids.append(item)
            else:
                chunk_ids.extend(_extract_chunk_ids(item))
    elif isinstance(value, list):
        for item in value:
            chunk_ids.extend(_extract_chunk_ids(item))
    return chunk_ids


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
