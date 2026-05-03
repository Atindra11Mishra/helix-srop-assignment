# Helix SROP

AI Support Concierge for the Helix take-home assignment. The service exposes a
FastAPI API that creates support sessions, routes chat turns through Google ADK
agents, answers product questions from a local RAG corpus, handles mocked account
lookups, persists session state, and stores structured traces.

## Setup

```powershell
uv sync --extra dev
Copy-Item .env.example .env
```

Set a valid Gemini key in `.env`:

```env
GOOGLE_API_KEY=your-google-api-key
ADK_MODEL=gemini-2.0-flash
DATABASE_URL=sqlite+aiosqlite:///./helix_srop.db
CHROMA_PERSIST_DIR=./chroma_db
```

Ingest the bundled docs into Chroma:

```powershell
uv run python -m app.rag.ingest --path docs/
```

Start the API:

```powershell
uv run uvicorn app.main:app --reload
```

Health check:

```powershell
curl.exe "http://localhost:8000/healthz"
```

## Quick Test

PowerShell:

```powershell
$session = Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/v1/sessions" `
  -ContentType "application/json" `
  -Body (@{ user_id = "u_demo_pro"; plan_tier = "pro" } | ConvertTo-Json)

$chat = Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/v1/chat/$($session.session_id)" `
  -ContentType "application/json" `
  -Body (@{ content = "How do I rotate a deploy key?" } | ConvertTo-Json)

$chat

Invoke-RestMethod -Method Get `
  -Uri "http://localhost:8000/v1/traces/$($chat.trace_id)"
```

If the Gemini project has no available quota, `/v1/chat/{session_id}` returns a
structured `RATE_LIMITED` 429 response. The local test suite mocks ADK at the
agent boundary and does not require live Gemini quota.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/healthz` | Health check |
| `POST` | `/v1/sessions` | Create a session with `{user_id, plan_tier}` |
| `POST` | `/v1/chat/{session_id}` | Run one support turn |
| `GET` | `/v1/traces/{trace_id}` | Return the structured trace for a turn |

## Architecture

```text
POST /v1/chat/{session_id}
        |
        v
  SROP Pipeline
  - load Session + SessionState from SQLite
  - build ADK root agent with persisted context
  - run root orchestrator with timeout
  - collect final response, route, tool calls, chunk IDs
  - persist messages, updated state, and AgentTrace
        |
        v
  Google ADK Root Agent
  - routes via AgentTool, not manual string parsing
        |
        +--> KnowledgeAgent
        |    - search_helix_docs tool
        |    - Chroma vector store
        |    - citations use chunk IDs
        |
        +--> AccountAgent
             - recent_builds tool
             - account_status tool
             - deterministic mocked account data
```

## Design Decisions

### State Persistence

I used Pattern 3 from the ADK guide: store compact `SessionState` in the
`sessions.state` JSON column and inject that state into the root ADK instruction
on each turn.

This keeps the persistence model explicit and restart-safe while avoiding a
custom ADK session service. The persisted state currently tracks `user_id`,
`plan_tier`, `last_agent`, and `turn_count`.

### Chunking Strategy

The RAG ingest uses heading-aware Markdown chunking. Markdown docs are already
structured by headings, so preserving sections gives more coherent retrieval
than fixed character splits. Long sections are split further on sentence
boundaries with overlap.

Chunk IDs are deterministic:

```text
chunk_ + sha256(relative_file_path::chunk_index)[:16]
```

Re-running ingest upserts the same IDs instead of duplicating documents.

### Vector Store

I chose Chroma because it is already included in the project dependencies,
persists locally with no external service, and supports metadata filters. The
implementation uses deterministic local hash embeddings so tests and local
development do not require a paid embedding API.

### ADK Routing

The root orchestrator uses ADK `AgentTool` wrappers around `KnowledgeAgent` and
`AccountAgent`. Routing is delegated to the model's tool selection rather than
manual parsing of the user's message.

### Account Data

Account tools return deterministic mocked data. This is allowed by the
assignment and keeps the wiring testable without adding unrelated build/account
tables.

## Trace Shape

Each chat turn writes one `agent_traces` row:

```json
{
  "trace_id": "...",
  "session_id": "...",
  "routed_to": "knowledge",
  "tool_calls": [
    {
      "tool_name": "knowledge",
      "args": {},
      "result": {}
    }
  ],
  "retrieved_chunk_ids": ["chunk_..."],
  "latency_ms": 1234
}
```

## Validation

Run:

```powershell
uv run pytest -q
uv run ruff check app tests
```

Current local result:

```text
7 passed
All checks passed!
```

## Known Limitations

- Live chat depends on a valid Gemini API key and available Gemini quota.
- Retrieval uses deterministic hash embeddings rather than a semantic embedding
  model. This keeps the project self-contained, but a production system should
  use a stronger embedding model.
- Account tools use mocked data rather than a real builds/account schema.
- The ADK session itself is not persisted; compact app-level state is persisted
  and re-injected each turn.
- No streaming, idempotency, escalation agent, Docker, reranker, guardrail, or
  eval harness extensions are implemented.

## What I'd Do With More Time

- Replace hash embeddings with Gemini/OpenAI embeddings and add retrieval evals.
- Add an escalation agent and persisted tickets table.
- Add idempotency keys for chat requests.
- Add SSE streaming support.
- Add guardrails for out-of-scope requests and PII-safe logging.
- Add Docker Compose for reproducible local startup.

## Time Spent

| Phase | Time |
|-------|------|
| Setup + DB + FastAPI boilerplate | 45 min |
| RAG ingest + search_docs | 60 min |
| Account tools | 20 min |
| ADK agents + AgentTool orchestration | 60 min |
| pipeline.py + state persistence + traces | 75 min |
| Tests + lint cleanup | 45 min |
| README | 20 min |
| **Total** | **~5h 25m** |

## Extensions Completed

- [ ] E1: Idempotency
- [ ] E2: Escalation agent
- [ ] E3: Streaming SSE
- [ ] E4: Reranking
- [ ] E5: Guardrails
- [ ] E6: Docker
- [ ] E7: Eval harness
