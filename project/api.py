"""
api.py — Streaming REST API gateway for the CPI RAG system.

Layered architecture reflected in every request:
  Layer 2  Pre-Guardrail   check_domain_guard()  (instant, no LLM, no graph)
  Layer 3  Orchestrator    graph.astream_events() handles internally
  Layer 4  Retriever       graph.astream_events() handles internally
  Layer 5  Post-Guardrail  graph.astream_events() handles internally
  Layer 6  Answer Gen.     token events from aggregate_answers / fallback nodes

Endpoints:
  POST   /v1/chat/stream            Server-Sent Events (SSE) token streaming
  POST   /v1/chat                   Synchronous JSON response
  DELETE /v1/sessions/{thread_id}   Clear conversation history
  GET    /health                    Liveness check

Agent step trace (JSONL, optional):
  Default file: project/logs/agent_steps.jsonl — one JSON object per line for LangGraph node
  boundaries (``on_chain_start`` / ``on_chain_end``) and tool calls (inputs + truncated outputs).
  Disable file: ``AGENT_TRACE_LOG=0``. Custom path: ``AGENT_TRACE_LOG=/var/log/cpi/agent.jsonl``.
  Mirror steps to stdout (``docker logs``): ``AGENT_TRACE_STDOUT=1``.

Stdlib chat summary (``logging`` on stdout): after each ``/v1/chat`` or ``/v1/chat/stream`` turn,
one line ``timestamp | INFO | {<json>}`` with ``query``, ``graph_route`` (LangGraph node visit order,
consecutive duplicates collapsed), ``retrieved_docs``, ``response`` (truncated), ``response_chars``,
and ``status`` (``success`` / ``guardrail_blocked`` / ``error``). Pre-graph guardrail blocks use
``graph_route: []``.

SSE event envelope  (each line is:  data: <json>\\n\\n):
  {"event": "start",     "thread_id": "..."}
  {"event": "guardrail", "status": "passed|blocked_offtopic|blocked_year"}
  {"event": "status",    "message": "human-readable progress hint"}
  {"event": "retrieval", "status": "ok|low_confidence"}
  {"event": "token",     "content": "..."}     ← streamed LLM tokens
  {"event": "done",      "thread_id": "..."}
  {"event": "error",     "message": "..."}     ← on unexpected failure
  data: [DONE]                                 ← stream terminator

Usage:
  python project/api.py                  # default: 0.0.0.0:8000
  python project/api.py --port 9000
  uvicorn project.api:app --reload       # dev mode from repo root
"""

import asyncio
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel

# ── Bootstrap path / env ──────────────────────────────────────────────────────
_project_dir = Path(__file__).resolve().parent
_root_dir = _project_dir.parent
for _p in (_project_dir, _root_dir):
    _env = _p / ".env"
    if _env.exists():
        load_dotenv(_env)
        break

sys.path.insert(0, str(_project_dir))

from core.rag_system import RAGSystem                   # noqa: E402
from rag_agent.nodes import check_domain_guard          # noqa: E402
from agent_trace import record_agent_graph_event       # noqa: E402
from mcp_server import log_summary as _log_summary      # noqa: E402
from mcp_server import export_cpi_data as _mcp_export   # noqa: E402

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="CPI RAG API",
    description=(
        "Streaming AI gateway over Malaysia's annual Consumer Price Index (CPI) "
        "from the DOSM OpenDOSM dataset."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton ─────────────────────────────────────────────────────────────────

_rag: RAGSystem | None = None


def _get_rag() -> RAGSystem:
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialised yet.")
    return _rag


@app.on_event("startup")
async def _startup():
    global _rag
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logger.info("Initialising RAG system …")
    _rag = RAGSystem()
    _rag.initialize()
    logger.info("RAG system ready — collection: {}", _rag.collection_name)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None   # omit to start a new session


class ExportRequest(BaseModel):
    start_year: int
    end_year: int
    division: str = "overall"   # "overall" (default) | "all" | COICOP code 01–13


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    guardrail_status: str


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(event_type: str, payload: dict) -> str:
    """Single SSE line (double-newline terminated)."""
    return f"data: {json.dumps({'event': event_type, **payload})}\n\n"


def _count_retrieved_docs_from_search_output(content: str) -> int:
    """Count ranked chunks in a search_child_chunks tool string (0 if empty / low-confidence)."""
    if not content or "NO_RELEVANT_CHUNKS" in content:
        return 0
    return len(re.findall(r"\[\d+\]\s+relevance=", content))


def _log_chat_request_summary(
    *,
    query: str,
    response: str,
    retrieved_docs: int,
    status: str,
    thread_id: str | None = None,
    guardrail_status: str | None = None,
    error: str | None = None,
    graph_route: list[str] | None = None,
) -> None:
    """One JSON line per chat completion on the stdlib root logger (docker-friendly)."""
    payload: dict[str, Any] = {
        "query": (query or "")[:800],
        "retrieved_docs": retrieved_docs,
        "response": (response or "")[:2000] + ("..." if len(response or "") > 2000 else ""),
        "response_chars": len(response or ""),
        "status": status,
    }
    if graph_route is not None:
        payload["graph_route"] = graph_route
        payload["graph_route_text"] = " → ".join(graph_route) if graph_route else "(no graph — blocked or not run)"
    if thread_id:
        payload["thread_id"] = thread_id
    if guardrail_status:
        payload["guardrail_status"] = guardrail_status
    if error:
        payload["error"] = error
    logging.info("%s", json.dumps(payload, ensure_ascii=False))


# Nodes whose LLM output is the final answer that should be streamed to clients.
_ANSWER_NODES = frozenset({"aggregate_answers", "low_confidence_response", "fallback_response"})

# Human-readable progress messages emitted when a node starts.
_NODE_STATUS: dict[str, str] = {
    "orchestrator":          "Searching knowledge base …",
    "aggregate_answers":     "Generating answer …",
    "low_confidence_response": "Evaluating confidence …",
    "fallback_response":     "Generating best-effort answer …",
}


class _RunAccum:
    __slots__ = ("token_count", "full_answer", "captured_csv", "retrieved_docs", "graph_route")

    def __init__(self) -> None:
        self.token_count = 0
        self.full_answer = ""
        self.captured_csv: str | None = None
        self.retrieved_docs = 0
        self.graph_route: list[str] = []


def _record_graph_route_step(accum: _RunAccum, event: dict) -> None:
    """Append LangGraph node id on each node start (collapse consecutive duplicates)."""
    if event.get("event") != "on_chain_start":
        return
    node = (event.get("metadata") or {}).get("langgraph_node") or ""
    if not node:
        return
    # Drop noisy generic runnables if any slip through with empty semantic name
    if not accum.graph_route or accum.graph_route[-1] != node:
        accum.graph_route.append(node)


def _handle_graph_event(event: dict, accum: _RunAccum, *, emit_sse: bool) -> list[str]:
    """Map one LangGraph astream_events event to SSE lines (if emit_sse) and update accum."""
    out: list[str] = []
    ev_type = event["event"]
    node = event.get("metadata", {}).get("langgraph_node", "")

    if ev_type == "on_chain_start" and node in _NODE_STATUS:
        if emit_sse:
            out.append(_sse("status", {"message": _NODE_STATUS[node]}))

    elif ev_type == "on_tool_end" and event.get("name") == "search_child_chunks":
        raw = event.get("data", {}).get("output", "")
        text = raw.content if hasattr(raw, "content") else str(raw)
        accum.retrieved_docs += _count_retrieved_docs_from_search_output(text)

    elif ev_type == "on_tool_end" and event.get("name") == "export_cpi_data":
        try:
            raw = event.get("data", {}).get("output", "")
            content_str = raw.content if hasattr(raw, "content") else str(raw)
            tool_result = json.loads(content_str)
            accum.captured_csv = tool_result.get("csv_content", "")

            sy = tool_result.get("start_year", "")
            ey = tool_result.get("end_year", "")
            div = tool_result.get("division", "all")
            download_url = f"/v1/export/csv?start_year={sy}&end_year={ey}&division={div}"
            filename = f"cpi_{div}_{sy}_{ey}.csv"

            if emit_sse:
                out.append(
                    _sse(
                        "export_ready",
                        {
                            "download_url": download_url,
                            "filename": filename,
                            "division": div,
                            "division_summary": tool_result.get("division_summary", ""),
                            "start_year": sy,
                            "end_year": ey,
                            "record_count": tool_result.get("record_count", 0),
                        },
                    )
                )
        except Exception:
            pass

    elif ev_type == "on_chain_end" and node == "retrieval_guard":
        o = event.get("data", {}).get("output", {})
        r_status = o.get("retrieval_status", "ok") if isinstance(o, dict) else "ok"
        if emit_sse:
            out.append(_sse("retrieval", {"status": r_status}))

    elif ev_type == "on_chat_model_stream" and node in _ANSWER_NODES:
        chunk = event["data"].get("chunk")
        content = (getattr(chunk, "content", None) or "")
        if content:
            accum.token_count += 1
            accum.full_answer += content
            if emit_sse:
                out.append(_sse("token", {"content": content}))

    return out


def _finalize_graph_run(rag: RAGSystem, config: dict, accum: _RunAccum, *, emit_sse: bool) -> list[str]:
    """Fallback answer from checkpoint state, optional CSV injection, return extra SSE lines."""
    out: list[str] = []
    if accum.token_count == 0:
        state = rag.agent_graph.get_state(config)
        msgs = state.values.get("messages", [])
        if msgs:
            full = getattr(msgs[-1], "content", "") or ""
            if full:
                accum.full_answer = full
                if emit_sse:
                    out.append(_sse("token", {"content": full}))

    if accum.captured_csv and "```csv" not in accum.full_answer:
        csv_injection = f"\n\n```csv\n{accum.captured_csv}```"
        accum.full_answer += csv_injection
        if emit_sse:
            out.append(_sse("token", {"content": csv_injection}))

    return out


async def _event_stream(rag: RAGSystem, message: str, thread_id: str) -> AsyncIterator[str]:
    """Core async generator — maps the 6-layer architecture onto SSE events."""

    yield _sse("start", {"thread_id": thread_id})

    # ── Layer 2 — Pre-Guardrail (zero LLM, instant) ──────────────────────────
    status, blocking_msg = check_domain_guard(message)
    yield _sse("guardrail", {"status": status})

    if status != "passed":
        # Return the blocking message as a single token for consistent UX.
        _log_chat_request_summary(
            query=message,
            response=blocking_msg,
            retrieved_docs=0,
            status="guardrail_blocked",
            thread_id=thread_id,
            guardrail_status=status,
            graph_route=[],
        )
        yield _sse("token", {"content": blocking_msg})
        yield _sse("done",  {"thread_id": thread_id})
        yield "data: [DONE]\n\n"
        return

    # ── Layers 3-6 — LangGraph pipeline ──────────────────────────────────────
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    input_data = {"messages": [HumanMessage(content=message.strip())]}
    accum = _RunAccum()

    try:
        async for event in rag.agent_graph.astream_events(input_data, config, version="v2"):
            record_agent_graph_event(thread_id, message, event)
            _record_graph_route_step(accum, event)
            for line in _handle_graph_event(event, accum, emit_sse=True):
                yield line

        for line in _finalize_graph_run(rag, config, accum, emit_sse=True):
            yield line

        # MCP log_summary — fire-and-forget audit log after every response
        try:
            _log_summary(query=message, answer=accum.full_answer, sources="CPI 2D Annual.csv")
        except Exception:
            pass

        _log_chat_request_summary(
            query=message,
            response=accum.full_answer,
            retrieved_docs=accum.retrieved_docs,
            status="success",
            thread_id=thread_id,
            graph_route=list(accum.graph_route),
        )

        yield _sse("done", {"thread_id": thread_id})
        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.exception("Stream error — thread_id={}", thread_id)
        _log_chat_request_summary(
            query=message,
            response="",
            retrieved_docs=getattr(accum, "retrieved_docs", 0),
            status="error",
            thread_id=thread_id,
            error=str(exc),
            graph_route=list(getattr(accum, "graph_route", [])),
        )
        yield _sse("error", {"message": str(exc)})
        yield "data: [DONE]\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/v1/chat/stream",
    summary="SSE streaming chat",
    response_description="text/event-stream with token-level SSE events",
)
async def chat_stream(req: ChatRequest):
    """Stream the RAG response as Server-Sent Events.

    Clients receive events in order:
    1. `start`      → session info
    2. `guardrail`  → pre-flight result (passed / blocked)
    3. `status`     → progress hints (optional)
    4. `retrieval`  → Qdrant result confidence
    5. `token`      → streamed LLM tokens (multiple)
    6. `done`       → stream complete
    """
    rag = _get_rag()
    thread_id = req.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _event_stream(rag, req.message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx/proxy buffering
            "Connection":       "keep-alive",
        },
    )


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    summary="Synchronous chat (waits for complete answer)",
)
async def chat_sync(req: ChatRequest):
    """Non-streaming endpoint — blocks until the full answer is ready.

    Useful for services that don't support SSE or need a single JSON payload.
    """
    rag = _get_rag()
    thread_id = req.thread_id or str(uuid.uuid4())

    # Layer 2 pre-check
    status, blocking_msg = check_domain_guard(req.message)
    if status != "passed":
        _log_chat_request_summary(
            query=req.message,
            response=blocking_msg,
            retrieved_docs=0,
            status="guardrail_blocked",
            thread_id=thread_id,
            guardrail_status=status,
            graph_route=[],
        )
        return ChatResponse(reply=blocking_msg, thread_id=thread_id, guardrail_status=status)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    input_data = {"messages": [HumanMessage(content=req.message.strip())]}
    accum = _RunAccum()

    try:
        async for event in rag.agent_graph.astream_events(input_data, config, version="v2"):
            record_agent_graph_event(thread_id, req.message, event)
            _record_graph_route_step(accum, event)
            _handle_graph_event(event, accum, emit_sse=False)

        for _ in _finalize_graph_run(rag, config, accum, emit_sse=False):
            pass

        try:
            _log_summary(query=req.message, answer=accum.full_answer, sources="CPI 2D Annual.csv")
        except Exception:
            pass

        state = rag.agent_graph.get_state(config)
        g_stat = (state.values or {}).get("guardrail_status", "passed")
        _log_chat_request_summary(
            query=req.message,
            response=accum.full_answer,
            retrieved_docs=accum.retrieved_docs,
            status="success",
            thread_id=thread_id,
            guardrail_status=g_stat if isinstance(g_stat, str) else "passed",
            graph_route=list(accum.graph_route),
        )
        return ChatResponse(
            reply=accum.full_answer,
            thread_id=thread_id,
            guardrail_status=g_stat if isinstance(g_stat, str) else "passed",
        )
    except Exception as exc:
        logger.exception("Sync chat error — thread_id={}", thread_id)
        _log_chat_request_summary(
            query=req.message,
            response="",
            retrieved_docs=getattr(accum, "retrieved_docs", 0),
            status="error",
            thread_id=thread_id,
            error=str(exc),
            graph_route=list(getattr(accum, "graph_route", [])),
        )
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete(
    "/v1/sessions/{thread_id}",
    status_code=204,
    summary="Clear conversation history",
)
async def clear_session(thread_id: str):
    """Delete the LangGraph checkpoint for a given thread_id.

    Call this to reset conversation history between sessions.
    """
    rag = _get_rag()
    try:
        rag.agent_graph.checkpointer.delete_thread(thread_id)
    except Exception:
        pass   # ignore missing threads


def _build_csv_response(start_year: int, end_year: int, division: str):
    """Shared logic for both POST and GET CSV export endpoints."""
    import json
    from fastapi.responses import Response

    result = json.loads(_mcp_export(start_year=start_year, end_year=end_year, division=division))

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    csv_content: str = result["csv_content"]
    div          = result["division"]
    start        = result["start_year"]
    end          = result["end_year"]
    filename     = f"cpi_{div}_{start}_{end}.csv"

    safe_source  = (result.get("source", "")).encode("ascii", "ignore").decode()
    safe_license = (result.get("license", "")).encode("ascii", "ignore").decode()

    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
            "X-Source":  safe_source,
            "X-License": safe_license,
        },
    )


@app.post(
    "/v1/export/csv",
    summary="CSV export via POST body (curl / server-to-server)",
    response_description="text/csv file download",
)
async def export_csv_post(req: ExportRequest):
    """Download CPI data as a raw CSV file — no LLM involved.

    Use this from curl or server-to-server calls where you can send a JSON body.

    ```bash
    curl -X POST http://localhost:8000/v1/export/csv \\
      -H "Content-Type: application/json" \\
      -d '{"start_year": 2021, "end_year": 2024, "division": "all"}' \\
      -o cpi_all_2021_2024.csv
    ```
    """
    return _build_csv_response(req.start_year, req.end_year, req.division)


@app.get(
    "/v1/export/csv",
    summary="CSV export via GET query params (browser download link)",
    response_description="text/csv file download",
)
async def export_csv_get(
    start_year: int,
    end_year: int,
    division: str = "overall",
):
    """Download CPI data as a CSV file using query parameters.

    This endpoint works as a plain browser link or ``<a href>`` tag —
    no JavaScript required.

    Examples:
    ```
    # All divisions, 2021-2024
    GET /v1/export/csv?start_year=2021&end_year=2024

    # Transport only, 2010-2024
    GET /v1/export/csv?start_year=2010&end_year=2024&division=07

    # Overall headline, 2000-2024
    GET /v1/export/csv?start_year=2000&end_year=2024&division=overall
    ```

    Browser usage (HTML):
    ```html
    <a href="http://localhost:8000/v1/export/csv?start_year=2021&end_year=2024">
      Download CPI 2021-2024 (all divisions)
    </a>
    ```

    JavaScript fetch:
    ```js
    const res = await fetch(
      'http://localhost:8000/v1/export/csv?start_year=2021&end_year=2024'
    );
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = res.headers.get('content-disposition').match(/filename="(.+)"/)[1];
    a.click();
    ```
    """
    return _build_csv_response(start_year, end_year, division)


@app.get("/health", summary="Liveness check")
async def health():
    """Returns 200 when the RAG system is initialised and ready."""
    rag = _get_rag()
    return {"status": "ok", "collection": rag.collection_name}


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="CPI RAG API server")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Hot-reload (dev only)")
    args = parser.parse_args()

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
