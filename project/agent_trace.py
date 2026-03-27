"""
Append-only JSONL trace of LangGraph / agent execution steps.

Default log: project/logs/agent_steps.jsonl
  This is written to a **file** inside the container — it does **not** appear in ``docker logs``.
  To mirror each step line to stdout (visible in ``docker logs -f``), set::

    AGENT_TRACE_STDOUT=1

Override path: ``AGENT_TRACE_LOG`` (absolute or repo-relative). Disable file only: ``AGENT_TRACE_LOG=0``
(still mirrors to stdout if ``AGENT_TRACE_STDOUT=1``).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_DEFAULT_LOG = _LOG_DIR / "agent_steps.jsonl"


def _trace_path() -> Path | None:
    raw = os.getenv("AGENT_TRACE_LOG", "").strip()
    if raw.lower() in ("0", "false", "no", "off"):
        return None
    if not raw:
        return _DEFAULT_LOG
    p = Path(raw)
    return p if p.is_absolute() else Path(__file__).resolve().parent.parent / p


def _trace_stdout_enabled() -> bool:
    return os.getenv("AGENT_TRACE_STDOUT", "").strip().lower() in ("1", "true", "yes", "on")


def _truncate(s: str, max_len: int = 2000) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... [{len(s)} chars total]"


def _serialize_tool_output(out) -> str | None:
    if out is None:
        return None
    if hasattr(out, "content"):
        return _truncate(str(getattr(out, "content", "")))
    return _truncate(str(out))


def _serialize_tool_input(inp) -> object:
    if inp is None:
        return None
    if isinstance(inp, dict):
        out: dict = {}
        for k, v in inp.items():
            if isinstance(v, str):
                out[k] = _truncate(v, 800)
            else:
                try:
                    json.dumps(v)
                    out[k] = v
                except TypeError:
                    out[k] = _truncate(str(v), 400)
        return out
    return _truncate(str(inp), 1200)


def record_agent_graph_event(
    thread_id: str,
    user_message: str,
    event: dict,
) -> None:
    """Append one JSON line for tool calls and LangGraph node boundaries."""
    path = _trace_path()
    to_stdout = _trace_stdout_enabled()
    if path is None and not to_stdout:
        return

    ev_type = event.get("event") or ""
    if ev_type == "on_chat_model_stream":
        return

    meta = event.get("metadata") or {}
    node = meta.get("langgraph_node") or ""
    name = event.get("name") or ""

    row: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "user_message_preview": (user_message or "")[:300],
        "event": ev_type,
        "langgraph_node": node,
        "name": name,
    }

    data = event.get("data") or {}

    if ev_type == "on_tool_start":
        row["tool"] = name
        row["tool_input"] = _serialize_tool_input(data.get("input"))
    elif ev_type == "on_tool_end":
        row["tool"] = name
        row["tool_output_preview"] = _serialize_tool_output(data.get("output"))
    elif ev_type == "on_chain_start":
        if not node:
            return
        row["phase"] = "node_start"
    elif ev_type == "on_chain_end":
        if not node:
            return
        row["phase"] = "node_end"
        out = data.get("output")
        if isinstance(out, dict):
            row["output_keys"] = list(out.keys())[:48]
    else:
        return

    line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
    if to_stdout:
        try:
            print(line, end="", flush=True)
        except Exception:
            pass
