"""
nodes.py — LangGraph node implementations.

Layered architecture
────────────────────
  Layer 1  User / Chat UI          app.py (outside this file)
  Layer 2  Pre-Guardrail           domain_guard
  Layer 3  LangGraph Orchestrator  summarize_history · rewrite_query · request_clarification
  Layer 4  Retriever Layer         (tools.py  ─ search_child_chunks / retrieve_parent_chunks)
  Layer 5  Post-Guardrail          retrieval_guard · low_confidence_response
  Layer 6  LLM Answer Generator    orchestrator · fallback_response · compress_context
                                   · collect_answer · aggregate_answers
"""

import re
from typing import Literal, Set

from langchain_core.messages import (
    AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage,
)
from langgraph.graph import END
from langgraph.types import Command

from .graph_state import AgentState, State
from .prompts import (
    get_aggregation_prompt,
    get_context_compression_prompt,
    get_conversation_summary_prompt,
    get_fallback_response_prompt,
    get_low_confidence_prompt,
    get_orchestrator_prompt,
    get_rewrite_query_prompt,
)
from .schemas import QueryAnalysis
from utils import estimate_context_tokens
from config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR

# Phrases / tokens that indicate the user wants a downloadable CSV (checked with substring search).
_EXPORT_KEYWORDS = frozenset({
    "export",
    "download",
    "csv",
    "spreadsheet",
    "excel",
    "give me data",
    "send me the data",
    "attach the",
    "raw data",
    "dump the",
    "pull the data",
    "get the data as",
})


def _user_wants_export(state: AgentState) -> bool:
    blob = " ".join([
        (state.get("original_question") or ""),
        (state.get("question") or ""),
    ]).lower()
    return any(kw in blob for kw in _EXPORT_KEYWORDS)


def _export_cpi_tool_already_ran(messages: list) -> bool:
    for m in messages or []:
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "export_cpi_data":
            return True
    return False


# ══════════════════════════════════════════════════════════════════
# LAYER 2 — Pre-Guardrail
# Fast, zero-LLM checks that run BEFORE the LangGraph orchestrator.
# Rejects off-topic or out-of-range queries immediately.
# ══════════════════════════════════════════════════════════════════

_DOMAIN_KEYWORDS: frozenset[str] = frozenset([
    # Dataset / source
    "cpi", "consumer price", "price index", "inflation", "dosm", "opendosm",
    # Geography
    "malaysia", "malaysian",
    # Time tokens covered by the dataset
    "annual", "yearly", "year", "1960", "1970", "1980", "1990", "2000",
    "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018",
    "2019", "2020", "2021", "2022", "2023", "2024", "2025",
    # COICOP divisions & common aliases
    "overall", "division", "food", "beverage", "f&b", "fnb",
    "alcohol", "tobacco", "clothing", "footwear", "housing", "water",
    "electricity", "furnishing", "household", "health", "transport",
    "communication", "recreation", "culture", "education",
    "restaurant", "hotel", "miscellaneous",
    # Generic stat terms
    "index", "value", "statistic", "compare", "trend", "change", "growth",
    "increase", "decrease", "rise", "fall", "rate",
])

# Inclusive year range of the indexed dataset. Update when re-ingesting.
_DATASET_MIN_YEAR = 1960
_DATASET_MAX_YEAR = 2025

_OFFTOPIC_REFUSAL = (
    "I can only answer questions about Malaysia's **Consumer Price Index (CPI)** data "
    "from the DOSM annual dataset (OpenDOSM). Your question appears to be outside that scope.\n\n"
    "Here are some things I *can* help with:\n"
    "- *What was the overall CPI in 2020?*\n"
    "- *Compare food & beverage CPI between 2015 and 2022.*\n"
    "- *Which division had the highest CPI growth from 2010 to 2024?*\n"
    "- *Show me the transport CPI trend from 2000 onwards.*"
)


def _extract_years(text: str) -> list[int]:
    return [int(y) for y in re.findall(r"\b(1[0-9]{3}|2[0-9]{3})\b", text)]


def check_domain_guard(text: str) -> tuple[str, str]:
    """Pure-Python domain guard check — no LangGraph, no LLM.

    Called by both the LangGraph ``domain_guard`` node and the streaming API
    pre-flight so blocked queries never reach the graph.

    Returns:
        (status, blocking_message)
        status is one of: "passed" | "blocked_offtopic" | "blocked_year"
        blocking_message is empty when status == "passed".
    """
    lowered = text.lower()

    # Check 1 — off-topic scope
    if not any(kw in lowered for kw in _DOMAIN_KEYWORDS):
        return "blocked_offtopic", _OFFTOPIC_REFUSAL

    # Check 2 — year range
    years = _extract_years(lowered)
    out_of_range = [y for y in years if y > _DATASET_MAX_YEAR or y < _DATASET_MIN_YEAR]
    if out_of_range:
        oor_str = ", ".join(str(y) for y in sorted(set(out_of_range)))
        msg = (
            f"The indexed dataset covers **{_DATASET_MIN_YEAR}–{_DATASET_MAX_YEAR}** only. "
            f"Data for **{oor_str}** is not available in the current extract.\n\n"
            f"The most recent year available is **{_DATASET_MAX_YEAR}**. "
            f"You can ask about any year between {_DATASET_MIN_YEAR} and {_DATASET_MAX_YEAR}."
        )
        return "blocked_year", msg

    return "passed", ""


def domain_guard(state: State) -> Command[Literal["summarize_history", "__end__"]]:
    """Layer 2 — Pre-Guardrail (LangGraph node wrapper around check_domain_guard).

    Runs FIRST — before summarize_history and any LLM call.
    Delegates all logic to check_domain_guard() so the same rules apply
    whether the caller is the graph or the streaming API pre-flight.
    """
    last = state["messages"][-1]
    status, msg = check_domain_guard(last.content or "")

    if status != "passed":
        return Command(
            update={
                "guardrail_status": status,
                "messages": [AIMessage(content=msg)],
            },
            goto=END,
        )

    return Command(
        update={"guardrail_status": "passed"},
        goto="summarize_history",
    )


# ══════════════════════════════════════════════════════════════════
# LAYER 3 — LangGraph Orchestrator
# Manages conversation state: history summarisation, query rewriting,
# and clarification when the query is ambiguous.
# ══════════════════════════════════════════════════════════════════

def summarize_history(state: State, llm):
    """Layer 3 — condense prior conversation to keep context window lean."""
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]
    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    response = llm.with_config(temperature=0.2).invoke(
        [SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)]
    )
    return {"conversation_summary": response.content, "agent_answers": [{"__reset__": True}]}


def rewrite_query(state: State, llm):
    """Layer 3 — rewrite the user query into one or more retrieval-ready sub-queries."""
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (
        (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "")
        + f"User Query:\n{last_message.content}\n"
    )

    llm_structured = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_structured.invoke(
        [SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)]
    )

    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions,
        }

    clarification = (
        response.clarification_needed
        if response.clarification_needed and len(response.clarification_needed.strip()) > 10
        else "I need more information to understand your question."
    )
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}


def request_clarification(state: State):
    """Layer 3 — interrupt point; human-in-the-loop clarification before re-entering retrieval."""
    return {}


# ══════════════════════════════════════════════════════════════════
# LAYER 4 — Retriever Layer   (tool execution lives in tools.py)
# The orchestrator node drives tool calls; results feed Layer 5.
# ══════════════════════════════════════════════════════════════════

def orchestrator(state: AgentState, llm_with_tools):
    """Layer 4 / 6 — decides which tool to call (retrieval) then, once
    enough context is gathered, writes the grounded answer (generation).
    """
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )

    is_export = _user_wants_export(state)
    msgs = state.get("messages") or []
    export_done = _export_cpi_tool_already_ran(msgs)

    if not msgs:
        human_msg = HumanMessage(content=state["question"])
        force_step = HumanMessage(content=(
            "FIRST TOOL CALL: The user asked for an export/download/CSV. "
            "Call `export_cpi_data` with start_year, end_year, and division (COICOP: food→01, education→10, transport→07, overall headline→overall). "
            "Do NOT call search_child_chunks."
            if is_export else
            "FIRST TOOL CALL: Call `search_child_chunks` now to retrieve relevant CPI data before answering."
        ))
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_step])
        return {
            "messages": [human_msg, response],
            "tool_call_count": len(response.tool_calls or []),
            "iteration_count": 1,
        }

    # Export intent was lost on later turns if the model searched first — keep nudging until export runs.
    if is_export and not export_done:
        nudge = HumanMessage(
            content=(
                "CRITICAL: The user requested an **export** (downloadable data / CSV). "
                "Call `export_cpi_data` now with the correct start_year, end_year, and division. "
                "Do NOT call search_child_chunks or retrieve_parent_chunks. "
                "After the tool returns, your final answer MUST include the complete tool `csv_content` inside a ```csv fenced code block."
            )
        )
        response = llm_with_tools.invoke([sys_msg] + summary_injection + msgs + [nudge])
    else:
        response = llm_with_tools.invoke([sys_msg] + summary_injection + msgs)

    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {
        "messages": [response],
        "tool_call_count": len(tool_calls) if tool_calls else 0,
        "iteration_count": 1,
    }


# ══════════════════════════════════════════════════════════════════
# LAYER 5 — Post-Guardrail / Relevance Check
# Inspects retrieval output; routes low-confidence results to a
# clarify/refuse path rather than allowing hallucinated answers.
# ══════════════════════════════════════════════════════════════════

def _latest_tool_messages_after_last_ai(messages: list) -> list:
    """Return ToolMessages that follow the most recent tool-calling AIMessage."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tail = []
            for j in range(i + 1, len(messages)):
                if isinstance(messages[j], ToolMessage):
                    tail.append(messages[j])
                elif isinstance(messages[j], AIMessage):
                    break
            return tail
    return []


def _is_low_confidence(content: str) -> bool:
    if not content or not content.strip():
        return True
    if "NO_RELEVANT_CHUNKS" in content:
        return True
    if content.strip().startswith("<<<RETRIEVAL_STATUS:LOW_CONFIDENCE"):
        return True
    return False


def retrieval_guard(
    state: AgentState,
) -> Command[Literal["should_compress_context", "low_confidence_response"]]:
    """Layer 5 — Post-Guardrail.

    Examines the most recent search_child_chunks result.
    • LOW_CONFIDENCE → low_confidence_response (clarify / refuse; no hallucination).
    • OK             → should_compress_context  (continue to answer generation).
    Sets retrieval_status in state for traceability.
    """
    tail = _latest_tool_messages_after_last_ai(state["messages"])
    for msg in tail:
        if not isinstance(msg, ToolMessage):
            continue
        name = getattr(msg, "name", "") or ""
        if name == "retrieve_parent_chunks":
            continue
        if name == "search_child_chunks" or "<<<RETRIEVAL_STATUS:" in (msg.content or ""):
            if _is_low_confidence(msg.content or ""):
                return Command(
                    update={"retrieval_status": "low_confidence"},
                    goto="low_confidence_response",
                )
    return Command(
        update={"retrieval_status": "ok"},
        goto="should_compress_context",
    )


async def low_confidence_response(state: AgentState, llm):
    """Layer 5 — politely clarify or refuse when retrieval confidence is too low.

    Async so astream_events can emit on_chat_model_stream token events.
    """
    tail = _latest_tool_messages_after_last_ai(state["messages"])
    tool_blob = "\n".join((m.content or "") for m in tail if isinstance(m, ToolMessage))
    q = state.get("question", "")
    full = ""
    async for chunk in llm.astream([
        SystemMessage(content=get_low_confidence_prompt()),
        HumanMessage(content=f"User question:\n{q}\n\nRetrieval output:\n{tool_blob}"),
    ]):
        full += (chunk.content or "")
    return {"messages": [AIMessage(content=full)]}


# ══════════════════════════════════════════════════════════════════
# LAYER 6 — LLM Answer Generator + context management helpers
# ══════════════════════════════════════════════════════════════════

def should_compress_context(
    state: AgentState,
) -> Command[Literal["compress_context", "orchestrator"]]:
    """Layer 6 helper — decide whether to compress history before the next orchestrator call."""
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)
                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    token_msgs = estimate_context_tokens(messages)
    token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    total_tokens = token_msgs + token_summary
    max_allowed = BASE_TOKEN_THRESHOLD + int(token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if total_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)


def compress_context(state: AgentState, llm):
    """Layer 6 helper — compress the running message history into a summary to save tokens."""
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()
    if not messages:
        return {}

    text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_info = f" | Tool calls: {calls}"
            text += f"[ASSISTANT{tool_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            text += f"[TOOL RESULT — {getattr(msg, 'name', 'tool')}]\n{msg.content}\n\n"

    response = llm.invoke(
        [SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=text)]
    )
    new_summary = response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_qs = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))
        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(
                f"- {p.replace('parent::', '')}" for p in parent_ids
            ) + "\n"
        if search_qs:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_qs) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}


async def fallback_response(state: AgentState, llm):
    """Layer 6 — best-effort answer when the agent exhausts its iteration/tool budget.

    Async so astream_events can emit on_chat_model_stream token events.
    """
    seen: set = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()
    parts = []
    if context_summary:
        parts.append(f"## Compressed Research Context\n\n{context_summary}")
    if unique_contents:
        parts.append(
            "## Retrieved Data (current iteration)\n\n"
            + "\n\n".join(f"--- DATA SOURCE {i} ---\n{c}" for i, c in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(parts) if parts else "No data was retrieved from the documents."
    prompt = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        "INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    full = ""
    async for chunk in llm.astream([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt)]):
        full += (chunk.content or "")
    return {"messages": [AIMessage(content=full)]}


def collect_answer(state: AgentState):
    """Layer 6 — package the final answer into the accumulator for the outer graph."""
    last = state["messages"][-1]
    is_valid = isinstance(last, AIMessage) and last.content and not last.tool_calls
    answer = last.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}],
    }


async def aggregate_answers(state: State, llm):
    """Layer 6 — synthesise all sub-query answers into one coherent final response.

    Async so astream_events can emit on_chat_model_stream token events —
    these are the tokens streamed to the API client.
    """
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])
    formatted = "".join(f"\nAnswer {i}:\n{ans['answer']}\n" for i, ans in enumerate(sorted_answers, 1))

    user_msg = HumanMessage(
        content=f"Original user question: {state['originalQuery']}\nRetrieved answers:{formatted}"
    )
    full = ""
    async for chunk in llm.astream([SystemMessage(content=get_aggregation_prompt()), user_msg]):
        full += (chunk.content or "")
    return {"messages": [AIMessage(content=full)]}
