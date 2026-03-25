import re
from typing import Literal, Set
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command
from .graph_state import State, AgentState
from .schemas import QueryAnalysis
from .prompts import *
from utils import estimate_context_tokens
from config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR

# ---------------------------------------------------------------------------
# Domain guard — fast, no-LLM pre-flight checks (runs before ANY LLM call).
# Handles two cases:
#   1. Off-topic query → polite refusal explaining what the bot covers.
#   2. Out-of-range year → immediate "data not available" without touching the LLM.
# ---------------------------------------------------------------------------
_DOMAIN_KEYWORDS: frozenset[str] = frozenset([
    # Dataset / source
    "cpi", "consumer price", "price index", "inflation", "dosm", "opendosm",
    # Geography
    "malaysia", "malaysian",
    # Time-related (years covered by the dataset)
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

# Inclusive year range of the indexed dataset (update if you re-ingest with more data).
_DATASET_MIN_YEAR = 1960
_DATASET_MAX_YEAR = 2025

_REFUSAL_MSG = (
    "I can only answer questions about Malaysia's **Consumer Price Index (CPI)** data "
    "from the DOSM annual dataset (OpenDOSM). Your question appears to be outside that scope.\n\n"
    "Here are some things I *can* help with:\n"
    "- *What was the overall CPI in 2020?*\n"
    "- *Compare food & beverage CPI between 2015 and 2022.*\n"
    "- *Which division had the highest CPI growth from 2010 to 2024?*\n"
    "- *Show me the transport CPI trend from 2000 onwards.*"
)


def _extract_years(text: str) -> list[int]:
    """Extract 4-digit years from query text."""
    return [int(y) for y in re.findall(r"\b(1[0-9]{3}|2[0-9]{3})\b", text)]


def domain_guard(state: State) -> Command[Literal["rewrite_query", "__end__"]]:
    """Fast (no-LLM) domain + year-range guard. Short-circuits before any LLM call."""
    last = state["messages"][-1]
    text = (last.content or "").lower()

    # 1. Off-topic: no domain keywords present → refuse
    if not any(kw in text for kw in _DOMAIN_KEYWORDS):
        return Command(
            update={"messages": [AIMessage(content=_REFUSAL_MSG)]},
            goto=END,
        )

    # 2. Year-range: query explicitly mentions a year outside the dataset
    years = _extract_years(text)
    out_of_range = [y for y in years if y > _DATASET_MAX_YEAR or y < _DATASET_MIN_YEAR]
    if out_of_range:
        oor_str = ", ".join(str(y) for y in sorted(set(out_of_range)))
        msg = (
            f"The indexed dataset covers **{_DATASET_MIN_YEAR}-{_DATASET_MAX_YEAR}** only. "
            f"Data for **{oor_str}** is not available in the current extract.\n\n"
            f"The most recent year available is **{_DATASET_MAX_YEAR}**. "
            f"You can ask about any year between {_DATASET_MIN_YEAR} and {_DATASET_MAX_YEAR}."
        )
        return Command(
            update={"messages": [AIMessage(content=msg)]},
            goto=END,
        )

    # 3. In-scope, in-range → proceed to LLM pipeline
    return Command(goto="rewrite_query")


def summarize_history(state: State, llm):
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

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def rewrite_query(state: State, llm):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])

    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {"questionIsClear": True, "messages": delete_all, "originalQuery": last_message.content, "rewrittenQuestions": response.questions}

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "I need more information to understand your question."
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}

def request_clarification(state: State):
    return {}

# --- Agent Nodes ---
def orchestrator(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

def fallback_response(state: AgentState, llm):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}


def _latest_tool_messages_after_last_ai(messages) -> list:
    """ToolMessages following the most recent AIMessage that requested tools."""
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


def _is_low_confidence_search(content: str) -> bool:
    if not content or not content.strip():
        return True
    if "NO_RELEVANT_CHUNKS" in content:
        return True
    if content.strip().startswith("<<<RETRIEVAL_STATUS:LOW_CONFIDENCE"):
        return True
    return False


def retrieval_guard(state: AgentState) -> Command[Literal["should_compress_context", "low_confidence_response"]]:
    """Route low-confidence retrieval to a dedicated clarify/refuse node (LangGraph guardrail)."""
    tail = _latest_tool_messages_after_last_ai(state["messages"])
    for msg in tail:
        if not isinstance(msg, ToolMessage):
            continue
        name = getattr(msg, "name", "") or ""
        if name == "retrieve_parent_chunks":
            continue
        if name == "search_child_chunks" or "<<<RETRIEVAL_STATUS:" in (msg.content or ""):
            if _is_low_confidence_search(msg.content or ""):
                return Command(goto="low_confidence_response")
    return Command(goto="should_compress_context")


def low_confidence_response(state: AgentState, llm):
    tail = _latest_tool_messages_after_last_ai(state["messages"])
    tool_blob = "\n".join((m.content or "") for m in tail if isinstance(m, ToolMessage))
    q = state.get("question", "")
    response = llm.invoke([
        SystemMessage(content=get_low_confidence_prompt()),
        HumanMessage(content=f"User question:\n{q}\n\nRetrieval output:\n{tool_blob}"),
    ])
    return {"messages": [response]}


def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
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

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState, llm):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }
# --- End of Agent Nodes---

def aggregate_answers(state: State, llm):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}