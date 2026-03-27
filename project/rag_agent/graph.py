"""
graph.py — Builds the two-tier LangGraph RAG graph.

Layered architecture implemented here
──────────────────────────────────────
  Layer 1  User / Chat UI          - app.py (entry point, outside this graph)
  ┌─────────────────────────────────────────── OUTER GRAPH ───┐
  │ Layer 2  Pre-Guardrail         - domain_guard             │ ← FIRST node
  │ Layer 3  LangGraph Orchestrator- summarize_history        │
  │                                  rewrite_query            │
  │                                  request_clarification    │
  │  ┌──────────────────────── AGENT SUB-GRAPH ─────────┐     │
  │  │ Layer 4  Retriever          - ToolNode           │     │
  │  │                               (search + retrieve)│     │
  │  │ Layer 5  Post-Guardrail     - retrieval_guard    │     │
  │  │                               low_confidence_resp│     │
  │  │ Layer 6  LLM Answer Gen.    - orchestrator       │     │
  │  │                               compress_context   │     │
  │  │                               fallback_response  │     │
  │  │                               collect_answer     │     │
  │  └──────────────────────────────────────────────────┘     │
  │ Layer 6  Final Aggregation     - aggregate_answers        │
  └───────────────────────────────────────────────────────────┘
  Layer 6  Final Response to User - messages[-1] returned by app.py
"""

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import (
    aggregate_answers,
    collect_answer,
    compress_context,
    domain_guard,
    fallback_response,
    low_confidence_response,
    orchestrator,
    request_clarification,
    retrieval_guard,
    rewrite_query,
    should_compress_context,
    summarize_history,
)
from .edges import route_after_rewrite, route_after_orchestrator_call
from .graph_state import AgentState, State


def create_agent_graph(llm, tools_list):
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)
    checkpointer = InMemorySaver()

    print("Compiling agent graph...")

    # ── Agent sub-graph (Layers 4 · 5 · 6) ──────────────────────────────────
    agent = StateGraph(AgentState)

    # Layer 4 — Retriever
    agent.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
    agent.add_node("tools", tool_node)                                   # ← Qdrant search

    # Layer 5 — Post-Guardrail
    agent.add_node("retrieval_guard", retrieval_guard)
    agent.add_node("low_confidence_response", partial(low_confidence_response, llm=llm))

    # Layer 6 — LLM Answer Generator
    agent.add_node("compress_context", partial(compress_context, llm=llm))
    agent.add_node("fallback_response", partial(fallback_response, llm=llm))
    agent.add_node(should_compress_context)
    agent.add_node(collect_answer)

    # Sub-graph edges
    agent.add_edge(START, "orchestrator")
    agent.add_conditional_edges(
        "orchestrator", route_after_orchestrator_call,
        {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"},
    )
    # Layer 4 → Layer 5
    agent.add_edge("tools", "retrieval_guard")
    # Layer 5 paths
    agent.add_edge("low_confidence_response", "collect_answer")
    # Layer 5 → Layer 6 (context/answer loop)
    agent.add_edge("compress_context", "orchestrator")
    agent.add_edge("fallback_response", "collect_answer")
    agent.add_edge("collect_answer", END)

    agent_subgraph = agent.compile()

    # ── Outer graph (Layers 2 · 3 · 6 aggregation) ───────────────────────────
    outer = StateGraph(State)

    # Layer 2 — Pre-Guardrail (FIRST — before any LLM call)
    outer.add_node("domain_guard", domain_guard)

    # Layer 3 — LangGraph Orchestrator
    outer.add_node("summarize_history", partial(summarize_history, llm=llm))
    outer.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    outer.add_node(request_clarification)

    # Layer 4-6 sub-graph
    outer.add_node("agent", agent_subgraph)

    # Layer 6 — Final aggregation
    outer.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))

    # Outer edges — follows the architecture top-to-bottom
    outer.add_edge(START, "domain_guard")               # Layer 2 first
    # domain_guard routes via Command → "summarize_history" or END
    outer.add_edge("summarize_history", "rewrite_query")  # Layer 3
    outer.add_conditional_edges("rewrite_query", route_after_rewrite)
    outer.add_edge("request_clarification", "rewrite_query")
    outer.add_edge(["agent"], "aggregate_answers")        # Layer 6
    outer.add_edge("aggregate_answers", END)

    graph = outer.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"],
    )
    print("✓ Agent graph compiled successfully.")
    return graph
