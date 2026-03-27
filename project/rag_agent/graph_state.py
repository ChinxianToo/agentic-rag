from typing import List, Annotated, Set
from langgraph.graph import MessagesState
import operator


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get("__reset__") for item in new):
        return []
    return existing + new


def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b


# ─────────────────────────────────────────────────────────────────
# Layer 2 + 3 + 6 state  — outer graph (pre-guard → orchestrator → aggregator)
# ─────────────────────────────────────────────────────────────────
class State(MessagesState):
    """Outer-graph state.

    Architecture layer mapping
    ──────────────────────────
    Layer 1  User / Chat UI          - messages field (inherited)
    Layer 2  Pre-Guardrail           - guardrail_status written by domain_guard
    Layer 3  LangGraph Orchestrator  - conversation_summary, questionIsClear,
                                       originalQuery, rewrittenQuestions
    Layer 6  Final response          - agent_answers, messages (last entry)
    """
    # Layer 2 — pre-guardrail outcome
    guardrail_status: str = "pending"   # "passed" | "blocked_offtopic" | "blocked_year"

    # Layer 3 — query understanding / orchestration
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []

    # Layer 6 — aggregated answers from agent sub-graphs
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []


# ─────────────────────────────────────────────────────────────────
# Layer 4 + 5 state  — agent sub-graph (retriever → post-guard → answer)
# ─────────────────────────────────────────────────────────────────
class AgentState(MessagesState):
    """Inner-graph (agent sub-graph) state.

    Architecture layer mapping
    ──────────────────────────
    Layer 4  Retriever Layer     – retrieval_keys (tracks calls already made)
    Layer 5  Post-Guardrail      – retrieval_status written by retrieval_guard
    Layer 6  LLM Answer Gen.     – context_summary, final_answer, agent_answers
    """
    question: str = ""
    question_index: int = 0
    # Preserved from the original user message so export-intent survives rewriting
    original_question: str = ""

    # Layer 4 — retriever tracking (dedup, compression)
    retrieval_keys: Annotated[Set[str], set_union] = set()
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0

    # Layer 5 — post-retrieval guardrail outcome
    retrieval_status: str = "pending"   # "ok" | "low_confidence"

    # Layer 6 — answer generation
    context_summary: str = ""
    final_answer: str = ""
    agent_answers: List[dict] = []
