import json
import logging
import re
from typing import Any, List

import config
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager

# MCP tool functions — imported from the FastMCP server so the same logic
# is used whether called by the LangGraph agent (LangChain wrapper) or an
# external MCP client talking to the standalone mcp_server.py process.
from mcp_server import export_cpi_data as _mcp_export_cpi_data
from mcp_server import log_summary as _mcp_log_summary

_tools_log = logging.getLogger("cpi_rag.tools")


def _log_tool_line(event: str, tool: str, payload: dict[str, Any]) -> None:
    row = {"event": event, "tool": tool, **payload}
    _tools_log.info("%s", json.dumps(row, ensure_ascii=False, default=str))


def _search_outcome_meta(text: str) -> dict[str, Any]:
    if "RETRIEVAL_ERROR" in text[:80]:
        return {"result": "retrieval_error"}
    if "RETRIEVAL_STATUS:OK" in text:
        return {"result": "ok"}
    if "LOW_CONFIDENCE" in text or "NO_RELEVANT_CHUNKS" in text:
        return {"result": "low_confidence_or_empty"}
    return {"result": "other"}


def _query_lexically_overlaps_doc(query: str, doc_text: str) -> bool:
    """Require overlap for multi-token queries so hybrid search cannot 'match' unrelated text with high score."""
    q_words = [w for w in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(w) > 2]
    if len(q_words) < 2:
        return True
    blob = (doc_text or "").lower()
    return any(w in blob for w in q_words)


def _format_citation_line(doc, score: float, rank: int) -> str:
    md = doc.metadata or {}
    div = md.get("division", "")
    yr = md.get("year", "")
    src = md.get("source", "unknown")
    pid = md.get("parent_id", "")
    parts = [
        f"[{rank}] relevance={score:.4f}",
        f"source={src}",
    ]
    if div != "":
        parts.append(f"division={div}")
    if yr != "":
        parts.append(f"year={yr}")
    if pid:
        parts.append(f"parent_id={pid}")
    head = " | ".join(parts)
    return f"{head}\nContent: {doc.page_content.strip()}"


class ToolFactory:

    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()

    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks with relevance scores and citation metadata.

        Args:
            query: Search query string
            limit: Maximum number of results to return (capped)
        """
        try:
            limit = int(limit)
            limit = max(1, min(limit, 15))
            k_fetch = max(limit, getattr(config, "SEARCH_TOP_K_DEFAULT", 7))

            pairs = self.collection.similarity_search_with_relevance_scores(
                query, k=k_fetch, score_threshold=None
            )
            if not pairs:
                ts = 0.0
                return (
                    f"<<<RETRIEVAL_STATUS:LOW_CONFIDENCE|top_score={ts:.4f}>>>\n"
                    "NO_RELEVANT_CHUNKS: No documents were returned from the vector store for this query."
                )

            pairs.sort(key=lambda x: x[1], reverse=True)
            top_doc, top_score = pairs[0]
            top_score = float(top_score)

            if getattr(config, "RETRIEVAL_LEXICAL_OVERLAP_CHECK", True) and not _query_lexically_overlaps_doc(
                query, top_doc.page_content
            ):
                return (
                    f"<<<RETRIEVAL_STATUS:LOW_CONFIDENCE|top_score={top_score:.4f}>>>\n"
                    "NO_RELEVANT_CHUNKS: Top hit has no lexical overlap with the query "
                    "(possible off-topic question or wrong language). Do not invent data."
                )

            min_top = float(getattr(config, "RETRIEVAL_MIN_TOP_SCORE", 0.3))
            if top_score < min_top:
                return (
                    f"<<<RETRIEVAL_STATUS:LOW_CONFIDENCE|top_score={top_score:.4f}>>>\n"
                    f"NO_RELEVANT_CHUNKS: Best match similarity {top_score:.4f} is below the "
                    f"confidence threshold ({min_top:.2f}). Do not invent figures; ask a clarifying question or refuse."
                )

            thresh = float(getattr(config, "RETRIEVAL_SCORE_THRESHOLD", 0.22))
            filtered = [(d, s) for d, s in pairs if s >= thresh][:limit]
            if not filtered:
                filtered = pairs[:limit]

            header = f"<<<RETRIEVAL_STATUS:OK|top_score={top_score:.4f}>>>"
            blocks = [header, ""]
            for i, (doc, score) in enumerate(filtered, 1):
                blocks.append(_format_citation_line(doc, score, i))
            return "\n\n".join(blocks)

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunk by ID (full division time series when ingested from CPI CSV).

        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            data = self.parent_store_manager.load(parent_id)
            if not data:
                return "NO_PARENT_DOCUMENT"

            meta = data.get("metadata", {})
            body = (data.get("page_content") or "").strip()
            return (
                f"<<<PARENT_RETRIEVAL:OK>>>\n"
                f"Parent ID: {meta.get('parent_id', parent_id)}\n"
                f"File Name: {meta.get('source', 'unknown')}\n"
                f"Division: {meta.get('division', 'n/a')}\n"
                f"Content:\n{body}"
            )

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"

    # ── MCP Tool wrappers ─────────────────────────────────────────────────────

    def _export_cpi_data(self, start_year: int, end_year: int, division: str = "overall") -> str:
        """Malaysia annual CPI for a year range and division (JSON with **csv_content** for download).

        **Use this tool** when the user says export, download, CSV, spreadsheet, or wants the full table.
        **Do not** use search_child_chunks for the same export request. Education → division "10", transport → "07".

        Args:
            start_year: First year of the range (1960–2025).
            end_year:   Last year of the range (1960–2025, must be >= start_year).
            division:   "overall" (default) — headline CPI only. Use when no division is mentioned.
                        "all" — all divisions (overall + 01 to 13). Use ONLY when the user
                                explicitly says "all divisions", "by division", or "breakdown".
                        "01" to "13" — single COICOP category (food → "01", education → "10").

        Returns:
            JSON string with records, csv_content (4 d.p.), and source metadata.
        """
        return _mcp_export_cpi_data(
            start_year=int(start_year),
            end_year=int(end_year),
            division=str(division),
        )

    def _log_summary(self, query: str, answer: str, sources: str = "") -> str:
        """Append a Q&A interaction to the audit log for monitoring and evaluation.

        Call this after delivering an answer to record the interaction.

        Args:
            query:   The user's original question.
            answer:  The answer text (first 500 chars stored).
            sources: Comma-separated source filenames cited (optional).
        """
        return _mcp_log_summary(query=query, answer=answer, sources=sources)

    def create_tools(self) -> List:
        """Create and return the list of tools (RAG + MCP export).

        log_summary is intentionally excluded here — it is called from the API
        layer after the agent finishes so it never mid-conversation-loops the agent.

        Each tool is wrapped to emit stdlib ``logging`` lines (logger ``cpi_rag.tools``):
        ``invoke`` before the body runs, ``complete`` after (visible in ``docker logs`` when
        the API has configured root/basic logging).
        """

        def search_child_chunks(query: str, limit: int) -> str:
            _log_tool_line("invoke", "search_child_chunks", {"query": (query or "")[:500], "limit": limit})
            out = self._search_child_chunks(query, limit)
            meta = _search_outcome_meta(out)
            meta["output_chars"] = len(out or "")
            _log_tool_line("complete", "search_child_chunks", meta)
            return out

        def retrieve_parent_chunks(parent_id: str) -> str:
            _log_tool_line("invoke", "retrieve_parent_chunks", {"parent_id": (parent_id or "")[:200]})
            out = self._retrieve_parent_chunks(parent_id)
            ok = "<<<PARENT_RETRIEVAL:OK>>>" in (out or "")
            err = (out or "").startswith("PARENT_RETRIEVAL_ERROR")
            _log_tool_line(
                "complete",
                "retrieve_parent_chunks",
                {
                    "result": "ok" if ok else ("error" if err else "not_found_or_other"),
                    "output_chars": len(out or ""),
                },
            )
            return out

        def export_cpi_data(start_year: int, end_year: int, division: str = "overall") -> str:
            _log_tool_line(
                "invoke",
                "export_cpi_data",
                {"start_year": int(start_year), "end_year": int(end_year), "division": str(division)},
            )
            out = self._export_cpi_data(start_year, end_year, division)
            extra: dict[str, Any] = {"output_chars": len(out or "")}
            try:
                parsed = json.loads(out)
                if "error" in parsed:
                    extra["result"] = "error"
                    extra["error"] = str(parsed.get("error", ""))[:300]
                else:
                    extra["result"] = "ok"
                    extra["record_count"] = parsed.get("record_count")
                    extra["division"] = parsed.get("division")
            except Exception:
                extra["result"] = "parse_error"
            _log_tool_line("complete", "export_cpi_data", extra)
            return out

        search_child_chunks.__doc__ = self._search_child_chunks.__doc__
        retrieve_parent_chunks.__doc__ = self._retrieve_parent_chunks.__doc__
        export_cpi_data.__doc__ = self._export_cpi_data.__doc__

        search_tool = tool("search_child_chunks")(search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(retrieve_parent_chunks)
        export_tool = tool("export_cpi_data")(export_cpi_data)

        return [search_tool, retrieve_tool, export_tool]
