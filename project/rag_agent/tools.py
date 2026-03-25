import re
from typing import List

import config
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager


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

    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """Retrieve full parent chunks by their IDs.

        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"

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

    def create_tools(self) -> List:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)

        return [search_tool, retrieve_tool]
