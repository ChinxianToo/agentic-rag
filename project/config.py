import glob
import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(__file__)

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
# DOSM CPI (tabular) — default CSV for `ingest_cpi.py`
CPI_CSV_FILENAME = "CPI 2D Annual.csv"
CPI_CSV_PATH = os.path.join(PARENT_STORE_PATH, CPI_CSV_FILENAME)
# Optional: keep a copy outside gitignored `parent_store/` (e.g. commit a small extract)
CPI_DATA_DIR = os.path.join(_BASE_DIR, "data")
CPI_CSV_PATH_DATA = os.path.join(CPI_DATA_DIR, "cpi_2d_annual.csv")


def resolve_cpi_csv_path() -> str | None:
    """First path that exists on disk, or None. `parent_store/` is often gitignored — use `data/` or `--csv`."""
    candidates = [
        CPI_CSV_PATH,
        os.path.join(PARENT_STORE_PATH, "cpi_2d_annual.csv"),
        CPI_CSV_PATH_DATA,
        os.path.join(CPI_DATA_DIR, "CPI 2D Annual.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    store_csvs = sorted(glob.glob(os.path.join(PARENT_STORE_PATH, "*.csv")))
    if store_csvs:
        cpi_matches = [p for p in store_csvs if "cpi" in os.path.basename(p).lower()]
        return (cpi_matches or store_csvs)[0]
    return None
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")

# --- Qdrant Configuration ---
# Set to Docker Qdrant URL, or None for embedded (local file) mode
QDRANT_URL = "http://localhost:6333"  # Docker: docker run -d -p 6333:6333 -p 6334:6334 --name agentic-rag-db qdrant/qdrant
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
# Dense embeddings: local Ollama model (run: ollama pull nomic-embed-text)
DENSE_MODEL = "nomic-embed-text"
SPARSE_MODEL = "Qdrant/bm25"

# --- OpenRouter LLM Configuration ---
# Get API key from https://openrouter.ai/settings/keys and set OPENROUTER_API_KEY env var
# Model format: "provider/model-name" (e.g. openai/gpt-4o-mini, anthropic/claude-3.5-sonnet)
LLM_MODEL = "google/gemini-2.5-flash-lite"
LLM_TEMPERATURE = 0

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- RAG retrieval (Phase 3) ---
# similarity in [0,1] from Qdrant/LangChain; hybrid fusion can score unrelated queries high — tune for your index.
RETRIEVAL_SCORE_THRESHOLD = 0.22  # drop chunks below this when listing (keep top-k above)
# If the best hit is below this, route to low_confidence_response (clarify / refuse). Raise to ~0.8–0.9 for stricter demos.
RETRIEVAL_MIN_TOP_SCORE = 0.30
SEARCH_TOP_K_DEFAULT = 7
# If True, require at least one token (>2 chars) from multi-token queries to appear in the top hit (guards hybrid false positives).
RETRIEVAL_LEXICAL_OVERLAP_CHECK = True

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]
