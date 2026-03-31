import glob
import os


def _env_stripped(*keys: str) -> str | None:
    """First env value that is non-empty after stripping whitespace, else None."""
    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        val = raw.strip()
        if val:
            return val
    return None


# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(__file__)

PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
# DOSM CPI (tabular) — default CSV for `ingest_cpi.py`
CPI_CSV_PATH = os.path.join(PARENT_STORE_PATH, "CPI 2D Annual.csv")
# Alternative location (tracked in git, outside gitignored parent_store/)
CPI_DATA_DIR = os.path.join(_BASE_DIR, "data")
CPI_CSV_PATH_DATA = os.path.join(CPI_DATA_DIR, "cpi_2d_annual.csv")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")


def resolve_cpi_csv_path() -> str | None:
    """First path that exists on disk, or None.
    `parent_store/` is often gitignored — use `data/` or pass --csv explicitly."""
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


# --- Qdrant Configuration ---
# Override with env var for Docker: QDRANT_URL=http://qdrant:6333
# Set to empty string / None to use embedded (local file) mode
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333") or None
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Embedding Model ---
# Dense: Ollama model (run: ollama pull nomic-embed-text)
# Override base URL for Docker: OLLAMA_BASE_URL=http://ollama:11434
DENSE_MODEL = "nomic-embed-text"
SPARSE_MODEL = "Qdrant/bm25"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- LLM (via OpenRouter) ---
# Keys: https://openrouter.ai/settings/keys — set OPENROUTER_API_KEY in project/.env
# (optional alias OPENROUTER_KEY). Trim avoids 401 / "User not found" from stray spaces.
# Optional: OPENROUTER_APP_URL, OPENROUTER_APP_TITLE (attribution), OPENROUTER_API_BASE.
OPENROUTER_API_KEY = _env_stripped("OPENROUTER_API_KEY", "OPENROUTER_KEY")
OPENROUTER_APP_URL = _env_stripped("OPENROUTER_APP_URL")
OPENROUTER_APP_TITLE = _env_stripped("OPENROUTER_APP_TITLE")
OPENROUTER_API_BASE = _env_stripped("OPENROUTER_API_BASE", "OPENROUTER_BASE_URL")
# Format: "provider/model-name"  e.g. openai/gpt-4o-mini, anthropic/claude-3.5-sonnet
LLM_MODEL = "openai/gpt-4o-mini"
LLM_TEMPERATURE = 0

# --- Agent loop limits ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Retrieval confidence thresholds ---
# Scores are in [0, 1] from Qdrant hybrid search; tune for your index.
RETRIEVAL_SCORE_THRESHOLD = 0.22   # min per-chunk score included in the returned list
RETRIEVAL_MIN_TOP_SCORE = 0.30     # below this → low_confidence_response (clarify / refuse)
SEARCH_TOP_K_DEFAULT = 7
# Require ≥1 query token (>2 chars) to appear in the top hit (guards hybrid false positives)
RETRIEVAL_LEXICAL_OVERLAP_CHECK = True
