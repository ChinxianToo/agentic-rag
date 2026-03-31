# Chunking strategy and embedding configuration

Technical summary for reports and methodology sections. Implementation lives in `cpi_document_builder.py`, `db/vector_db_manager.py`, and `config.py`.

---

## Source data

- **Input**: Long-format CSV with columns `date`, `division`, `index` (one row per calendar year per division).
- **Divisions**: `overall` (headline / all groups) plus COICOP-style codes `01`–`13`.
- **Ingestion**: Rows are grouped by `division`, sorted by date; no recursive character splitting on raw CSV so numeric rows are never split across chunks.

---

## Two-tier chunking (child vs parent)

| Aspect | **Child chunks** | **Parent chunks** |
|--------|------------------|-------------------|
| **Granularity** | **One CSV row → one chunk** (one year × one division). | **One division → one document** (full annual series for that division in the extract). |
| **Text form** | Short natural-language sentence stating division label, year, index value, and source filename. | Markdown-style heading plus `year: index` lines for every year in that division. |
| **Purpose** | Precision retrieval for factual Q&A (“CPI for division X in year Y”). | Full time-series context after the model knows `parent_id` (trends, multi-year answers). |
| **Storage** | Embedded and indexed in **Qdrant** (`document_child_chunks`). | Saved as JSON in **`parent_store/`** (`{parent_id}.json`); **not** vector-indexed. |
| **Stable ID** | Metadata includes `parent_id`, `division`, `year`, `index_value`, `source`, `kind: cpi_row`. | `parent_id` pattern: `cpi_2dannual_{division}_parent`; metadata includes `division`, `source`, `kind: cpi_division_series`. |

**Design rationale**: Tabular CPI is not prose; treating each observation as its own chunk maximizes hybrid (semantic + lexical) retrieval for year/division queries. Parents avoid re-embedding long series while still giving the agent authoritative full-series text when needed.

---

## Embedding and retrieval models

### Dense vectors

- **Model**: `nomic-embed-text` via **Ollama** (`langchain_ollama.OllamaEmbeddings`).
- **Service**: `OLLAMA_BASE_URL` (default `http://localhost:11434`; Docker override typical `http://ollama:11434`).
- **Vector config**: **Cosine** distance in Qdrant; dimension is the model output size (probed at collection creation with a short test embed).

### Sparse vectors

- **Model**: `Qdrant/bm25` via **FastEmbed** (`langchain_qdrant.FastEmbedSparse`), i.e. BM25-style sparse encoding **locally** (not a hosted API).
- **Qdrant field name**: Sparse vector name `sparse` (`SPARSE_VECTOR_NAME`).

### Hybrid search

- **Mode**: `RetrievalMode.HYBRID` on `QdrantVectorStore`.
- **Behaviour**: Each query gets a dense embedding and a sparse vector; Qdrant runs **prefetch** on both branches and fuses rankings with **RRF** (reciprocal rank fusion), producing a single ranked list and per-hit scores used downstream.

---

## Operational notes (report checklist)

- **Reproducibility**: Record Ollama image/tag, `nomic-embed-text` digest if pinned, `fastembed` / `langchain-qdrant` versions, and Qdrant mode (URL vs embedded path).
- **LLM vs embeddings**: Chat generation uses **OpenRouter** (`ChatOpenRouter`) with `LLM_MODEL` from config; that is **separate** from the embedding stack above—cite both if the report covers “models used.”
- **Export path**: Tabular answers can also be served via **`export_cpi_data`** reading the same CSV; that path does not change chunking or embeddings.

---

## Quick reference (`config.py`)

| Setting | Typical value |
|---------|----------------|
| `CHILD_COLLECTION` | `document_child_chunks` |
| `DENSE_MODEL` | `nomic-embed-text` |
| `SPARSE_MODEL` | `Qdrant/bm25` |
| `SPARSE_VECTOR_NAME` | `sparse` |

For dataset provenance and licence, see `DataCard.md`.
