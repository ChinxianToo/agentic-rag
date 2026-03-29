# CPI Agentic RAG — Project README

LangGraph agent over **Malaysia annual CPI** (DOSM / OpenDOSM): hybrid retrieval in **Qdrant**, local **Ollama** embeddings, **OpenRouter** chat models, optional **MCP** export tools.

---

## Table of contents

1. [Quickstart (≤10 min)](#quickstart-10-min)  
2. [Tool choice & model provider](#tool-choice--model-provider)  
3. [Data card](#data-card)  
4. [RAG design](#rag-design-chunking-embeddings-k)  
5. [Eval methodology & results](#eval-methodology--results)  
6. [Limitations & future work](#limitations--future-work)  

---

## Quickstart (≤10 min)

**You need:** Docker (Compose v2), an **OpenRouter** account + key, and ~10 minutes for first-time image build and model pull.

### 1. API key (redacted in docs)

Create `project/.env`:

```bash
# Required for the chat LLM (OpenRouter). Do not commit real keys.
OPENROUTER_API_KEY=<your_openrouter_key>
```

### 2. CPI CSV

Place the annual CPI file (long format: `date`, `division`, `index`) at **`project/data/cpi_2d_annual.csv`**.

The Compose stack mounts `./project/data` read-only. The API entrypoint can auto-ingest if the Qdrant collection is empty.

### 3. Run the stack

From the **repository root** (where `docker-compose.yml` lives):

```bash
docker compose up --build
```

Wait for:

- Qdrant healthy  
- Ollama up; **`nomic-embed-text`** pulled (first run)  
- Optional: one-time CPI ingest into `document_child_chunks`  
- **rag-api** listening on **8000**

### 4. Call the API

```bash
# Streaming chat (SSE)
curl -sN -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What was overall CPI in 2020?"}'

# Health
curl -s http://localhost:8000/health
```

**Local dev (no Docker):** install `requirements.txt`, run **Ollama** + **Qdrant** (or set `QDRANT_URL`), then:

```bash
cd project && python app.py server
# or: uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Tool choice & model provider

### Chat model (reasoning & tool calling)

| Component | Default | Config |
|-----------|---------|--------|
| **Provider** | **OpenRouter** (`langchain_openrouter.ChatOpenRouter`) | `OPENROUTER_API_KEY` in `project/.env` |
| **Model id** | `openai/gpt-4o-mini` | `LLM_MODEL` in `project/config.py` or env override in Compose |
| **Temperature** | `0` | `LLM_TEMPERATURE` |

Any OpenRouter model string works (e.g. `google/gemini-2.5-flash-lite`). **API keys must never appear in README or commits** — use `.env` only.

### Embeddings (retrieval)

| Component | Default | Config |
|-----------|---------|--------|
| **Provider** | **Ollama** | `OLLAMA_BASE_URL` (Docker: `http://ollama:11434`) |
| **Dense model** | `nomic-embed-text` | `DENSE_MODEL` |
| **Sparse** | BM25 via FastEmbed | `SPARSE_MODEL` = `Qdrant/bm25`, field `SPARSE_VECTOR_NAME` = `sparse` |

Changing the embedding model requires **re-ingestion** (`ingest_cpi.py --reset`).

### Vector store

| Component | Default | Config |
|-----------|---------|--------|
| **Engine** | **Qdrant** | `QDRANT_URL` (e.g. `http://localhost:6333` or embedded path `QDRANT_DB_PATH` if unset) |
| **Collection** | `document_child_chunks` | `CHILD_COLLECTION` |

### Agent tools (LangGraph)

| Tool | Role |
|------|------|
| `search_child_chunks` | Hybrid dense + sparse search over **child** chunks (one CPI row → one chunk). |
| `retrieve_parent_chunks` | Load full **division time series** from `parent_store` by `parent_id`. |
| `export_cpi_data` | Read canonical CSV → JSON + `csv_content` (export/download flows); shared with MCP server. |

Structured **query analysis** (rewrite layer) uses Pydantic **`QueryAnalysis`** in `rag_agent/schemas.py`, not the tool args.

---

## Data card

| Field | Detail |
|--------|--------|
| **Source** | DOSM / OpenDOSM **annual** CPI (Malaysia), typically `cpi_annual` / `CPI 2D Annual` style extracts |
| **Granularity** | One index value per **(year, division)** — **not** monthly |
| **Year range** | **1960–2025** (as configured in guardrails / prompts; clip to your CSV) |
| **Divisions** | `overall` (headline) + COICOP-style groups **01–13** (food, transport, education, …) |
| **On-disk CSV** | Long format: `date`, `division`, `index` (see `cpi_document_builder.py`) |
| **License / attribution** | Follow OpenDOSM / DOSM terms (export tool JSON includes a **license** string when present) |
| **PII** | None expected — aggregate statistics only |

---

## RAG design (chunking, embeddings, k)

### Chunking (tabular CPI)

- **Child (embedded):** **one CSV row → one LangChain `Document`** — natural-language fact + metadata (`division`, `year`, `parent_id`, `source`, …). No sliding overlapping windows on raw CSV.  
- **Parent (file store):** **one document per `division`**, full annual series as `year: index` lines in `parent_store` JSON, id `cpi_2dannual_{division}_parent`.  

Implementation: `cpi_document_builder.py` + `ingest_cpi.py`.

### Retrieval mode

- **Hybrid:** `QdrantVectorStore` with `RetrievalMode.HYBRID` (`langchain_qdrant`): dense cosine + sparse BM25.

### Effective *k* and thresholds

| Setting | Role | Default (`config.py`) |
|---------|------|------------------------|
| Tool `limit` | Max chunks returned to the LLM (agent-chosen, capped in tool) | Up to **15** in code |
| `SEARCH_TOP_K_DEFAULT` | Fetch size before filtering | **7** |
| `RETRIEVAL_SCORE_THRESHOLD` | Min score to keep in the result list | **0.22** |
| `RETRIEVAL_MIN_TOP_SCORE` | Below best score → **low_confidence** path | **0.30** |
| `RETRIEVAL_LEXICAL_OVERLAP_CHECK` | Extra guard on hybrid false positives | **True** |

Tune these if you see empty hits or noisy matches for your index size and embedding model.

### Memory & compression

- Outer graph: conversation summary + query rewrite; checkpointed threads (`InMemorySaver` — lost on API restart unless changed).  
- Inner agent: token estimate → optional **compress_context** summarization; limits `MAX_TOOL_CALLS` / `MAX_ITERATIONS`.

---

## Eval methodology & results

**There is no fixed benchmark script in this repo.** Suggested lightweight methodology:

1. **Gold Q&A set:** Curate ~20–50 questions with answers computed from the CSV (exact division/year index).  
2. **Metrics:** Exact match or tolerance on numeric CPI; optional citation match (`division`, `year`, source file).  
3. **Runs:** Fix `LLM_MODEL`, embedding model, and ingestion hash; run each question via `POST /v1/chat` or stream.  
4. **Logging for post-hoc analysis:**  
   - `project/logs/interactions.jsonl` — `log_summary` (query + answer preview)  
   - `project/logs/agent_steps.jsonl` — optional LangGraph step trace (`AGENT_TRACE_LOG`)  
   - Stdlib **JSON logs** from `api.py` (query, `graph_route`, `retrieved_docs`, status) and **`cpi_rag.tools`** (tool invoke/complete)  

**Published results:** N/A in-repo — fill this section after you run your eval set.

---

## Limitations & future work

- **Annual only:** Questions implying monthly CPI cannot be answered from this index.  
- **National index:** No city- or regional-level CPI in the default dataset card.  
- **In-memory checkpoints:** Conversation history does not survive `rag-api` container recreation unless you swap in a persistent checkpointer.  
- **Export vs search:** Export intent is prompt- + orchestrator-gated; edge cases can still mix tools — tighter schemas or deterministic export routing could harden this.  
- **Eval:** Add automated regression tests + a small public CPI-QA gold file.  
- **Observability:** Optional OpenTelemetry; structured log shipping from `cpi_rag.tools` + API summaries.  
- **Multi-collection:** Currently centered on one CPI CSV collection; extending to other DOSM tables would need separate ingest pipelines and tool routing.

---

## Related paths

| Path | Purpose |
|------|---------|
| `docker-compose.yml` | Qdrant, Ollama, `rag-api`, optional `mcp-server` |
| `project/docker-entrypoint.sh` | Health checks, optional auto-ingest, `uvicorn project.api:app` |
| `project/api.py` | FastAPI: `/v1/chat`, `/v1/chat/stream`, CSV export routes |
| `project/mcp_server.py` | Standalone FastMCP CPI tools |
| `project/API_GATEWAY.md` | Extra API / gateway notes (if present) |

---

*Last rewritten to match the CPI RAG stack in this repository (LangGraph + Qdrant + Ollama + OpenRouter).*
