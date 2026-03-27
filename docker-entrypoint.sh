#!/bin/bash
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
EMBED_MODEL="${DENSE_MODEL:-nomic-embed-text}"
COLLECTION="${CHILD_COLLECTION:-document_child_chunks}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"

banner() { echo ""; echo "══════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════"; }
step()   { echo ""; echo "▶ $1"; }
ok()     { echo "  ✓ $1"; }

banner "CPI RAG API — startup"

# ── 1. Wait for Qdrant ────────────────────────────────────────────────────────
step "Waiting for Qdrant at ${QDRANT_URL} ..."
until curl -sf "${QDRANT_URL}/healthz" > /dev/null 2>&1; do
    echo "  … not ready yet, retrying in 3 s"
    sleep 3
done
ok "Qdrant ready"

# ── 2. Wait for Ollama ────────────────────────────────────────────────────────
step "Waiting for Ollama at ${OLLAMA_URL} ..."
until curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; do
    echo "  … not ready yet, retrying in 3 s"
    sleep 3
done
ok "Ollama ready"

# ── 3. Ensure embedding model is available ────────────────────────────────────
step "Checking embedding model '${EMBED_MODEL}' ..."
HAS_MODEL=$(curl -sf "${OLLAMA_URL}/api/tags" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(any('${EMBED_MODEL}' in m.get('name','') for m in d.get('models',[])))" \
    2>/dev/null || echo "False")

if [ "${HAS_MODEL}" != "True" ]; then
    echo "  Pulling '${EMBED_MODEL}' — this may take a few minutes on first run ..."
    curl -sf -X POST "${OLLAMA_URL}/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${EMBED_MODEL}\", \"stream\": false}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print('  status:', d.get('status','done'))"
fi
ok "'${EMBED_MODEL}' ready"

# ── 4. Auto-ingest CPI data if collection is empty ────────────────────────────
step "Checking CPI index (collection: ${COLLECTION}) ..."
VECTOR_COUNT=$(curl -sf "${QDRANT_URL}/collections/${COLLECTION}" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',{}).get('vectors_count',0))" \
    2>/dev/null || echo "0")

if [ "${VECTOR_COUNT}" = "0" ]; then
    echo "  Collection empty — ingesting CPI data ..."
    # Check that the CSV file exists first
    CSV_CHECK=$(python3 -c "
import sys
sys.path.insert(0, '/app/project')
import config
p = config.resolve_cpi_csv_path()
print(p if p else 'NOT_FOUND')
" 2>/dev/null || echo "NOT_FOUND")

    if [ "${CSV_CHECK}" = "NOT_FOUND" ]; then
        echo ""
        echo "  ╔══════════════════════════════════════════════════════════╗"
        echo "  ║  CPI CSV not found!                                      ║"
        echo "  ║  Place the file at one of:                               ║"
        echo "  ║    • ./project/data/cpi_2d_annual.csv   (recommended)    ║"
        echo "  ║    • ./project/parent_store/CPI 2D Annual.csv            ║"
        echo "  ║  Download from: open.dosm.gov.my (cpi_annual)            ║"
        echo "  ╚══════════════════════════════════════════════════════════╝"
        echo ""
        echo "  Starting API anyway — queries will fail until data is loaded."
        echo "  Re-run ingestion later: docker compose exec rag-api python project/ingest_cpi.py --reset"
    else
        echo "  Found CSV at: ${CSV_CHECK}"
        cd /app && python3 project/ingest_cpi.py --reset
        ok "CPI data indexed"
    fi
else
    ok "Index already loaded (${VECTOR_COUNT} vectors)"
fi

# ── 5. Start API server ───────────────────────────────────────────────────────
banner "Starting API on ${API_HOST}:${API_PORT}"
echo "  Docs:    http://localhost:${API_PORT}/docs"
echo "  Stream:  POST http://localhost:${API_PORT}/v1/chat/stream"
echo "  Sync:    POST http://localhost:${API_PORT}/v1/chat"
echo ""

cd /app
exec uvicorn project.api:app \
    --host "${API_HOST}" \
    --port "${API_PORT}" \
    --workers 1
