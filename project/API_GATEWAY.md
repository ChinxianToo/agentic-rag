# CPI RAG API Gateway — Integration Reference

> **Target audience:** AI coding tools, service integrations, and developers connecting external services to this gateway.
>
> This document is the single source of truth for **request/response contracts**, **SSE event schemas**, **error handling**, and **code examples**.

---

## Base URL

```
http://<host>:8000
```

Default when running locally: `http://localhost:8000`

Interactive OpenAPI docs (auto-generated): `http://localhost:8000/docs`

---

## Scope

This gateway answers questions **exclusively** about Malaysia's **Consumer Price Index (CPI)** from the DOSM [OpenDOSM](https://open.dosm.gov.my/data-catalogue/cpi_annual) annual dataset.

- Covered years: **1960 – 2025**
- Divisions: `overall` + COICOP 2-digit groups `01`–`13`
- Language: English queries (and English CPI terminology)

Queries outside this scope receive an **instant refusal** — no LLM call is made.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/stream` | **Streaming** — Server-Sent Events (SSE), token-by-token |
| `POST` | `/v1/chat` | **Sync** — blocks until full answer, returns JSON |
| `DELETE` | `/v1/sessions/{thread_id}` | Reset conversation history for a session |
| `GET` | `/health` | Liveness check |

---

## Request Schema (both POST endpoints)

```json
{
  "message": "string",        // required — the user's question
  "thread_id": "string|null"  // optional — omit to start a new session;
                               // send the same value on follow-up turns
                               // to maintain conversation history
}
```

**Field rules**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `message` | `string` | ✅ | Plain-text question. Do not pre-prompt. |
| `thread_id` | `string` (UUID) | ❌ | If omitted, the server generates one and returns it. Reuse across turns for multi-turn chat. |

---

## `POST /v1/chat/stream` — SSE Streaming

### Headers (request)

```
Content-Type: application/json
```

### Headers (response)

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

### SSE wire format

Every line on the stream follows standard SSE:

```
data: <json payload>\n\n
```

The stream **always** ends with the bare terminator line:

```
data: [DONE]\n\n
```

### Event sequence

Events are emitted **in this order**, on every request:

```
start  →  guardrail  →  [status …]  →  [retrieval]  →  token …  →  done  →  [DONE]
```

#### 1. `start`
Emitted immediately, before any processing.

```json
{ "event": "start", "thread_id": "550e8400-e29b-41d4-a716-446655440000" }
```

#### 2. `guardrail`
Result of the pre-flight domain check (zero LLM cost).

```json
{ "event": "guardrail", "status": "passed" }
```

| `status` value | Meaning |
|---|---|
| `passed` | Query is in scope — RAG pipeline will continue. |
| `blocked_offtopic` | Query has no CPI/DOSM keywords — refused instantly. |
| `blocked_year` | Query mentions a year outside 1960–2025 — refused instantly. |

When status is `blocked_*`, the next events are `token` (the refusal message) → `done` → `[DONE]`. No further processing occurs.

#### 3. `status` *(optional, multiple)*
Human-readable progress hints emitted as each pipeline layer starts.

```json
{ "event": "status", "message": "Searching knowledge base …" }
{ "event": "status", "message": "Generating answer …" }
```

Safe to display as a loading indicator. May appear 1–3 times.

#### 4. `retrieval`
Emitted after the Qdrant search result is evaluated by the post-guardrail.

```json
{ "event": "retrieval", "status": "ok" }
```

| `status` value | Meaning |
|---|---|
| `ok` | Retrieved chunks meet the confidence threshold — answer generation proceeds. |
| `low_confidence` | Best similarity score too low — system will ask a clarifying question or refuse. |

#### 5. `token` *(multiple)*
Individual LLM output tokens. Concatenate all `content` values to build the full answer.

```json
{ "event": "token", "content": "The " }
{ "event": "token", "content": "overall " }
{ "event": "token", "content": "CPI for Malaysia in 2020 was **120.10**." }
```

> **Note:** Token granularity depends on the LLM provider. Some models emit word-by-word, others emit larger chunks. Always concatenate, never assume a single token = full answer.

#### 6. `done`
Stream is complete. No further events follow (except the wire terminator).

```json
{ "event": "done", "thread_id": "550e8400-e29b-41d4-a716-446655440000" }
```

Save the `thread_id` if you want to continue this conversation.

#### 7. `error` *(conditional)*
Emitted only if an unexpected server-side exception occurs. Treat as fatal for the current request.

```json
{ "event": "error", "message": "Qdrant connection refused" }
```

---

## `POST /v1/chat` — Synchronous JSON

Returns the full answer in one response. Use this when your service does **not** support SSE.

### Response body (`200 OK`)

```json
{
  "reply": "The overall CPI for Malaysia in 2020 was 120.10.\n\n---\n**Sources:**\n- CPI 2D Annual.csv — division: overall, year(s): 2020",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "guardrail_status": "passed"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `reply` | `string` | Full Markdown answer with inline citations and Sources section. |
| `thread_id` | `string` | Reuse on next request to continue the conversation. |
| `guardrail_status` | `string` | `passed` / `blocked_offtopic` / `blocked_year` |

### Error responses

| HTTP code | When |
|-----------|------|
| `500` | Unhandled exception in the RAG pipeline. `detail` field has the error message. |
| `503` | Server starting up — RAG system not yet initialised. Retry after 5 s. |

---

## `DELETE /v1/sessions/{thread_id}`

Clears LangGraph conversation history for the given session.

```
DELETE /v1/sessions/550e8400-e29b-41d4-a716-446655440000
→ 204 No Content
```

Call this when a user explicitly starts a new conversation (e.g. "New chat" button).

---

## `GET /health`

```json
{ "status": "ok", "collection": "document_child_chunks" }
```

Returns `503` if the RAG system has not finished initialising.

---

## Citation format in answers

All answers include structured citations at the end. Parse the `reply` field for the `Sources` section:

```markdown
The overall CPI in Malaysia for 2020 was **120.10** ([CPI 2D Annual.csv], division: overall, year: 2020 [1]).

---
**Sources:**
- CPI 2D Annual.csv — division: overall (All groups), year(s): 2020
```

The citation format is: `filename — division: <code> (<label>), year(s): <years>`

---

## Multi-turn conversation

Maintain conversation context by reusing `thread_id` across requests:

```
Turn 1: POST /v1/chat  { "message": "What was CPI in 2020?" }
        ← { "reply": "...", "thread_id": "abc-123" }

Turn 2: POST /v1/chat  { "message": "And for 2021?", "thread_id": "abc-123" }
        ← { "reply": "The CPI in 2021 was..." }   // system knows context
```

To start fresh: call `DELETE /v1/sessions/abc-123` or simply omit `thread_id`.

---

## Code examples

### Python — SSE streaming

```python
import httpx
import json

BASE = "http://localhost:8000"

def stream_chat(message: str, thread_id: str | None = None):
    payload = {"message": message}
    if thread_id:
        payload["thread_id"] = thread_id

    full_answer = ""
    current_thread_id = thread_id

    with httpx.Client(timeout=120) as client:
        with client.stream("POST", f"{BASE}/v1/chat/stream", json=payload) as r:
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line.removeprefix("data:").strip()
                if raw == "[DONE]":
                    break
                event = json.loads(raw)

                match event["event"]:
                    case "start":
                        current_thread_id = event["thread_id"]
                    case "guardrail":
                        if event["status"] != "passed":
                            print(f"[BLOCKED: {event['status']}]")
                    case "status":
                        print(f"  … {event['message']}", flush=True)
                    case "token":
                        full_answer += event["content"]
                        print(event["content"], end="", flush=True)
                    case "done":
                        print()  # newline after streamed tokens

    return full_answer, current_thread_id


answer, tid = stream_chat("What was the overall CPI in 2020?")
answer2, _  = stream_chat("And for food & beverages?", thread_id=tid)
```

### Python — Synchronous (no SSE)

```python
import httpx

BASE = "http://localhost:8000"

def chat(message: str, thread_id: str | None = None) -> dict:
    payload = {"message": message}
    if thread_id:
        payload["thread_id"] = thread_id
    r = httpx.post(f"{BASE}/v1/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()   # {"reply": "...", "thread_id": "...", "guardrail_status": "..."}

result = chat("Compare food CPI in 2015 vs 2022")
print(result["reply"])
```

### JavaScript / TypeScript — SSE streaming

```typescript
async function streamChat(message: string, threadId?: string): Promise<string> {
  const body = JSON.stringify({ message, thread_id: threadId ?? null });
  const response = await fetch("http://localhost:8000/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let fullAnswer = "";
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop()!;          // keep incomplete chunk

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const raw = line.slice(5).trim();
      if (raw === "[DONE]") return fullAnswer;

      const event = JSON.parse(raw);
      if (event.event === "token") {
        fullAnswer += event.content;
        process.stdout.write(event.content);  // or update your UI
      }
    }
  }
  return fullAnswer;
}
```

### cURL — quick test

```bash
# Streaming
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What was CPI for transport in 2019?"}'

# Sync
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Compare overall CPI 2015 vs 2020"}' | jq .

# Health
curl http://localhost:8000/health

# Clear session
curl -X DELETE http://localhost:8000/v1/sessions/<thread_id>
```

---

## Guardrail behaviour reference

| Query example | `guardrail.status` | Behaviour |
|---|---|---|
| `"What was overall CPI in 2020?"` | `passed` | Full RAG pipeline, answer with citations |
| `"Compare food CPI 2015–2022"` | `passed` | Full RAG pipeline, trend answer |
| `"What is the weather today?"` | `blocked_offtopic` | Instant refusal, no LLM cost |
| `"What will CPI be in 2030?"` | `blocked_year` | Instant refusal, dataset range message |
| `"CPI in 1850?"` | `blocked_year` | Instant refusal, dataset range message |
| `"How are you?"` | `blocked_offtopic` | Instant refusal |
| `"CPI for quantum computing"` | `passed` → `low_confidence` | Passes scope check, but retrieval returns low similarity; LLM asks clarifying question |

---

## Known limitations

- **No forecasting** — only historical data (1960–2025). Future year queries are refused.
- **National-level only** — this index covers Malaysia as a whole, not by state.
- **Annual granularity** — monthly CPI data is not in this extract.
- **English only** — domain keywords and query understanding are English-based.
- **In-memory sessions** — conversation history is lost on server restart (no persistent store).
