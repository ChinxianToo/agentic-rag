#!/usr/bin/env python3
"""
Run each line of queries.jsonl against the CPI RAG API (sync chat) and write JSONL
with model responses.

Requires a running API (e.g. docker compose up) and OPENROUTER_API_KEY in the server env.

Usage (from repo root):
  python eval/run_eval.py
  python eval/run_eval.py --base-url http://localhost:8000 --output eval/results.jsonl
  python eval/run_eval.py --queries eval/queries.jsonl --sleep 2

After a successful run, prints latency p50/p95, retrieval hit-rate, and a heuristic
hallucination rate; writes the same metrics to ``<output_stem>_summary.json`` next to
the results file.

NOTE: Hallucination scoring is a lightweight **numeric + guardrail rubric** against
``expected_note`` (not human or LLM judge); see ``_hallucination_eval`` docstring.

NOTE: --output overwrites the file. Reference hints live in ``queries.expected_note``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

DEFAULT_TIMEOUT = 300.0

_NUMERIC_SCORE_CATEGORIES = frozenset(
    {
        "factual_overall",
        "factual_division",
        "compare_same_division",
        "compare_overall",
        "multi_aspect",
        "export",
    }
)
_GUARDRAIL_EXPECTED = frozenset({"guardrail_offtopic", "guardrail_year_range"})

# Decimal CPI indices in expected_note (avoids matching years like 2020).
_REF_FLOAT_RE = re.compile(r"\b\d+\.\d+\b")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile_ms(values: list[float], q: float) -> float:
    """q in [0, 100]; linear interpolation between ranks."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def _extract_ref_floats(expected_note: str) -> list[float]:
    if not expected_note:
        return []
    return [float(m.group(0)) for m in _REF_FLOAT_RE.finditer(expected_note)]


def _response_numeric_candidates(response: str) -> list[float]:
    """Floats from the model reply (CPI-style decimals)."""
    if not response:
        return []
    return [float(m.group(0)) for m in _REF_FLOAT_RE.finditer(response)]


def _ref_covered(ref: float, response: str, resp_floats: list[float]) -> bool:
    tol = max(1.0, 0.04 * ref)
    if any(abs(p - ref) <= tol for p in resp_floats):
        return True
    for dec in (1, 2):
        s = str(round(ref, dec)).rstrip("0").rstrip(".")
        if len(s) >= 1 and s in response:
            return True
    return False


def _hallucination_eval(
    *,
    category: str,
    guardrail_status: str,
    expected_note: str,
    response: str,
    error: str | None,
) -> dict[str, object]:
    """Return evaluation dict for one row.

    - ``numeric``: categories with CPI decimals in ``expected_note`` — failure if any ref not covered.
    - ``guardrail``: off-topic / year queries — failure if guardrail did not trigger as expected.
    """
    out: dict[str, object] = {
        "numeric_scored": False,
        "numeric_hallucination": None,
        "guardrail_scored": False,
        "guardrail_failed": None,
    }
    if error:
        return out

    if category in _GUARDRAIL_EXPECTED:
        out["guardrail_scored"] = True
        if category == "guardrail_offtopic":
            out["guardrail_failed"] = guardrail_status != "blocked_offtopic"
        elif category == "guardrail_year_range":
            out["guardrail_failed"] = guardrail_status != "blocked_year"
        return out

    if category in _NUMERIC_SCORE_CATEGORIES:
        refs = _extract_ref_floats(expected_note)
        if not refs:
            return out
        out["numeric_scored"] = True
        resp_floats = _response_numeric_candidates(response)
        uncovered = [r for r in refs if not _ref_covered(r, response, resp_floats)]
        out["numeric_hallucination"] = len(uncovered) > 0
        out["reference_floats"] = refs
        out["uncovered_floats"] = uncovered

    return out


def _build_summary(rows: list[dict]) -> dict[str, object]:
    ok = [r for r in rows if not r.get("error")]
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]

    elig = [r for r in ok if r.get("guardrail_status") == "passed"]
    retrieval_hits = sum(1 for r in elig if int(r.get("retrieved_docs") or 0) > 0)
    retrieval_eligible = len(elig)

    hall_num_scored = 0
    hall_num_flag = 0
    grd_scored = 0
    grd_flag = 0

    for r in rows:
        ev = r.get("eval") or {}
        if ev.get("numeric_scored"):
            hall_num_scored += 1
            if ev.get("numeric_hallucination"):
                hall_num_flag += 1
        if ev.get("guardrail_scored"):
            grd_scored += 1
            if ev.get("guardrail_failed"):
                grd_flag += 1

    hall_parts = hall_num_scored + grd_scored
    hall_flag_total = hall_num_flag + grd_flag

    return {
        "generated_at": _iso_now(),
        "n_queries": len(rows),
        "n_request_errors": sum(1 for r in rows if r.get("error")),
        "latency_ms": {
            "p50": round(_percentile_ms(latencies, 50), 2) if latencies else None,
            "p95": round(_percentile_ms(latencies, 95), 2) if latencies else None,
            "n": len(latencies),
            "definition": "Round-trip /v1/chat from client (includes HTTP errors that returned a body)",
        },
        "retrieval": {
            "eligible_turns": retrieval_eligible,
            "hits": retrieval_hits,
            "hit_rate": round(retrieval_hits / retrieval_eligible, 4) if retrieval_eligible else None,
            "definition": "share of successful responses with guardrail_status==passed and retrieved_docs>0",
        },
        "hallucination": {
            "method": "numeric_rubric_on_expected_note_plus_guardrail_compliance",
            "numeric_scored": hall_num_scored,
            "numeric_flagged": hall_num_flag,
            "guardrail_scored": grd_scored,
            "guardrail_flagged": grd_flag,
            "denominator": hall_parts,
            "flagged_total": hall_flag_total,
            "rate": round(hall_flag_total / hall_parts, 4) if hall_parts else None,
            "skipped_categories_note": "structure_impossible and rows without decimal refs in expected_note are not scored numerically",
        },
    }


def run_one(client: httpx.Client, base_url: str, message: str, thread_id: str | None) -> dict:
    url = f"{base_url.rstrip('/')}/v1/chat"
    payload: dict = {"message": message}
    if thread_id:
        payload["thread_id"] = thread_id
    t0 = time.perf_counter()
    resp = client.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    if not resp.is_success:
        try:
            err_body = resp.json()
            detail = err_body.get("detail", err_body)
        except Exception:
            detail = (resp.text or "")[:2000]
        exc = RuntimeError(f"HTTP {resp.status_code}: {detail}")
        setattr(exc, "latency_ms", latency_ms)
        raise exc
    data = resp.json()
    return {
        "reply": data.get("reply", ""),
        "thread_id": data.get("thread_id", ""),
        "guardrail_status": data.get("guardrail_status", ""),
        "retrieved_docs": int(data.get("retrieved_docs", 0) or 0),
        "latency_ms": latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval queries against /v1/chat and save JSONL.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="RAG API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(__file__).resolve().parent / "queries.jsonl",
        help="Input queries.jsonl path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results.jsonl",
        help="Output JSONL (overwritten)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Metrics JSON path (default: <output_stem>_summary.json)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to wait between requests (rate limiting)",
    )
    parser.add_argument(
        "--shared-thread",
        action="store_true",
        help="Reuse one thread_id for all queries (default: new session per query)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payloads only; do not call API",
    )
    parser.add_argument(
        "--health-retries",
        type=int,
        default=5,
        help="Retry /health this many times (docker may need a few seconds after start)",
    )
    parser.add_argument(
        "--health-wait",
        type=float,
        default=2.0,
        help="Seconds to sleep between /health retries",
    )
    args = parser.parse_args()

    summary_path = args.summary
    if summary_path is None:
        summary_path = args.output.parent / f"{args.output.stem}_summary.json"

    if not args.queries.is_file():
        print(f"Queries file not found: {args.queries}", file=sys.stderr)
        return 1

    lines = [ln.strip() for ln in args.queries.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows_out: list[dict] = []

    shared_tid: str | None = str(uuid.uuid4()) if args.shared_thread else None

    if args.dry_run:
        for line in lines:
            print(line)
        return 0

    with httpx.Client() as client:
        health_url = f"{args.base_url.rstrip('/')}/health"
        last_err: Exception | None = None
        for attempt in range(max(1, args.health_retries)):
            try:
                h = client.get(health_url, timeout=30.0)
                h.raise_for_status()
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                if attempt + 1 < args.health_retries:
                    time.sleep(args.health_wait)

        if last_err is not None:
            print(
                f"API health check failed after {args.health_retries} attempt(s) ({health_url}):\n"
                f"  {last_err}\n"
                "\n"
                "Common fixes:\n"
                "  • Start the API:  docker compose up rag-api   (from repo root)\n"
                "    or:             cd project && python app.py server\n"
                "  • Confirm port:   curl -sS http://localhost:8000/health\n"
                "  • If using Docker: wait until logs show 'Application startup complete'\n"
                "  • Connection reset often means nothing is listening or the container crashed.\n",
                file=sys.stderr,
            )
            return 1

        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skip invalid JSON line: {line[:80]}… ({exc})", file=sys.stderr)
                continue

            qid = record.get("id", "")
            category = record.get("category", "")
            query = record.get("query", "")
            expected_note = record.get("expected_note", "") or ""
            if not query:
                rows_out.append(
                    {
                        "id": qid,
                        "category": category,
                        "query": query,
                        "expected_note": expected_note,
                        "response": "",
                        "thread_id": "",
                        "guardrail_status": "",
                        "retrieved_docs": 0,
                        "latency_ms": None,
                        "error": "empty query",
                        "eval": {},
                        "completed_at": _iso_now(),
                    }
                )
                continue

            tid = shared_tid if args.shared_thread else None
            err: str | None = None
            reply = ""
            out_tid = ""
            gstatus = ""
            retrieved_docs = 0
            latency_ms: float | None = None

            try:
                out = run_one(client, args.base_url, query, tid)
                reply = out["reply"]
                out_tid = out["thread_id"]
                gstatus = out["guardrail_status"]
                retrieved_docs = out["retrieved_docs"]
                latency_ms = out["latency_ms"]
                if args.shared_thread and shared_tid is None:
                    shared_tid = out_tid
            except Exception as exc:
                err = str(exc)
                latency_ms = getattr(exc, "latency_ms", None)

            eval_row = _hallucination_eval(
                category=category,
                guardrail_status=gstatus,
                expected_note=expected_note,
                response=reply,
                error=err,
            )

            rows_out.append(
                {
                    "id": qid,
                    "category": category,
                    "query": query,
                    "expected_note": expected_note,
                    "response": reply,
                    "thread_id": out_tid,
                    "guardrail_status": gstatus,
                    "retrieved_docs": retrieved_docs,
                    "latency_ms": latency_ms,
                    "error": err,
                    "eval": eval_row,
                    "completed_at": _iso_now(),
                }
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in rows_out if not r.get("error"))
    print(f"Wrote {len(rows_out)} rows to {args.output} ({n_ok} without request error).")

    summary = _build_summary(rows_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lat = summary["latency_ms"]
    ret = summary["retrieval"]
    hall = summary["hallucination"]
    print("\n── Eval metrics ──")
    print(f"Latency (ms)      p50={lat['p50']}  p95={lat['p95']}  (n={lat['n']} timed requests)")
    if ret["hit_rate"] is None:
        print("Retrieval hit-rate  n/a (no passed-guardrail successes)")
    else:
        print(
            f"Retrieval hit-rate  {ret['hit_rate']:.2%}  "
            f"({ret['hits']}/{ret['eligible_turns']} turns with retrieved_docs>0)"
        )
    if hall["rate"] is None:
        print("Hallucination rate  n/a (no scored rows)")
    else:
        print(
            f"Hallucination rate  {hall['rate']:.2%}  "
            f"({hall['flagged_total']}/{hall['denominator']} flagged — {hall['method']})"
        )
    print(f"\nWrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
