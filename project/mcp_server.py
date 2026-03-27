"""
mcp_server.py — FastMCP server exposing CPI data tools.

This server is the MCP layer in the architecture:

  LangGraph Orchestrator
    └─► MCP Client (LangChain tool wrapper in tools.py)
          └─► MCP Server (this file — FastMCP)
                └─► export_cpi_data / log_summary tools
                      └─► CPI CSV data

Tools exposed:
  export_cpi_data  — filter CPI data by year range + division, return JSON + CSV
  log_summary      — append a Q&A interaction to a JSONL audit log

Standalone usage (HTTP mode, port 8001):
  python project/mcp_server.py

Calling via MCP protocol (from other services):
  POST http://localhost:8001/mcp
  See API_GATEWAY.md for full integration docs.
"""

from __future__ import annotations

import datetime
import io
import json
import sys
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

# ── MCP Server instance ───────────────────────────────────────────────────────

mcp = FastMCP(
    "CPI Data Server",
    instructions=(
        "Malaysia annual Consumer Price Index (CPI) data tools. "
        "Use export_cpi_data to retrieve structured CPI records for a year range. "
        "Use log_summary to record Q&A interactions."
    ),
)

# ── Constants ─────────────────────────────────────────────────────────────────

DIVISION_LABELS: dict[str, str] = {
    "overall": "All groups (headline CPI)",
    "01": "Food and non-alcoholic beverages",
    "02": "Alcoholic beverages and tobacco",
    "03": "Clothing and footwear",
    "04": "Housing, water, electricity, gas and other fuels",
    "05": "Furnishings, household equipment and routine maintenance",
    "06": "Health",
    "07": "Transport",
    "08": "Communication",
    "09": "Recreation and culture",
    "10": "Education",
    "11": "Restaurants and hotels",
    "12": "Miscellaneous goods and services",
    "13": "Other",
}

VALID_DIVISIONS = set(DIVISION_LABELS.keys())
DATASET_MIN_YEAR = 1960
DATASET_MAX_YEAR = 2025

# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_cpi_df() -> pd.DataFrame:
    csv_path = config.resolve_cpi_csv_path()
    if not csv_path:
        raise FileNotFoundError(
            "CPI CSV not found. Place it at project/data/cpi_2d_annual.csv "
            "or project/parent_store/CPI 2D Annual.csv."
        )
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year.astype("Int64")
    df["division"] = df["division"].astype(str).str.strip()
    return df


# ══════════════════════════════════════════════════════════════════
# MCP TOOL 1 — export_cpi_data
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
def export_cpi_data(
    start_year: int,
    end_year: int,
    division: str = "overall",
) -> str:
    """Malaysia annual CPI for a year range and division (JSON + csv_content).

    Expose via agent **only** when the user clearly requests **export/download/CSV/spreadsheet/raw file**.
    For factual Q&A use RAG tools (search + parent retrieval), not this endpoint.

    Args:
        start_year: First year of the range (1960–2025).
        end_year:   Last year of the range (1960–2025, must be >= start_year).
        division:   "overall" (default) — headline CPI only. Use when no division specified.
                    "all"              — every division (overall + 01 to 13). Use ONLY when
                                         the user explicitly says "all divisions", "by division",
                                         "breakdown", or "every category".
                    "01" to "13"       — single named COICOP category (e.g. "food" → "01").

    Returns:
        JSON string with records, csv_content (clean 4 d.p.), and source metadata.
    """
    div = str(division).strip().lower()

    # Validate division
    if div not in ("all",) and div not in VALID_DIVISIONS:
        return json.dumps({
            "error": (
                f"Unknown division '{division}'. "
                f"Use 'all', 'overall', or a COICOP code: {sorted(VALID_DIVISIONS)}"
            )
        })

    sy = max(int(start_year), DATASET_MIN_YEAR)
    ey = min(int(end_year), DATASET_MAX_YEAR)
    if sy > ey:
        return json.dumps({"error": f"start_year ({sy}) must be <= end_year ({ey})."})

    try:
        df = _load_cpi_df()
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)})

    # Filter by year; optionally by division
    year_mask = (df["year"] >= sy) & (df["year"] <= ey)
    filtered = df[year_mask].copy() if div == "all" else df[year_mask & (df["division"] == div)].copy()
    filtered = filtered.sort_values(["division", "year"])[["year", "division", "index"]].copy()
    filtered["year"] = filtered["year"].astype(int)
    filtered["index"] = filtered["index"].round(4)

    if filtered.empty:
        return json.dumps({
            "error": f"No data found for division='{div}' between {sy} and {ey}."
        })

    # Build records with human-readable labels
    records = [
        {
            "year": int(row["year"]),
            "division": str(row["division"]),
            "division_label": DIVISION_LABELS.get(str(row["division"]), f"Division {row['division']}"),
            "index": float(row["index"]),
        }
        for _, row in filtered.iterrows()
    ]

    buf = io.StringIO()
    filtered.to_csv(buf, index=False, float_format="%.4f")
    csv_content = buf.getvalue()

    div_summary = (
        "all divisions (overall + 01 to 13)"
        if div == "all"
        else f"{div} — {DIVISION_LABELS.get(div, '')}"
    )

    return json.dumps(
        {
            "division": div,
            "division_summary": div_summary,
            "start_year": sy,
            "end_year": ey,
            "record_count": len(records),
            "records": records,
            "csv_content": csv_content,
            "source": "CPI 2D Annual.csv",
            "dataset": "OpenDOSM Annual CPI by 2-digit division (DOSM Malaysia)",
            "license": "CC BY 4.0 - https://open.dosm.gov.my/data-catalogue/cpi_annual",
        },
        ensure_ascii=False,
        indent=2,
    )


# ══════════════════════════════════════════════════════════════════
# MCP TOOL 2 — log_summary
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
def log_summary(query: str, answer: str, sources: str = "") -> str:
    """Append a Q&A interaction to the audit log (JSONL).

    Call this after generating an answer so interactions are recorded for
    evaluation, monitoring, and responsible-AI auditing.

    Args:
        query:   The user's original question.
        answer:  The answer provided (first 500 chars stored).
        sources: Comma-separated source filenames cited (optional).

    Returns:
        Confirmation string with timestamp.
    """
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "interactions.jsonl"

    entry = {
        "timestamp": ts,
        "query": query,
        "answer_preview": (answer or "")[:500],
        "sources": sources,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return f"Logged at {ts} → {log_path}"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CPI MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="http",
        help="Transport: 'http' (default, port-based) or 'stdio' (for Claude Desktop)",
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if args.transport == "http":
        print(f"Starting CPI MCP Server on http://{args.host}:{args.port}/mcp")
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
