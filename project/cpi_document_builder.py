"""
Build LangChain Documents for DOSM annual CPI (2-digit division) CSV.

Chunking strategy (tabular, not prose):
- Child: one row → one short natural-language fact (best for semantic + BM25 retrieval).
- Parent: one document per division → full available time series in that division (trend context).
Recursive character splitting on raw CSV is avoided so numeric rows are never torn apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from langchain_core.documents import Document

# COICOP-style labels aligned with Malaysian CPI 2-digit groups (verify against DOSM if needed).
DIVISION_LABELS: dict[str, str] = {
    "overall": "All groups (headline CPI)",
    "01": "Food and non-alcoholic beverages",
    "02": "Alcoholic beverages and tobacco",
    "03": "Clothing and footwear",
    "04": "Housing, water, electricity, gas and other fuels",
    "05": "Furnishings, household equipment and routine household maintenance",
    "06": "Health",
    "07": "Transport",
    "08": "Communication",
    "09": "Recreation and culture",
    "10": "Education",
    "11": "Restaurants and hotels",
    "12": "Miscellaneous goods and services",
    "13": "Division 13 (see DOSM COICOP mapping for exact label)",
}

DATASET_META = "cpi_dosm_annual_2d"


def division_label(code: str) -> str:
    return DIVISION_LABELS.get(str(code).strip(), f"Division {code}")


def parent_id_for_division(division: str) -> str:
    safe = str(division).strip().replace("/", "_")
    return f"cpi_2dannual_{safe}_parent"


def build_cpi_corpus(csv_path: str | Path) -> Tuple[List[Tuple[str, Document]], List[Document]]:
    """
    Returns:
        parent_pairs: list of (parent_id, Document) for ParentStoreManager.save_many
        child_docs: Documents to embed into Qdrant (child collection)
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CPI CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"date", "division", "index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CPI CSV missing columns {missing}; got {list(df.columns)}")

    df = df.copy()
    df["division"] = df["division"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "division"])
    df["year"] = df["date"].dt.year.astype(int)

    source_name = path.name

    parent_pairs: List[Tuple[str, Document]] = []
    child_docs: List[Document] = []

    for division, g in df.groupby("division", sort=True):
        g = g.sort_values("date")
        label = division_label(division)
        pid = parent_id_for_division(division)

        lines = [
            f"{int(row['year'])}: {float(row['index']):.6f}"
            for _, row in g.iterrows()
        ]
        parent_body = (
            f"## Malaysia CPI — annual index by 2-digit division (DOSM / OpenDOSM)\n"
            f"Division: {label} (code: {division})\n"
            f"Source file: {source_name}\n"
            f"Series (year: index value):\n"
            + "\n".join(lines)
        )

        parent_meta = {
            "source": source_name,
            "parent_id": pid,
            "division": division,
            "dataset": DATASET_META,
            "kind": "cpi_division_series",
        }
        parent_pairs.append((pid, Document(page_content=parent_body, metadata=parent_meta)))

        for _, row in g.iterrows():
            year = int(row["year"])
            idx = float(row["index"])
            child_text = (
                f"Malaysia annual Consumer Price Index (CPI), 2-digit division series (DOSM OpenDOSM). "
                f"{label} (division code {division}). "
                f"Year {year}. CPI index value: {idx:.6f}. "
                f"Source: {source_name}."
            )
            child_meta = {
                "source": source_name,
                "parent_id": pid,
                "division": division,
                "year": year,
                "index_value": idx,
                "dataset": DATASET_META,
                "kind": "cpi_row",
            }
            child_docs.append(Document(page_content=child_text, metadata=child_meta))

    return parent_pairs, child_docs
