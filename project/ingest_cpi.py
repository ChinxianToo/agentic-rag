"""
Ingest DOSM annual CPI CSV into Qdrant (hybrid dense + sparse) and parent_store.

Requires: Ollama running with DENSE_MODEL (see config.py), Qdrant reachable (QDRANT_URL or embedded path).

Usage (from repo root):
  python project/ingest_cpi.py --reset

Or from project/:
  cd project && python ingest_cpi.py --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_project_dir = Path(__file__).resolve().parent
_root_dir = _project_dir.parent
for _p in (_project_dir, _root_dir):
    _env = _p / ".env"
    if _env.exists():
        load_dotenv(_env)
        break

sys.path.insert(0, str(_project_dir))

import config
from cpi_document_builder import build_cpi_corpus
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CPI 2D Annual CSV into Qdrant + parent_store")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to CPI CSV (default: first existing path from config.resolve_cpi_csv_path())",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete vector collection, recreate it, and clear parent_store before ingest (recommended for a clean CPI-only index)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build documents and print counts only (no Qdrant / no disk writes)",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        resolved = config.resolve_cpi_csv_path()
        if resolved is None:
            print(
                "CPI CSV not found.\n"
                "  Download the annual series from OpenDOSM (data catalogue: cpi_annual) and save it as:\n"
                f"    • {config.CPI_CSV_PATH}\n"
                f"    • {config.CPI_CSV_PATH_DATA}  (recommended if parent_store is gitignored)\n"
                "  Or pass an explicit path:  python project/ingest_cpi.py --csv /path/to/file.csv\n"
                "\n"
                "Note: `parent_store/` is gitignored — clones often have no CSV until you add one.",
                file=sys.stderr,
            )
            sys.exit(1)
        csv_path = Path(resolved)
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    parent_pairs, child_docs = build_cpi_corpus(csv_path)
    print(f"Built {len(parent_pairs)} parent series and {len(child_docs)} child (row) documents.")

    if args.dry_run:
        return

    if args.reset:
        vdb = VectorDbManager()
        vdb.delete_collection(config.CHILD_COLLECTION)
        vdb.create_collection(config.CHILD_COLLECTION)
        ParentStoreManager().clear_store()

    vdb = VectorDbManager()
    vdb.create_collection(config.CHILD_COLLECTION)
    collection = vdb.get_collection(config.CHILD_COLLECTION)

    batch = 64
    for i in range(0, len(child_docs), batch):
        collection.add_documents(child_docs[i : i + batch])

    ParentStoreManager().save_many(parent_pairs)
    print(f"Indexed {len(child_docs)} vectors into collection '{config.CHILD_COLLECTION}'.")
    print(f"Saved {len(parent_pairs)} parent JSON files under {config.PARENT_STORE_PATH}.")


if __name__ == "__main__":
    main()
