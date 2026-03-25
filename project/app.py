import sys
import os
import argparse

from pathlib import Path
from dotenv import load_dotenv

_project_dir = Path(__file__).resolve().parent
_root_dir = _project_dir.parent
for _p in (_project_dir, _root_dir):
    _env_file = _p / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        break

sys.path.insert(0, str(_project_dir))

from core.rag_system import RAGSystem
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager


def run_chat():
    print("Initializing RAG system...")
    rag = RAGSystem()
    rag.initialize()
    chat = ChatInterface(rag)

    print("\nRAG Chat — type your message and press Enter.")
    print("Commands: /reset  clear conversation history")
    print("          /quit   exit\n")

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Bye!")
            break

        if user_input.lower() == "/reset":
            chat.clear_session()
            history = []
            print("[Conversation reset]\n")
            continue

        response = chat.chat(user_input, history)
        history.append((user_input, response))
        print(f"\nAssistant: {response}\n")


def run_index(paths: list[str], clear: bool, list_docs: bool):
    print("Initializing RAG system...")
    rag = RAGSystem()
    rag.initialize()
    doc_manager = DocumentManager(rag)

    if clear:
        doc_manager.clear_all()
        print("All indexed documents cleared.")
        return

    if list_docs:
        files = doc_manager.get_markdown_files()
        if files:
            print(f"Indexed documents ({len(files)}):")
            for f in files:
                print(f"  - {f}")
        else:
            print("No documents indexed yet.")
        return

    if not paths:
        print("No file paths provided. Use --list to see indexed docs or --clear to wipe them.")
        return

    def progress(ratio, msg):
        bar_len = 30
        filled = int(bar_len * ratio)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {msg}", end="", flush=True)

    print(f"Indexing {len(paths)} file(s)...")
    added, skipped = doc_manager.add_documents(paths, progress_callback=progress)
    print(f"\nDone. Added: {added}, Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG — terminal interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py chat
  python app.py index path/to/doc.pdf path/to/doc2.md
  python app.py index --list
  python app.py index --clear
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("chat", help="Start an interactive chat session")

    index_parser = subparsers.add_parser("index", help="Index documents into the knowledge base")
    index_parser.add_argument("paths", nargs="*", help="PDF or Markdown files to index")
    index_parser.add_argument("--list", action="store_true", help="List currently indexed documents")
    index_parser.add_argument("--clear", action="store_true", help="Clear all indexed documents")

    args = parser.parse_args()

    if args.command == "chat":
        run_chat()
    elif args.command == "index":
        run_index(args.paths, args.clear, args.list)


if __name__ == "__main__":
    main()
