"""
Agentic RAG — CPI chatbot entry point.

Usage:
  python project/app.py chat             # interactive terminal chat
  python project/app.py chat --reset     # clear history, then chat
  python project/app.py server           # start FastAPI streaming API (default: 0.0.0.0:8000)
  python project/app.py server --port 9000 --reload
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

_project_dir = Path(__file__).resolve().parent
_root_dir = _project_dir.parent
for _p in (_project_dir, _root_dir):
    _env_file = _p / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        break

sys.path.insert(0, str(_project_dir))

from core.rag_system import RAGSystem


def chat(rag: RAGSystem, message: str) -> str:
    if not rag.agent_graph:
        return "System not initialized."
    try:
        result = rag.agent_graph.invoke(
            {"messages": [HumanMessage(content=message.strip())]},
            rag.get_config(),
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {e}"


def run_chat(reset: bool = False):
    print("Initializing RAG system...")
    rag = RAGSystem()
    rag.initialize()

    if reset:
        rag.reset_thread()

    print("\nCPI RAG Chat — type your question and press Enter.")
    print("Commands:  /reset   clear conversation history")
    print("           /quit    exit\n")

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
            rag.reset_thread()
            print("[Conversation reset]\n")
            continue

        response = chat(rag, user_input)
        print(f"\nAssistant: {response}\n")


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI streaming API server."""
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    print(f"Starting CPI RAG API server on http://{host}:{port}")
    print(f"  Docs:    http://{host}:{port}/docs")
    print(f"  Stream:  POST http://{host}:{port}/v1/chat/stream")
    print(f"  Sync:    POST http://{host}:{port}/v1/chat\n")
    uvicorn.run("api:app", host=host, port=port, reload=reload)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CPI Agentic RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python app.py chat\n"
            "  python app.py chat --reset\n"
            "  python app.py server\n"
            "  python app.py server --port 9000 --reload\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chat_p = sub.add_parser("chat", help="Interactive terminal chat")
    chat_p.add_argument("--reset", action="store_true", help="Clear history before starting")

    srv_p = sub.add_parser("server", help="Start FastAPI streaming API")
    srv_p.add_argument("--host",   default="0.0.0.0")
    srv_p.add_argument("--port",   type=int, default=8000)
    srv_p.add_argument("--reload", action="store_true", help="Hot-reload (dev only)")

    args = parser.parse_args()

    if args.command == "chat":
        run_chat(reset=getattr(args, "reset", False))
    elif args.command == "server":
        run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
