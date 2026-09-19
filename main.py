"""CLI entry point for LocalMind Agent."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent import run_agent, synthesize_answer
from src.config import RAW_DOCS_PATH
from src.loader import build_chunks
from src.vectorstore import collection_size, index_documents

console = Console()


def cmd_index(args) -> None:
    console.print(Panel(f"[bold]Indexing[/bold] from {args.path}", style="cyan"))
    chunks = build_chunks(args.path)
    if not chunks:
        console.print("[yellow]No documents found.[/yellow]")
        return
    n = index_documents(chunks, reset=args.reset)
    console.print(
        f"[green]Indexed {n} chunks.[/green] Total stored: {collection_size()}"
    )


def cmd_ask(args) -> None:
    if collection_size() == 0:
        console.print(
            "[yellow]Vector store is empty. Run `python main.py index` first.[/yellow]"
        )
        sys.exit(1)

    console.print(Panel(f"[bold]Q:[/bold] {args.question}", style="blue"))
    with console.status("[dim]Thinking...[/dim]"):
        if args.mode == "rag":
            answer = synthesize_answer(args.question)
        else:
            answer = run_agent(args.question)
    console.print(Markdown(answer))


def cmd_shell(_args) -> None:
    console.print(Panel("LocalMind Agent REPL — type 'exit' to quit", style="green"))
    while True:
        try:
            q = console.input("[bold blue]you ›[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if q.lower() in {"exit", "quit", ":q"}:
            break
        if not q:
            continue
        with console.status("[dim]Thinking...[/dim]"):
            answer = run_agent(q)
        console.print(Markdown(answer))
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="localmind", description="LocalMind Agent CLI"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Ingest documents into the vector store")
    p_index.add_argument("--path", default=RAW_DOCS_PATH, help="Folder with raw docs")
    p_index.add_argument("--reset", action="store_true", help="Wipe collection first")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a one-off question")
    p_ask.add_argument("question", help="The question to ask")
    p_ask.add_argument(
        "--mode",
        choices=["agent", "rag"],
        default="agent",
        help="agent = ReAct loop, rag = direct retrieval + synthesis",
    )
    p_ask.set_defaults(func=cmd_ask)

    p_shell = sub.add_parser("shell", help="Interactive REPL")
    p_shell.set_defaults(func=cmd_shell)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    Path(RAW_DOCS_PATH).mkdir(parents=True, exist_ok=True)
    main()
