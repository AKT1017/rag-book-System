import argparse
import json
import os
import sys
from pathlib import Path

from rag_book_agent.config import load_settings
from rag_book_agent.evaluation import Evaluator
from rag_book_agent.service import RagService


def project_directory() -> Path:
    configured = os.environ.get("RAG_BOOK_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd()
    if (current / "pyproject.toml").exists() or (current / "config.json").exists():
        return current
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-book", description="Local RAG for books")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Create the local configuration and database")

    ingest_parser = subparsers.add_parser("ingest", help="Import a document or directory")
    ingest_parser.add_argument("path")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question")

    search_parser = subparsers.add_parser("search", help="Search without generation")
    search_parser.add_argument("question")
    search_parser.add_argument("--limit", type=int, default=6)

    subparsers.add_parser("stats", help="Show library statistics")
    subparsers.add_parser("doctor", help="Check the local runtime")

    golden_parser = subparsers.add_parser("golden-add", help="Add an evaluation question")
    golden_parser.add_argument("question")
    golden_parser.add_argument("chunk_ids", help="Comma-separated expected chunk IDs")
    golden_parser.add_argument("--answer", default="")

    eval_parser = subparsers.add_parser("eval", help="Run retrieval evaluation")
    eval_parser.add_argument("--top-k", type=int, default=10)
    eval_parser.add_argument("--dataset", default="all")
    eval_parser.add_argument(
        "--route", choices=("hybrid_rerank", "hybrid", "bm25", "dense"),
        default="hybrid_rerank",
    )
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--output", default="data/reports/latest.json")

    subparsers.add_parser("web", help="Start the local web interface")
    return parser


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        args.command = "web"

    home = project_directory()
    settings = load_settings(home)
    service = RagService(settings)

    if args.command == "web":
        service.close()
        from rag_book_agent.web.app import run

        run()
        return

    try:
        if args.command == "init":
            print("Initialized: %s" % home)
            print("Database: %s" % settings.database_path)
        elif args.command == "ingest":
            print(json.dumps(service.ingest(Path(args.path)), ensure_ascii=False, indent=2))
        elif args.command == "ask":
            answer = service.ask(args.question)
            print(answer.text)
            print_sources(answer.sources)
        elif args.command == "search":
            print_sources(service.search(args.question, args.limit))
        elif args.command == "stats":
            print(json.dumps(service.stats(), ensure_ascii=False, indent=2))
        elif args.command == "doctor":
            failed = False
            for name, passed, detail in service.health():
                mark = "OK" if passed else "FAIL"
                print("%-5s %-16s %s" % (mark, name, detail))
                failed = failed or not passed
            if failed:
                sys.exit(1)
        elif args.command == "golden-add":
            chunk_ids = [int(value.strip()) for value in args.chunk_ids.split(",") if value.strip()]
            service.storage.add_golden_question(args.question, chunk_ids, args.answer)
            print("Evaluation question saved.")
        elif args.command == "eval":
            evaluator = Evaluator(service.storage, service.retriever)
            report = evaluator.run(
                args.top_k, args.limit or None, args.dataset, args.route
            )
            output = Path(args.output)
            if not output.is_absolute():
                output = home / output
            evaluator.save(report, output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("Report: %s" % output)
    finally:
        service.close()


def print_sources(sources) -> None:
    if not sources:
        print("\nNo sources found.")
        return
    print("\nSources:")
    for index, source in enumerate(sources, start=1):
        print(
            "[S%d] %s | chunk=%d | score=%.4f"
            % (index, source.citation, source.chunk.id, source.rerank_score)
        )
        print("     " + source.chunk.text[:180].replace("\n", " "))


if __name__ == "__main__":
    main()
