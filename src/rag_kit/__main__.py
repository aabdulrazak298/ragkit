"""CLI entry point for rag-kit."""

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-kit",
        description="Standalone RAG for text files — load, search, query with LLM",
    )
    parser.add_argument(
        "--db",
        default=os.path.expanduser("~/.rag-kit/rag.db"),
        help="Database path (default: ~/.rag-kit/rag.db)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # load-url
    url_p = sub.add_parser("load-url", help="Load text from URL")
    url_p.add_argument("url")
    url_p.add_argument("--namespace", "-n", default="default")
    url_p.add_argument("--chunk-size", type=int, default=2500)
    url_p.add_argument("--overlap", type=int, default=200)

    # load-file
    file_p = sub.add_parser("load-file", help="Load text from local file")
    file_p.add_argument("path")
    file_p.add_argument("--namespace", "-n", default="default")
    file_p.add_argument("--chunk-size", type=int, default=2500)
    file_p.add_argument("--overlap", type=int, default=200)

    # query
    q_p = sub.add_parser("query", help="Ask a question")
    q_p.add_argument("question")
    q_p.add_argument("file_id", type=int, nargs="?", help="File ID (omit for cross-file)")
    q_p.add_argument("--namespace", "-n", help="Namespace (for cross-file query)")
    q_p.add_argument(
        "--loop",
        action="store_true",
        help="Loop-enabled search (iterative retrieval with verifier)",
    )
    q_p.add_argument(
        "--max-loops", type=int, default=4, help="Max retrieval rounds for --loop (default 4)"
    )

    # search
    s_p = sub.add_parser("search", help="Keyword search")
    s_p.add_argument("query")
    s_p.add_argument("file_id", type=int, nargs="?", help="File ID (omit for cross-file)")
    s_p.add_argument("--namespace", "-n", help="Namespace (for cross-file search)")

    # list
    list_p = sub.add_parser("list", help="List loaded files")
    list_p.add_argument("--namespace", "-n", help="Filter by namespace")

    # stats
    sub.add_parser("stats", help="Show database stats")

    # delete
    d_p = sub.add_parser("delete", help="Delete a file")
    d_p.add_argument("file_id", type=int)

    # ui
    ui_p = sub.add_parser("ui", help="Launch the Gradio web UI in your browser")
    ui_p.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    ui_p.add_argument("--port", type=int, default=7860, help="Port (default 7860)")
    ui_p.add_argument("--share", action="store_true", help="Create a temporary public share link")
    ui_p.add_argument(
        "--embed-backend",
        default="api",
        choices=["api", "local"],
        help="Vector backend: api (OpenRouter embeddings) or local (MiniLM, no key)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    from rag_kit import RAGSystem

    rag = RAGSystem(db_path=args.db)

    if args.command == "load-url":
        fid = rag.load_url(
            args.url,
            namespace=args.namespace,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        print(f"Loaded — file_id: {fid} (namespace: {args.namespace})")

    elif args.command == "load-file":
        fid = rag.load_file(
            args.path,
            namespace=args.namespace,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        print(f"Loaded — file_id: {fid} (namespace: {args.namespace})")

    elif args.command == "query":
        if args.loop and not args.file_id:
            print("query --loop requires a file_id (loop mode is file-scoped).")
            sys.exit(2)
        if args.loop:
            result = rag.query_loop(args.file_id, args.question, max_loops=args.max_loops)
        elif args.file_id:
            result = rag.query(args.file_id, args.question)
        elif args.namespace:
            result = rag.query(args.question, namespace=args.namespace)
        else:
            result = rag.query(args.question)
        print(result.answer)
        if result.metrics:
            print("\n--- Metrics ---")
            for k, v in result.metrics.items():
                print(f"  {k}: {v}")
        if result.citations:
            print("\n--- Citations ---")
            for c in result.citations:
                print(
                    f"  File #{c['file_id']}, chunk {c['chunk_index']} "
                    f"(score: {c.get('score', 0):.2f})"
                )

    elif args.command == "search":
        kwargs = {"query": args.query}
        if args.file_id:
            kwargs["file_id"] = args.file_id
        elif args.namespace:
            kwargs["namespace"] = args.namespace
        results = rag.search(**kwargs)
        if not results:
            print("No matches found.")
        else:
            for r in results[:10]:
                ci = r.get("chunk_index", r.get("index", "?"))
                fid = r.get("file_id", "?")
                ns = r.get("namespace", args.namespace or "?")
                print(f"  File #{fid} ({ns}) chunk {ci} (score: {r.get('score', 0):.2f})")
                print(f"  {r.get('preview', r.get('text', ''))[:200]}")
                print()

    elif args.command == "list":
        files = rag.list(namespace=args.namespace)
        if not files:
            print("No files loaded.")
        for f in files:
            print(
                f"  #{f['file_id']:>4}  {f['namespace']:15}  "
                f"{f['filename']:30}  ({f['total_chunks']} chunks, "
                f"{str(f['last_accessed'])[:19]})"
            )

    elif args.command == "stats":
        st = rag.stats()
        print(f"Files:  {st['total_files']}")
        print(f"Chunks: {st['total_chunks']}")

    elif args.command == "delete":
        ok = rag.delete_file(args.file_id)
        print("Deleted." if ok else "Not found.")

    elif args.command == "ui":
        try:
            from rag_kit._ui import RAGApp, build_app

            demo = build_app(RAGApp(db_path=args.db, embed_backend=args.embed_backend))
        except ImportError:
            print(
                'The web UI needs gradio. Install it with:  pip install "rag-kit[ui]"',
                file=sys.stderr,
            )
            sys.exit(1)
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
