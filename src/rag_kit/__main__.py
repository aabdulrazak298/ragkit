"""CLI entry point for rag-kit."""

import argparse
import json
import os
import sys


def main():
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
        if args.file_id:
            result = rag.query(args.file_id, args.question)
        elif args.namespace:
            result = rag.query(args.question, namespace=args.namespace)
        else:
            result = rag.query(args.question)
        print(result.answer)
        if result.citations:
            print("\n--- Citations ---")
            for c in result.citations:
                print(f"  File #{c['file_id']}, chunk {c['chunk_index']} "
                      f"(score: {c['score']:.2f})")

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
                print(f"  File #{fid} ({ns}) chunk {ci} "
                      f"(score: {r.get('score', 0):.2f})")
                print(f"  {r.get('preview', r.get('text', ''))[:200]}")
                print()

    elif args.command == "list":
        files = rag.list(namespace=args.namespace)
        if not files:
            print("No files loaded.")
        for f in files:
            print(f"  #{f['file_id']:>4}  {f['namespace']:15}  "
                  f"{f['filename']:30}  ({f['total_chunks']} chunks, "
                  f"{str(f['last_accessed'])[:19]})")

    elif args.command == "stats":
        st = rag.stats()
        print(f"Files:  {st['total_files']}")
        print(f"Chunks: {st['total_chunks']}")

    elif args.command == "delete":
        ok = rag.delete_file(args.file_id)
        print("Deleted." if ok else "Not found.")


if __name__ == "__main__":
    main()
