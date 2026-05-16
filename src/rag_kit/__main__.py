"""CLI entry point for rag-kit."""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="rag-kit — standalone RAG for text files"
    )
    parser.add_argument(
        "--db", default=os.path.expanduser("~/.rag-kit/rag.db"),
        help="Database path (default: ~/.rag-kit/rag.db)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # load-url
    url_p = sub.add_parser("load-url", help="Load text from URL")
    url_p.add_argument("url", help="URL to load")
    url_p.add_argument("--chunk-size", type=int, default=2500)
    url_p.add_argument("--overlap", type=int, default=200)

    # load-file
    file_p = sub.add_parser("load-file", help="Load text from local file")
    file_p.add_argument("path", help="File path")
    file_p.add_argument("--chunk-size", type=int, default=2500)
    file_p.add_argument("--overlap", type=int, default=200)

    # query
    q_p = sub.add_parser("query", help="Ask a question about a loaded file")
    q_p.add_argument("file_id", type=int, help="File ID")
    q_p.add_argument("question", help="Your question")

    # search
    s_p = sub.add_parser("search", help="Keyword search without LLM")
    s_p.add_argument("file_id", type=int)
    s_p.add_argument("query", help="Search keywords")

    # list
    sub.add_parser("list", help="List loaded files")

    # stats
    sub.add_parser("stats", help="Show database stats")

    # delete
    d_p = sub.add_parser("delete", help="Delete a file")
    d_p.add_argument("file_id", type=int)

    args = parser.parse_args()

    from rag_kit import RAGSystem

    rag = RAGSystem(db_path=args.db)

    if args.command == "load-url":
        fid = rag.load_url(args.url, args.chunk_size, args.overlap)
        print(f"Loaded — file_id: {fid}")

    elif args.command == "load-file":
        fid = rag.load_file(args.path, args.chunk_size, args.overlap)
        print(f"Loaded — file_id: {fid}")

    elif args.command == "query":
        answer = rag.query(args.file_id, args.question)
        print(answer)

    elif args.command == "search":
        results = rag.search(args.file_id, args.query)
        if not results:
            print("No matches found.")
        for r in results:
            print(f"  Chunk #{r['index']} (score: {r['score']})")
            print(f"  {r['preview'][:200]}")
            print()

    elif args.command == "list":
        files = rag.list_files()
        if not files:
            print("No files loaded.")
        for f in files:
            print(f"  #{f['file_id']}  {f['filename']}  "
                  f"({f['total_chunks']} chunks, "
                  f"accessed: {str(f['last_accessed'])[:19]})")

    elif args.command == "stats":
        st = rag.stats()
        print(f"Files: {st['total_files']}")
        print(f"Chunks: {st['total_chunks']}")

    elif args.command == "delete":
        ok = rag.delete_file(args.file_id)
        print("Deleted." if ok else "Not found.")


if __name__ == "__main__":
    main()
