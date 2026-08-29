#!/usr/bin/env python3
"""Docling conversion runner — documents (incl. audio/video/VTT) to markdown.

Converts any rag-kit docling format to markdown text without touching the
RAG database. Useful for batch-converting corpora, previewing extraction,
or transcribing audio/video into text.

Examples:
    python scripts/docling_convert.py manual.pdf notes.docx call.mp3
    python scripts/docling_convert.py --dir ./docs --recursive --out-dir ./md
    python scripts/docling_convert.py speech.flac --stdout
    python scripts/docling_convert.py --check-gpu
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Make the package importable from a git checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_kit import _docling  # noqa: E402


def collect_inputs(paths: list[Path], recursive: bool) -> list[Path]:
    """Expand file/dir arguments into a deduped, ordered file list."""
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            files.extend(f for f in it if f.is_file() and f.suffix.lower() in _docling.DOCLING_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"SKIP (not found): {p}", file=sys.stderr)
    seen: set[str] = set()
    unique: list[Path] = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def check_gpu() -> None:
    """Diagnostic: docling, CUDA EP, ASR, ffmpeg availability."""
    print(f"docling installed: {_docling.is_available()}")
    if not _docling.is_available():
        return
    print(f"CUDA EP usable (GPU OCR): {_docling.cuda_ep_usable()}")
    try:
        import onnxruntime as ort

        print(f"onnxruntime providers: {ort.get_available_providers()}")
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401

        print("Whisper (audio/video ASR): installed")
    except ImportError:
        print("Whisper (audio/video ASR): NOT installed — pip install 'rag-kit[docling-asr]'")
    print(f"ffmpeg (video decoding): {shutil.which('ffmpeg') or 'NOT found'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="docling_convert",
        description="Convert documents (incl. audio/video) to markdown via Docling",
    )
    ap.add_argument("inputs", nargs="*", help="Files and/or directories")
    ap.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into directories",
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        metavar="DIR",
        help="Write <name>.md files into DIR (default: next to each input)",
    )
    ap.add_argument(
        "--stdout",
        action="store_true",
        help="Print markdown to stdout instead of writing files",
    )
    ap.add_argument(
        "--check-gpu",
        action="store_true",
        help="Print docling/GPU/ASR diagnostics and exit",
    )
    args = ap.parse_args(argv)

    if args.check_gpu:
        check_gpu()
        return 0

    if not args.inputs:
        ap.error("no inputs given")
    if not _docling.is_available():
        print(
            f"docling is not installed. {_docling.DOCLING_INSTALL_HINT}",
            file=sys.stderr,
        )
        return 2

    files = collect_inputs(args.inputs, args.recursive)
    if not files:
        print("No supported files found.", file=sys.stderr)
        return 1

    ok = 0
    for f in files:
        if f.suffix.lower() not in _docling.DOCLING_EXTS:
            print(f"SKIP {f.name}: not a docling format (plain text needs no conversion)")
            continue
        try:
            out = _docling.extract_document(str(f))
        except Exception as exc:  # noqa: BLE001 — report per-file, keep going
            print(f"FAIL {f.name}: {exc}", file=sys.stderr)
            print(f"     hint: {_docling.install_hint_for(f.suffix)}", file=sys.stderr)
            continue

        md = out["text"]
        if not md.strip():
            print(f"WARN {f.name}: empty output (e.g. no speech detected in audio)")
        if args.stdout:
            print(f"===== {f.name} =====")
            print(md)
            dest = "stdout"
        else:
            if args.out_dir:
                dest = Path(args.out_dir) / f"{f.stem}.md"
            else:
                dest = f.with_suffix(".md")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(md, encoding="utf-8")
        print(f"OK   {f.name}: {len(md)} chars, {len(out['headings'])} headings -> {dest}")
        ok += 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
