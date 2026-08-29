"""Docling-backed document extraction — optional heavy extra.

When `rag-kit[docling]` is installed, rich formats (PDF, DOCX, PPTX,
XLSX, HTML, EPUB, ODT, images, ...) are converted by IBM Docling instead
of the lightweight per-format extractors. Docling preserves reading
order, renders tables as markdown, OCRs scanned content (RapidOCR ships
with it — no system tesseract needed), and produces a REAL heading
hierarchy that feeds rag-kit's TOC directly instead of regex guessing.

The `docling` import is lazy: importing this module never pulls the
heavy dependencies. Use `is_available()` to check usability.

First conversion downloads docling's layout/table/OCR models
(hundreds of MB) and caches them under ~/.cache/docling; later
conversions run fully offline.
"""

from __future__ import annotations

import os
import re

# Audio/video are transcribed (Whisper) — require rag-kit[docling-asr]
# (docling[asr]) plus ffmpeg for video decoding.
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

# Formats docling converts natively when installed.
DOCLING_EXTS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".epub",
    ".odt",
    ".csv",
    ".tex",
    ".eml",
    ".adoc",
    ".vtt",
    # Legacy Office + spreadsheet siblings: converted via LibreOffice —
    # need the `soffice` binary present, not just the python package.
    ".doc",
    ".ppt",
    ".xls",
    ".ods",
    ".odp",
    # Images are OCR'd (RapidOCR ships with docling).
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    # Audio/video are transcribed (Whisper) — require the asr extra
    # (rag-kit[docling-asr]) plus ffmpeg for video decoding.
    *AUDIO_EXTS,
    *VIDEO_EXTS,
}

# Formats with NO legacy extractor in rag-kit — docling is the only path.
DOCLING_ONLY_EXTS = {
    ".xlsx",
    ".xls",
    ".doc",
    ".ppt",
    ".ods",
    ".odp",
    ".tex",
    ".eml",
    ".adoc",
    ".vtt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    *AUDIO_EXTS,
    *VIDEO_EXTS,
}

# Map extension → storage source_type label (same labels the legacy
# extractors use where one exists).
_SOURCE_TYPE_BY_EXT = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".html": "html",
    ".htm": "html",
    ".epub": "epub",
    ".odt": "odt",
    ".csv": "csv",
    ".tex": "latex",
    ".eml": "email",
    ".adoc": "asciidoc",
    ".doc": "doc",
    ".ppt": "ppt",
    ".xls": "xls",
    ".ods": "ods",
    ".odp": "odp",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
    ".vtt": "vtt",
    **{ext: "audio" for ext in AUDIO_EXTS},
    **{ext: "video" for ext in VIDEO_EXTS},
}

DOCLING_INSTALL_HINT = "Install rag-kit[docling]: pip install 'rag-kit[docling]'"
DOCLING_ASR_INSTALL_HINT = (
    "Install rag-kit[docling-asr]: pip install 'rag-kit[docling-asr]' "
    "(adds Whisper transcription for audio/video)"
)


def install_hint_for(ext: str) -> str:
    """Actionable install hint for an extension."""
    if ext.lower() in AUDIO_EXTS or ext.lower() in VIDEO_EXTS:
        return DOCLING_ASR_INSTALL_HINT
    return DOCLING_INSTALL_HINT


# Markdown headings in the export: "# Title", "## Sub", ... offset = byte
# position of the heading line in the exported markdown (exact by
# construction — we parse the very text we return).
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

_converter = None
_gpu_checked = False


def is_available() -> bool:
    """True when docling is importable (i.e. rag-kit[docling] installed).

    Set RAGKIT_DOCLING=0 (or "false") to disable the docling path
    entirely — loads fall back to the fast legacy extractors (pypdf,
    python-docx, ...). Useful when docling's deep parsing is overkill
    for a file or too slow on CPU.
    """
    env = os.environ.get("RAGKIT_DOCLING", "")
    if env and env.lower() in ("0", "false", "no", "off"):
        return False
    try:
        import docling  # noqa: F401

        return True
    except ImportError:
        return False


def source_type_for(ext: str) -> str:
    """Storage label for an extension handled by docling."""
    return _SOURCE_TYPE_BY_EXT.get(ext.lower(), "docling")


def _get_converter():
    """Lazily build and cache a DocumentConverter (model artifacts are
    cached by docling itself, so repeated conversions are cheap)."""
    global _converter
    if _converter is None:
        _check_gpu_once()
        from docling.document_converter import DocumentConverter

        # Defaults already do the right thing in docling 2.x:
        # do_table_structure=True, do_ocr=True (auto-detected OCR engine,
        # RapidOCR bundled with the package).
        _converter = DocumentConverter()
    return _converter


def _probe_model_path() -> str | None:
    """Any local .onnx file to use for a CUDA-EP load test (rapidocr
    ships its models with the package)."""
    try:
        from importlib import resources

        base = resources.files("rapidocr") / "models"
        if base.is_dir():
            for p in base.iterdir():
                if p.name.endswith(".onnx"):
                    return str(p)
    except Exception:
        pass
    return None


def cuda_ep_usable() -> bool:
    """True when onnxruntime's CUDA execution provider REALLY loads.

    ort.get_available_providers() lists CUDAExecutionProvider even when
    the CUDA/cuDNN libs are missing or the versions don't match — the
    session then silently falls back to CPU. This does a real session
    creation on a local model to test what will actually run.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    model = _probe_model_path()
    if model is None:
        # Nothing to load-test with — trust the provider list.
        return True
    try:
        sess = ort.InferenceSession(
            model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        return sess.get_providers()[0] == "CUDAExecutionProvider"
    except Exception:
        return False


def _check_gpu_once() -> None:
    """Warn once when onnxruntime-gpu is installed but its CUDA EP can't
    actually load — otherwise GPU OCR silently degrades to CPU."""
    global _gpu_checked
    if _gpu_checked:
        return
    _gpu_checked = True
    try:
        import onnxruntime as ort
    except ImportError:
        return
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return  # CPU-only ort — expected on non-GPU installs
    if cuda_ep_usable():
        return
    import warnings

    warnings.warn(
        "onnxruntime-gpu is installed but the CUDA execution provider "
        "failed to load — docling OCR will run on CPU. Likely causes: "
        "(1) onnxruntime-gpu version vs CUDA/cuDNN mismatch (1.29 needs "
        "CUDA 13 / cuDNN 9; CUDA 12 machines: pip install "
        "'onnxruntime-gpu<1.23'), or (2) the CUDA runtime libs aren't on "
        "LD_LIBRARY_PATH — torch's bundled nvidia pip packages provide "
        "them: export LD_LIBRARY_PATH=<site-packages>/nvidia/*/lib:"
        "$LD_LIBRARY_PATH. Check with: from rag_kit import _docling; "
        "_docling.cuda_ep_usable()",
        RuntimeWarning,
        stacklevel=3,
    )


def extract_document(path: str) -> dict:
    """Convert a document with docling.

    Returns:
        {
          "text": markdown export (headings + tables preserved),
          "headings": [{"title", "level", "offset"}] parsed from the
                      markdown — level is the markdown heading depth
                      (1 = "#"), offset is the char position in `text`,
          "source_type": storage label,
        }
    """
    converter = _get_converter()
    result = converter.convert(path)
    doc = result.document
    text = doc.export_to_markdown()
    # Surrogate safety — same guard as the legacy extractors.
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    headings = [
        {
            "title": match.group(2).strip(),
            "level": len(match.group(1)),
            "offset": match.start(),
        }
        for match in _MD_HEADING_RE.finditer(text)
    ]
    return {"text": text, "headings": headings}
