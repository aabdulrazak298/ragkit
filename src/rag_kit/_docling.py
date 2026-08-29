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

import re

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
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
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
}

DOCLING_INSTALL_HINT = "Install rag-kit[docling]: pip install 'rag-kit[docling]'"

# Markdown headings in the export: "# Title", "## Sub", ... offset = byte
# position of the heading line in the exported markdown (exact by
# construction — we parse the very text we return).
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

_converter = None
_gpu_checked = False


def is_available() -> bool:
    """True when docling is importable (i.e. rag-kit[docling] installed)."""
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
