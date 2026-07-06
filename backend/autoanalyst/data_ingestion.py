"""
data_ingestion.py
=================
Unified multi-format file processor for the AutoAnalyst module.

Supported formats
-----------------
Structured  : .csv, .xlsx, .xls
Unstructured: .txt, .pdf, .docx
Images      : .png, .jpg, .jpeg, .webp, .gif  (via OpenAI Vision)

Each ingested file is stored as either:
  - A pandas DataFrame  (structured data)
  - A plain-text string (unstructured / OCR'd content)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
IngestedResult = Union[pd.DataFrame, str]
SessionStore = Dict[str, IngestedResult]

# Supported MIME / extensions
STRUCTURED_EXTS   = {".csv", ".xlsx", ".xls"}
TEXT_EXTS         = {".txt", ".pdf", ".docx"}
IMAGE_EXTS        = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
ALL_SUPPORTED_EXTS = STRUCTURED_EXTS | TEXT_EXTS | IMAGE_EXTS


# ---------------------------------------------------------------------------
# Helper – lazy imports with helpful error messages
# ---------------------------------------------------------------------------

def _require(package: str, install_name: str | None = None):
    """Import *package*; raise ImportError with install hint on failure."""
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        dep = install_name or package
        raise ImportError(
            f"Missing optional dependency '{dep}'. "
            f"Install it with:  pip install {dep}"
        ) from None


# ---------------------------------------------------------------------------
# Per-format readers
# ---------------------------------------------------------------------------

def _read_csv(data: bytes, filename: str) -> pd.DataFrame:
    """Read CSV bytes into a DataFrame, trying common encodings."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode '{filename}' with any common encoding.")


def _read_excel(data: bytes, filename: str) -> pd.DataFrame:
    """Read .xlsx / .xls bytes into a DataFrame (requires openpyxl / xlrd)."""
    suffix = Path(filename).suffix.lower()
    engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
    try:
        return pd.read_excel(io.BytesIO(data), engine=engine)
    except ImportError as exc:
        raise ImportError(
            f"Install the required Excel engine:  pip install {'openpyxl' if suffix == '.xlsx' else 'xlrd'}"
        ) from exc


def _read_txt(data: bytes) -> str:
    """Read plain-text bytes."""
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_pdf(data: bytes, filename: str) -> str:
    """Extract text from a PDF using pypdf (requires pypdf)."""
    pypdf = _require("pypdf")
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages_text: List[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        if not pages_text:
            return f"[PDF '{filename}' appears to contain no extractable text (may be image-based).]"
        return "\n\n".join(pages_text)
    except Exception as exc:
        raise ValueError(f"Failed to read PDF '{filename}': {exc}") from exc


def _read_docx(data: bytes, filename: str) -> str:
    """Extract text from a .docx file using python-docx (requires python-docx)."""
    docx = _require("docx", "python-docx")
    try:
        doc = docx.Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as exc:
        raise ValueError(f"Failed to read DOCX '{filename}': {exc}") from exc


def _read_image_via_vision(data: bytes, filename: str) -> str:
    """
    Use OpenAI Vision to extract / describe the content of an image.
    Falls back to a placeholder if OPENAI_API_KEY is not set.
    """
    from backend.config import OPENAI_API_KEY, LLM_MODEL  # local import to avoid circular deps

    if not OPENAI_API_KEY:
        return f"[Image '{filename}' — set OPENAI_API_KEY to enable OCR/vision extraction.]"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        b64 = base64.b64encode(data).decode("utf-8")
        ext = Path(filename).suffix.lstrip(".").lower()
        mime = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "webp", "gif") else "image/png"
        if ext == "jpg":
            mime = "image/jpeg"

        response = client.chat.completions.create(
            model="gpt-4o",          # vision is only available on gpt-4o family
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all text visible in this image using OCR. "
                                "If there is no text, describe the image content in detail "
                                "including any charts, graphs, tables, or data visualisations."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )
        return response.choices[0].message.content or f"[No content extracted from '{filename}']"
    except Exception as exc:
        logger.warning("Vision OCR failed for '%s': %s", filename, exc)
        return f"[Image OCR failed for '{filename}': {exc}]"


# ---------------------------------------------------------------------------
# FileProcessor
# ---------------------------------------------------------------------------

class FileProcessor:
    """
    Ingests multiple files in one session and stores results by filename.

    Usage
    -----
    processor = FileProcessor()
    processor.ingest(filename="sales.csv", data=<bytes>)
    processor.ingest(filename="report.pdf", data=<bytes>)
    processor.ingest(filename="chart.png",  data=<bytes>)

    dfs   = processor.get_dataframes()   # {name: DataFrame}
    texts = processor.get_texts()         # {name: str}
    """

    def __init__(self) -> None:
        self._store: SessionStore = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, filename: str, data: bytes) -> IngestedResult:
        """
        Ingest a single file.  Returns the parsed result (DataFrame or str).
        Raises ValueError for unsupported or corrupted files.
        """
        suffix = Path(filename).suffix.lower()

        if suffix not in ALL_SUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {sorted(ALL_SUPPORTED_EXTS)}"
            )

        if not data:
            raise ValueError(f"File '{filename}' is empty.")

        try:
            result = self._dispatch(suffix, data, filename)
        except (ValueError, ImportError):
            raise
        except Exception as exc:
            raise ValueError(f"Failed to process '{filename}': {exc}") from exc

        self._store[filename] = result
        logger.info("Ingested '%s' → %s", filename, type(result).__name__)
        return result

    def ingest_many(
        self, files: List[Tuple[str, bytes]]
    ) -> Dict[str, Union[IngestedResult, Exception]]:
        """
        Ingest multiple (filename, bytes) pairs.
        Returns a dict of {filename: result_or_exception}.
        """
        results: Dict[str, Union[IngestedResult, Exception]] = {}
        for filename, data in files:
            try:
                results[filename] = self.ingest(filename, data)
            except Exception as exc:
                logger.error("Error ingesting '%s': %s", filename, exc)
                results[filename] = exc
        return results

    def get_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return only the structured (DataFrame) results."""
        return {k: v for k, v in self._store.items() if isinstance(v, pd.DataFrame)}

    def get_texts(self) -> Dict[str, str]:
        """Return only the unstructured (text / OCR) results."""
        return {k: v for k, v in self._store.items() if isinstance(v, str)}

    def get_all(self) -> SessionStore:
        """Return everything."""
        return dict(self._store)

    def list_files(self) -> List[str]:
        """List all ingested filenames."""
        return list(self._store.keys())

    def clear(self) -> None:
        """Remove all ingested files from the session."""
        self._store.clear()

    def summary(self) -> str:
        """Human-readable summary of all ingested files."""
        lines: List[str] = []
        for name, result in self._store.items():
            if isinstance(result, pd.DataFrame):
                lines.append(
                    f"📊 {name} — DataFrame  "
                    f"({result.shape[0]} rows × {result.shape[1]} cols)  "
                    f"columns: {list(result.columns)}"
                )
            else:
                word_count = len(result.split())
                lines.append(f"📄 {name} — Text  ({word_count} words)")
        return "\n".join(lines) if lines else "No files ingested yet."

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, suffix: str, data: bytes, filename: str) -> IngestedResult:
        if suffix == ".csv":
            return _read_csv(data, filename)
        elif suffix in (".xlsx", ".xls"):
            return _read_excel(data, filename)
        elif suffix == ".txt":
            return _read_txt(data)
        elif suffix == ".pdf":
            return _read_pdf(data, filename)
        elif suffix == ".docx":
            return _read_docx(data, filename)
        elif suffix in IMAGE_EXTS:
            return _read_image_via_vision(data, filename)
        else:
            raise ValueError(f"No handler for extension '{suffix}'.")
