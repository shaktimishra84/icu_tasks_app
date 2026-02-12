from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

try:
    from docx import Document
except Exception:  # pragma: no cover - optional dependency at runtime
    Document = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency at runtime
    Image = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency at runtime
    PdfReader = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency at runtime
    pytesseract = None

LOGGER = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when text extraction fails."""


DOCX_REQUIRED_FIELDS = [
    "bed",
    "patient_id",
    "diagnosis",
    "status",
    "supports",
    "new_issues",
    "actions_done",
    "plan_next_12h",
    "pending",
    "key_labs_imaging",
]

DOCX_HEADER_SEQUENCE = [
    "bed",
    "patient_id",
    "diagnosis",
    "status",
    "supports",
    "new_issues",
    "actions_done",
    "plan_next_12h",
    "pending",
    "key_labs_imaging",
]

DOCX_HEADER_ALIASES = {
    "bed": ["bed", "icu bed", "bed no", "bed number", "bed #", "bed id"],
    "patient_id": ["patient id", "patientid", "mrn", "uhid", "id", "patient no"],
    "diagnosis": ["diagnosis", "dx", "working diagnosis"],
    "status": ["status", "condition", "clinical status"],
    "supports": ["supports", "support", "organ supports", "devices", "device support"],
    "new_issues": ["new issues", "new issue", "issues", "new problems", "problem updates"],
    "actions_done": ["actions done", "done", "actions", "interventions done", "treatment done"],
    "plan_next_12h": ["plan next 12h", "plan next 12 h", "next 12h plan", "next 12 h plan", "plan 12h", "plan"],
    "pending": ["pending", "pending items", "pending tests", "awaiting", "to follow", "pending workup"],
    "key_labs_imaging": [
        "key labs imaging",
        "key labs/imaging",
        "labs/imaging",
        "key investigations",
        "investigations",
        "labs and imaging",
        "key labs",
    ],
}

DOCX_TEMPLATE_TOKENS = {
    "(1 line)",
    "(2 lines)",
    "(3 lines)",
    "(stable/watch/sick/crit.)",
    "(stable/watch/sick/crit)",
    "(stable watch sick crit.)",
    "(stable watch sick crit)",
    "(yes/no)",
}


def _normalize_header(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-z0-9 #]+", "", normalized)
    return normalized


def _canonical_docx_field(header: str) -> str | None:
    normalized = _normalize_header(header)
    if not normalized:
        return None
    for canonical_key, aliases in DOCX_HEADER_ALIASES.items():
        for alias in aliases:
            if normalized == _normalize_header(alias):
                return canonical_key
    return None


def _pad_row(cells: list[str], size: int) -> list[str]:
    return cells + [""] * max(0, size - len(cells))


def _extract_bed_digits(value: str) -> str:
    match = re.search(r"(\d+)", value)
    return match.group(1) if match else ""


def _is_template_value(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    if not normalized:
        return True
    if normalized in DOCX_TEMPLATE_TOKENS:
        return True
    if normalized.startswith("(") and normalized.endswith(")") and "line" in normalized:
        return True
    if "stable/watch/sick/crit" in normalized:
        return True
    return False


def _row_matches_expected_header(cells: list[str]) -> bool:
    if len(cells) < len(DOCX_HEADER_SEQUENCE):
        return False
    first_ten = _pad_row(cells, len(DOCX_HEADER_SEQUENCE))[: len(DOCX_HEADER_SEQUENCE)]
    mapped = [_canonical_docx_field(cell) for cell in first_ten]
    return mapped == DOCX_HEADER_SEQUENCE


def _extract_pdf(data: bytes) -> str:
    if PdfReader is None:
        raise ExtractionError("PDF support missing. Install pypdf.")
    reader = PdfReader(BytesIO(data))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts).strip()


def extract_pdf_pages(data: bytes) -> list[tuple[int, str]]:
    if PdfReader is None:
        raise ExtractionError("PDF support missing. Install pypdf.")
    reader = PdfReader(BytesIO(data))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((index, text))
    return pages


def _extract_docx(data: bytes) -> str:
    payload = _extract_docx_content(data)
    return payload["raw_text"]


def _extract_docx_content(data: bytes) -> dict[str, Any]:
    if Document is None:
        raise ExtractionError("DOCX support missing. Install python-docx.")
    document = Document(BytesIO(data))

    paragraph_parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    table_lines: list[str] = []

    table_count = len(document.tables)
    LOGGER.info("DOCX tables detected: %d", table_count)

    parsed_rows: list[dict[str, str]] = []
    debug_raw_rows: list[list[str]] = []

    for table in document.tables:
        rows_as_cells: list[list[str]] = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows_as_cells.append(cells)
            if any(cells):
                table_lines.append(" | ".join(cells))
                if len(debug_raw_rows) < 200:
                    debug_raw_rows.append(cells)

        if not rows_as_cells:
            continue

        header_found = False
        current_record: dict[str, str] | None = None

        for cells in rows_as_cells:
            padded = _pad_row(cells, len(DOCX_HEADER_SEQUENCE))
            first_ten = padded[: len(DOCX_HEADER_SEQUENCE)]

            if _row_matches_expected_header(first_ten):
                header_found = True
                continue

            if not header_found:
                continue

            if all(_is_template_value(value) for value in first_ten):
                continue

            bed_digits = _extract_bed_digits(first_ten[0])
            if bed_digits:
                if current_record is not None and current_record.get("bed"):
                    parsed_rows.append(current_record)

                current_record = {field: "" for field in DOCX_REQUIRED_FIELDS}
                current_record["bed"] = bed_digits
                for index, field in enumerate(DOCX_HEADER_SEQUENCE[1:], start=1):
                    value = first_ten[index].strip()
                    if value and not _is_template_value(value):
                        current_record[field] = value
                continue

            if current_record is None:
                continue

            for index, field in enumerate(DOCX_HEADER_SEQUENCE[1:], start=1):
                value = first_ten[index].strip()
                if not value or _is_template_value(value):
                    continue
                if current_record[field]:
                    current_record[field] = f"{current_record[field]} | {value}"
                else:
                    current_record[field] = value

        if current_record is not None and current_record.get("bed"):
            parsed_rows.append(current_record)

    if paragraph_parts and table_lines:
        raw_text = "\n".join(paragraph_parts + [""] + table_lines).strip()
    elif paragraph_parts:
        raw_text = "\n".join(paragraph_parts).strip()
    else:
        raw_text = "\n".join(table_lines).strip()

    LOGGER.info("DOCX table rows parsed: %d", len(parsed_rows))
    return {
        "raw_text": raw_text,
        "table_rows": parsed_rows,
        "debug_raw_rows": debug_raw_rows[:50],
    }


def extract_docx_table_rows(data: bytes) -> list[dict[str, str]]:
    payload = _extract_docx_content(data)
    rows = payload.get("table_rows", [])
    return rows if isinstance(rows, list) else []


def _extract_text(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore").strip()


def _extract_image(data: bytes, image_ocr_fn: Callable[[bytes], str] | None = None) -> str:
    if Image is None:
        raise ExtractionError("Image support missing. Install Pillow.")
    image = Image.open(BytesIO(data))

    text = ""
    if pytesseract is not None:
        text = (pytesseract.image_to_string(image) or "").strip()

    if not text and image_ocr_fn is not None:
        text = (image_ocr_fn(data) or "").strip()

    if not text:
        raise ExtractionError(
            "Image OCR could not run. Install Tesseract locally or provide an OCR callback."
        )
    return text


def extract_text(
    filename: str,
    data: bytes,
    image_ocr_fn: Callable[[bytes], str] | None = None,
) -> str | dict[str, Any]:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text = _extract_pdf(data)
    elif ext == ".docx":
        text = _extract_docx_content(data)
    elif ext in {".txt", ".md"}:
        text = _extract_text(data)
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        text = _extract_image(data, image_ocr_fn=image_ocr_fn)
    elif ext == ".doc":
        raise ExtractionError("Legacy .doc files are not supported. Convert to .docx first.")
    else:
        raise ExtractionError(f"Unsupported file type: {ext or '(no extension)'}")

    if isinstance(text, dict):
        raw_text = text.get("raw_text", "")
        table_rows = text.get("table_rows", [])
        if not raw_text and not table_rows:
            raise ExtractionError("No text could be extracted from this file.")
    elif not text:
        raise ExtractionError("No text could be extracted from this file.")
    return text
