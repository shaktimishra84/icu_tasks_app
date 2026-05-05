from __future__ import annotations

import re
from datetime import date, datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

from src.extractors import ExtractionError, extract_text
from src.rmo_pdf import parse_combined_rmo_pdf, parse_combined_rmo_text
from src.rmo_pdf import _extract_bed as _rmo_extract_bed
from src.rmo_pdf import _extract_consultant_focus, _extract_patient_id
from src.rmo_pdf import _extract_red_flag_issue, _extract_status, _extract_supports
from src.rmo_pdf import _strip_datetime_prefix

try:
    from docx import Document
except Exception:
    Document = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except Exception:
    colors = None
    A4 = None
    landscape = None
    ParagraphStyle = None
    getSampleStyleSheet = None
    mm = None
    RLImage = None
    KeepTogether = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
SECTION_RE = re.compile(r"\bSECTION\s*(\d{1,2})\s*:", re.IGNORECASE)


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _clean_for_display(value: Any, *, default: str = "Not documented") -> str:
    text = _normalize(value).replace("■", " ").strip(" ;|,-")
    if not text:
        return default
    return text


def _bed_sort_value(value: Any) -> tuple[int, int | str]:
    text = _normalize(value)
    match = re.search(r"(\d+)", text)
    if match:
        return (0, int(match.group(1)))
    if text:
        return (1, text)
    return (2, "")


def _patient_key(bed: str, patient_id: str) -> str:
    pid = re.sub(r"[^a-z0-9]+", "", patient_id.lower())
    if pid:
        return f"pid:{pid}"
    bed_key = re.sub(r"[^a-z0-9]+", "", bed.lower())
    return f"bed:{bed_key}" if bed_key else ""


def _norm_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _table_cells(table: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [_normalize(cell.text.replace("\n", " ")) for cell in row.cells]
        if any(cells):
            rows.append(cells)
    return rows


def _table_text(rows: list[list[str]]) -> str:
    return " ".join(" ".join(cell for cell in row if cell) for row in rows)


def _detect_section_no(rows: list[list[str]]) -> int | None:
    text = _table_text(rows)
    match = SECTION_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _contains_header(rows: list[list[str]], *needles: str) -> bool:
    header_idx = _header_row_index(rows)
    if header_idx is None:
        header_text = _norm_header(_table_text(rows))
    else:
        header_text = _norm_header(" ".join(rows[header_idx]))
    return all(_norm_header(needle) in header_text for needle in needles)


def _is_section_title_row(cells: list[str]) -> bool:
    return SECTION_RE.search(" ".join(cells)) is not None


def _looks_like_header_row(cells: list[str]) -> bool:
    row_text = _norm_header(" ".join(cells))
    if not row_text or _is_section_title_row(cells):
        return False
    header_tokens = [
        "sl no",
        "slno",
        "uhid",
        "patient name",
        "bed no",
        "icu bed no",
        "category",
        "primary diagnosis",
        "gcs",
        "bp map",
        "spo2",
        "rr",
        "red flag present",
        "current status",
        "major concern",
        "rmo recommendation",
        "advice given",
    ]
    return any(token in row_text for token in header_tokens)


def _header_row_index(rows: list[list[str]]) -> int | None:
    for idx, cells in enumerate(rows):
        if _looks_like_header_row(cells):
            return idx
    return None


def _first_data_row(rows: list[list[str]]) -> list[str]:
    header_idx = _header_row_index(rows)
    candidates = rows[header_idx + 1 :] if header_idx is not None else rows
    for cells in candidates:
        text = _table_text([cells]).upper()
        if _is_section_title_row(cells) or _looks_like_header_row(cells):
            continue
        if any(cell.strip() for cell in cells):
            return cells
    return []


def _header_map(rows: list[list[str]]) -> dict[str, int]:
    header_idx = _header_row_index(rows)
    if header_idx is None:
        return {}
    headers = [_norm_header(cell) for cell in rows[header_idx]]
    return {header: idx for idx, header in enumerate(headers) if header}


def _value_for_header(rows: list[list[str]], aliases: list[str], default: str = "") -> str:
    mapping = _header_map(rows)
    data = _first_data_row(rows)
    if not data:
        return default
    for alias in aliases:
        alias_norm = _norm_header(alias)
        for header, idx in mapping.items():
            if alias_norm == header or alias_norm in header or header in alias_norm:
                if idx < len(data):
                    return _normalize(data[idx])
    return default


def _section1_from_table(rows: list[list[str]]) -> dict[str, str]:
    data = _first_data_row(rows)
    text = _table_text(rows)
    patient_id = _value_for_header(rows, ["UHID", "Patient ID"], default="")
    if not patient_id:
        patient_id = _extract_patient_id(text)

    bed = _value_for_header(rows, ["Bed No", "ICU Bed No", "Bed"], default="")
    bed_digits = re.search(r"\d+", bed)
    if bed_digits:
        bed = bed_digits.group(0)
    else:
        bed = _rmo_extract_bed(text)

    patient_name = _value_for_header(rows, ["Patient Name", "Name"], default="")
    category = _value_for_header(rows, ["Category", "ICU/HDU", "ICU HDU"], default="")
    diagnosis = _value_for_header(rows, ["Primary Diagnosis", "Diagnosis"], default="")
    if not diagnosis and data:
        diagnosis = " ".join(data[-2:]) if len(data) >= 2 else " ".join(data)

    return {
        "bed": _normalize(bed),
        "patient_id": _normalize(patient_id),
        "patient_name": _normalize(patient_name),
        "category": _normalize(category),
        "diagnosis": _normalize(diagnosis),
    }


def _section2_from_table(rows: list[list[str]]) -> dict[str, str]:
    data_row_text = _strip_datetime_prefix(" ".join(_first_data_row(rows)))
    gcs = _value_for_header(rows, ["GCS"], default="")
    bp = _value_for_header(rows, ["BP (MAP)", "BP", "MAP"], default="")
    spo2 = _value_for_header(rows, ["SpO2", "SPO2"], default="")
    rr = _value_for_header(rows, ["RR", "Respiratory Rate"], default="")
    return {
        "gcs": _normalize(gcs),
        "bp_map": _normalize(bp),
        "spo2": _normalize(spo2),
        "rr": _normalize(rr),
        "section2_status": _normalize(data_row_text),
    }


def _section10_from_table(rows: list[list[str]]) -> dict[str, str]:
    present = _value_for_header(rows, ["Red Flag Present", "Red Flag"], default="")
    data_row_text = _strip_datetime_prefix(" ".join(_first_data_row(rows)))
    red_flag_present = _normalize(present).upper() == "Y" or re.search(r"\bY\b", data_row_text, flags=re.IGNORECASE) is not None
    return {
        "red_flag_present": "Y" if red_flag_present else "N",
        "new_issues": _extract_red_flag_issue(data_row_text),
    }


def _section11_from_table(rows: list[list[str]]) -> dict[str, str]:
    data_row_text = _strip_datetime_prefix(" ".join(_first_data_row(rows)))
    current_status = _value_for_header(rows, ["Current Status", "Status"], default="")
    major_concern = _value_for_header(rows, ["Major Concern"], default="")
    rmo_recommendation = _value_for_header(rows, ["RMO Recommendation", "Recommendation"], default="")
    fallback_concern, fallback_recommendation = _extract_consultant_focus(data_row_text, "")
    return {
        "status": _normalize(current_status) or _extract_status(data_row_text, "", ""),
        "section11_summary": _normalize(data_row_text),
        "major_concern": _normalize(major_concern) or fallback_concern,
        "rmo_recommendation": _normalize(rmo_recommendation) or fallback_recommendation,
    }


def _section12_from_table(rows: list[list[str]]) -> dict[str, str]:
    advice = _value_for_header(rows, ["Advice Given", "Advice", "Orders"], default="")
    data_row_text = _strip_datetime_prefix(" ".join(_first_data_row(rows)))
    advice = _normalize(advice) or _normalize(data_row_text)
    return {
        "section12_orders": advice,
        "plan_next_12h": advice,
        "actions_done": advice,
    }


def parse_docx_patient_blocks(data: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    if Document is None:
        raise ExtractionError("DOCX support missing. Install python-docx.")

    document = Document(BytesIO(data))
    patients: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    warnings: list[str] = []

    for table in document.tables:
        rows = _table_cells(table)
        if not rows:
            continue
        text = _table_text(rows)
        section_no = _detect_section_no(rows)
        looks_section1 = section_no == 1 or "UHID" in text.upper()

        if looks_section1:
            if current is not None and (current.get("bed") or current.get("patient_id")):
                patients.append(current)
            current = _section1_from_table(rows)
            continue

        if current is None:
            continue

        if section_no == 2 or _contains_header(rows, "GCS"):
            current.update(_section2_from_table(rows))
        elif section_no == 10 or "RED FLAG" in text.upper():
            current.update(_section10_from_table(rows))
        elif section_no == 11 or "CURRENT STATUS" in text.upper() or "MAJOR CONCERN" in text.upper():
            current.update(_section11_from_table(rows))
        elif section_no == 12 or "ADVICE GIVEN" in text.upper():
            current.update(_section12_from_table(rows))
        elif section_no in {3, 4, 6, 9}:
            current[f"section{section_no}_row"] = _strip_datetime_prefix(" ".join(_first_data_row(rows)))

    if current is not None and (current.get("bed") or current.get("patient_id")):
        patients.append(current)

    for patient in patients:
        patient["supports"] = _extract_supports(
            patient.get("section3_row", ""),
            patient.get("section4_row", ""),
            patient.get("section6_row", ""),
            patient.get("section9_row", ""),
            patient.get("section12_orders", ""),
        )
        if not patient.get("status"):
            patient["status"] = _extract_status(
                patient.get("section11_summary", ""),
                patient.get("supports", ""),
                patient.get("new_issues", ""),
            )
        if not patient.get("pending"):
            patient["pending"] = patient.get("section9_row", "")

    if not patients:
        warnings.append("No patient blocks found using Section 1 to Section 13 DOCX parsing.")
    return patients, warnings


def _infer_date_shift_from_filename(filename: str) -> tuple[str, str]:
    lower_name = str(filename or "").lower()
    parsed_date: date | None = None
    patterns = [
        r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)",
        r"([0-3]\d)[-_]([01]\d)[-_](20\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower_name)
        if not match:
            continue
        g = match.groups()
        try:
            if len(g[0]) == 4:
                parsed_date = date(int(g[0]), int(g[1]), int(g[2]))
            else:
                parsed_date = date(int(g[2]), int(g[1]), int(g[0]))
            break
        except ValueError:
            continue

    if any(token in lower_name for token in ["9.30", "9:30", "evening", "pm", "night"]):
        shift = "Evening (9:30 PM)"
    else:
        shift = "Morning (7:30 AM)"

    return (parsed_date or date.today()).isoformat(), shift


def _derive_major_concern(row: dict[str, Any]) -> str:
    for key in ["major_concern", "section11_summary", "new_issues", "diagnosis"]:
        value = _normalize(row.get(key, ""))
        if value:
            return value
    return "Not documented"


def _derive_recommendation(row: dict[str, Any]) -> str:
    for key in ["rmo_recommendation", "section12_orders", "plan_next_12h", "actions_done"]:
        value = _normalize(row.get(key, ""))
        if value:
            return value
    return "Not documented"


def _to_tele_patient(row: dict[str, Any]) -> dict[str, Any]:
    bed = _normalize(row.get("bed", ""))
    patient_id = _normalize(row.get("patient_id", ""))
    status = _normalize(row.get("status", "")) or "Not documented"
    supports = _normalize(row.get("supports", "")) or "None documented"
    diagnosis = _clean_for_display(row.get("diagnosis", ""), default="Not documented")
    section2 = _clean_for_display(row.get("section2_status", ""), default="Not documented")
    major_concern = _clean_for_display(_derive_major_concern(row), default="Not documented")
    recommendation = _clean_for_display(_derive_recommendation(row), default="Not documented")
    red_flag_present = _normalize(row.get("red_flag_present", "")).upper() == "Y"
    red_flag_issue = _normalize(row.get("new_issues", ""))
    if red_flag_issue.upper() in {"NIL", "NA", "N/A", "NONE", "-"}:
        red_flag_issue = ""
    if red_flag_issue and not red_flag_present:
        red_flag_present = True

    return {
        "bed": bed,
        "patient_id": patient_id,
        "patient_name": _normalize(row.get("patient_name", "")),
        "category": _normalize(row.get("category", "")),
        "status": status,
        "supports": supports,
        "diagnosis": diagnosis,
        "clinical_status": section2,
        "gcs": _normalize(row.get("gcs", "")),
        "bp_map": _normalize(row.get("bp_map", "")),
        "spo2": _normalize(row.get("spo2", "")),
        "rr": _normalize(row.get("rr", "")),
        "major_concern": major_concern,
        "rmo_recommendation": recommendation,
        "previous_plan": _clean_for_display(row.get("section12_orders", ""), default="Not documented"),
        "red_flag": red_flag_present,
        "red_flag_issue": red_flag_issue,
        "source_pending": _normalize(row.get("pending", "")),
        "patient_key": _patient_key(bed, patient_id),
    }


def _rows_from_docx_payload(filename: str, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    raw_text = str(payload.get("raw_text", "") or "")
    parsed = parse_combined_rmo_text(raw_text)
    rows = parsed.get("table_rows", []) if isinstance(parsed, dict) else []
    if isinstance(rows, list) and rows:
        return rows, list(parsed.get("warnings", []) or [])

    table_rows = payload.get("table_rows", [])
    if isinstance(table_rows, list) and table_rows:
        warnings.append(
            "Section-wise RMO blocks were not detected in DOCX; using fallback table rows without section-level detail."
        )
        converted: list[dict[str, Any]] = []
        for row in table_rows:
            if not isinstance(row, dict):
                continue
            converted.append(
                {
                    "bed": _normalize(row.get("bed", "")),
                    "patient_id": _normalize(row.get("patient_id", "")),
                    "diagnosis": _normalize(row.get("diagnosis", "")),
                    "status": _normalize(row.get("status", "")),
                    "supports": _normalize(row.get("supports", "")),
                    "new_issues": _normalize(row.get("new_issues", "")),
                    "actions_done": _normalize(row.get("actions_done", "")),
                    "plan_next_12h": _normalize(row.get("plan_next_12h", "")),
                    "pending": _normalize(row.get("pending", "")),
                    "key_labs_imaging": _normalize(row.get("key_labs_imaging", "")),
                    "section2_status": _normalize(row.get("key_labs_imaging", "")),
                    "section11_summary": _normalize(row.get("new_issues", "")),
                    "section12_orders": _normalize(row.get("plan_next_12h", "")),
                    "major_concern": _normalize(row.get("new_issues", "")),
                    "rmo_recommendation": _normalize(row.get("plan_next_12h", "")),
                    "red_flag_present": "Y" if _normalize(row.get("new_issues", "")) else "N",
                }
            )
        return converted, warnings

    raise ExtractionError(f"No RMO rows parsed from DOCX `{filename}`.")


def process_icu_report(filename: str, data: bytes) -> dict[str, Any]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        payload = parse_combined_rmo_pdf(filename, data)
        rows = payload.get("table_rows", []) if isinstance(payload, dict) else []
        warnings = list(payload.get("warnings", []) or []) if isinstance(payload, dict) else []
    elif ext == ".docx":
        try:
            rows, warnings = parse_docx_patient_blocks(data)
        except Exception as error:  # noqa: BLE001 - malformed DOCX falls back to legacy extraction.
            rows = []
            warnings = [f"Patient-block DOCX parser skipped: {error}"]
        if not rows:
            raw = extract_text(filename, data)
            if not isinstance(raw, dict):
                raise ExtractionError("Unexpected DOCX extraction payload.")
            rows, fallback_warnings = _rows_from_docx_payload(filename, raw)
            warnings.extend(fallback_warnings)
    else:
        raise ExtractionError("Only PDF and DOCX are supported for tele-round workflow.")

    patients = [_to_tele_patient(row) for row in rows if isinstance(row, dict)]
    patients = [p for p in patients if p.get("bed") or p.get("patient_id")]
    patients.sort(key=lambda item: (_bed_sort_value(item.get("bed", "")), item.get("patient_id", "")))

    red_flags = [
        {
            "bed": p.get("bed", ""),
            "patient_id": p.get("patient_id", ""),
            "issue": p.get("red_flag_issue", "") or "Red flag marked Y",
        }
        for p in patients
        if p.get("red_flag")
    ]

    report_date, shift_label = _infer_date_shift_from_filename(filename)
    return {
        "filename": filename,
        "report_date": report_date,
        "shift_label": shift_label,
        "patients": patients,
        "red_flags": red_flags,
        "warnings": warnings,
    }


def _require_reportlab() -> None:
    if SimpleDocTemplate is None or landscape is None:
        raise RuntimeError("WhatsApp PDF export requires reportlab. Install with: pip install reportlab")


def _short(text: str, limit: int) -> str:
    clean = _normalize(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def generate_whatsapp_round_pdf(
    report: dict[str, Any],
    orders_by_key: dict[str, str],
    *,
    output_dir: Path = OUTPUT_DIR,
    logo_path: Path | None = None,
) -> tuple[bytes, Path]:
    _require_reportlab()

    patients = list(report.get("patients", []) or [])
    report_date = _normalize(report.get("report_date", "")) or date.today().isoformat()
    shift_label = _normalize(report.get("shift_label", "")) or "Morning (7:30 AM)"
    file_name = f"ICU_WhatsApp_Rounds_{report_date}_{'Morning' if 'Morning' in shift_label else 'Evening'}.pdf"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=8 * mm,
        rightMargin=8 * mm,
        topMargin=7 * mm,
        bottomMargin=7 * mm,
    )

    sheet = getSampleStyleSheet()
    title = ParagraphStyle(
        "wr_title",
        parent=sheet["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=16,
        textColor=colors.HexColor("#111111"),
        spaceAfter=3,
    )
    sub = ParagraphStyle(
        "wr_sub",
        parent=sheet["Normal"],
        fontName="Helvetica",
        fontSize=8.6,
        leading=10.2,
        textColor=colors.HexColor("#333333"),
        spaceAfter=4,
    )
    head = ParagraphStyle(
        "wr_head",
        parent=sheet["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=9.0,
        textColor=colors.HexColor("#111111"),
    )
    cell = ParagraphStyle(
        "wr_cell",
        parent=sheet["Normal"],
        fontName="Helvetica",
        fontSize=7.1,
        leading=8.5,
        textColor=colors.HexColor("#111111"),
    )
    card_header = ParagraphStyle(
        "wr_card_header",
        parent=sheet["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#111111"),
    )
    label = ParagraphStyle(
        "wr_label",
        parent=sheet["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=8.5,
        textColor=colors.HexColor("#374151"),
    )

    story: list[Any] = []

    logo = None
    if logo_path is not None and Path(logo_path).exists() and RLImage is not None:
        try:
            logo = RLImage(str(logo_path), width=22 * mm, height=10 * mm)
        except Exception:
            logo = None

    if logo is not None:
        header_table = Table(
            [[logo, Paragraph(f"<b>ICU Tele-Round Summary</b><br/>{escape(report_date)} - {escape(shift_label)}", title)]],
            colWidths=[24 * mm, doc.width - 24 * mm],
        )
        header_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(header_table)
    else:
        story.append(Paragraph(f"ICU Tele-Round Summary - {escape(report_date)} - {escape(shift_label)}", title))

    story.append(
        Paragraph(
            f"Current shift beds: {len(patients)} | New consultant orders included: {sum(1 for p in patients if _normalize(orders_by_key.get(str(p.get('patient_key', '')), '')))}",
            sub,
        )
    )

    high_priority = [patient for patient in patients if patient.get("red_flag")]
    if high_priority:
        story.append(Paragraph("High-priority red flags", head))
        flag_rows: list[list[Any]] = [[Paragraph("<b>Bed</b>", head), Paragraph("<b>Issue</b>", head)]]
        for patient in high_priority[:10]:
            flag_rows.append(
                [
                    Paragraph(escape(_short(_normalize(patient.get("bed", "-")) or "-", 8)), cell),
                    Paragraph(
                        escape(_short(_normalize(patient.get("red_flag_issue", "")) or "Red flag marked Y", 130)),
                        cell,
                    ),
                ]
            )
        flag_table = Table(flag_rows, colWidths=[doc.width * 0.12, doc.width * 0.88], repeatRows=1)
        flag_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#fca5a5")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]
            )
        )
        story.append(flag_table)
        story.append(Spacer(1, 2.0 * mm))

    for patient in patients:
        patient_key = str(patient.get("patient_key", ""))
        new_orders = _normalize(orders_by_key.get(patient_key, ""))
        order_text = new_orders or "____________________________________________________________"
        status = _normalize(patient.get("status", "Not documented")) or "Not documented"
        supports = _normalize(patient.get("supports", "None documented")) or "None documented"
        vitals = " | ".join(
            item
            for item in [
                f"GCS {patient.get('gcs')}" if _normalize(patient.get("gcs", "")) else "",
                f"BP {patient.get('bp_map')}" if _normalize(patient.get("bp_map", "")) else "",
                f"SpO2 {patient.get('spo2')}" if _normalize(patient.get("spo2", "")) else "",
                f"RR {patient.get('rr')}" if _normalize(patient.get("rr", "")) else "",
            ]
            if item
        )
        header_text = (
            f"Bed {escape(_short(_normalize(patient.get('bed', '-')) or '-', 8))} | "
            f"{escape(_short(_normalize(patient.get('patient_id', 'Not documented')), 18))} | "
            f"{escape(_short(status, 14))} | {escape(_short(supports, 42))}"
        )
        name_category = " | ".join(
            item
            for item in [
                _normalize(patient.get("patient_name", "")),
                _normalize(patient.get("category", "")),
            ]
            if item
        )
        if name_category:
            header_text += f"<br/><font color='#4b5563'>{escape(_short(name_category, 80))}</font>"

        rows = [
            [Paragraph(header_text, card_header), ""],
            [Paragraph("Diagnosis", label), Paragraph(escape(_short(_normalize(patient.get("diagnosis", "Not documented")), 115)), cell)],
            [Paragraph("Vitals / current status", label), Paragraph(escape(_short(vitals or _normalize(patient.get("clinical_status", "Not documented")), 135)), cell)],
            [Paragraph("Major concern", label), Paragraph(escape(_short(_normalize(patient.get("major_concern", "Not documented")), 135)), cell)],
            [Paragraph("RMO recommendation", label), Paragraph(escape(_short(_normalize(patient.get("rmo_recommendation", "Not documented")), 135)), cell)],
            [Paragraph("New Tele-Round Orders", label), Paragraph(escape(_short(order_text, 140)), cell)],
        ]
        card = Table(rows, colWidths=[doc.width * 0.24, doc.width * 0.76], hAlign="LEFT")
        background = colors.HexColor("#ffffff")
        border = colors.HexColor("#d1d5db")
        if _normalize(status).upper() == "CRITICAL":
            background = colors.HexColor("#fef2f2")
            border = colors.HexColor("#ef4444")
        elif patient.get("red_flag"):
            background = colors.HexColor("#fffbeb")
            border = colors.HexColor("#f59e0b")
        card.setStyle(
            TableStyle(
                [
                    ("SPAN", (0, 0), (-1, 0)),
                    ("BACKGROUND", (0, 0), (-1, -1), background),
                    ("BOX", (0, 0), (-1, -1), 0.7, border),
                    ("LINEBEFORE", (0, 0), (0, -1), 3.0, border),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story_item: Any = [card, Spacer(1, 1.8 * mm)]
        story.append(KeepTogether(story_item) if KeepTogether is not None else card)
        if KeepTogether is None:
            story.append(Spacer(1, 1.8 * mm))

    story.append(Spacer(1, 1.2 * mm))
    story.append(Paragraph("Prepared for tele-round handover and WhatsApp sharing.", sub))

    doc.build(story)
    return output_path.read_bytes(), output_path
