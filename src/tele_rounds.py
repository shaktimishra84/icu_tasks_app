from __future__ import annotations

import re
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any

from src.extractors import ExtractionError, extract_text
from src.rmo_pdf import parse_combined_rmo_pdf, parse_combined_rmo_text

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except Exception:
    colors = None
    A4 = None
    landscape = None
    ParagraphStyle = None
    getSampleStyleSheet = None
    mm = None
    RLImage = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


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
        "status": status,
        "supports": supports,
        "diagnosis": diagnosis,
        "clinical_status": section2,
        "major_concern": major_concern,
        "rmo_recommendation": recommendation,
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
        raw = extract_text(filename, data)
        if not isinstance(raw, dict):
            raise ExtractionError("Unexpected DOCX extraction payload.")
        rows, warnings = _rows_from_docx_payload(filename, raw)
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

    headers = [
        "Bed",
        "Patient ID",
        "Acuity",
        "Support",
        "Major concern",
        "RMO recommendation",
        "New consultant orders",
    ]
    table_rows: list[list[Any]] = [[Paragraph(f"<b>{escape(h)}</b>", head) for h in headers]]

    for patient in patients:
        patient_key = str(patient.get("patient_key", ""))
        new_orders = _normalize(orders_by_key.get(patient_key, "")) or "Not documented"
        table_rows.append(
            [
                Paragraph(escape(_short(_normalize(patient.get("bed", "-")) or "-", 8)), cell),
                Paragraph(escape(_short(_normalize(patient.get("patient_id", "Not documented")), 16)), cell),
                Paragraph(escape(_short(_normalize(patient.get("status", "Not documented")), 11)), cell),
                Paragraph(escape(_short(_normalize(patient.get("supports", "None documented")), 28)), cell),
                Paragraph(escape(_short(_normalize(patient.get("major_concern", "Not documented")), 95)), cell),
                Paragraph(escape(_short(_normalize(patient.get("rmo_recommendation", "Not documented")), 95)), cell),
                Paragraph(escape(_short(new_orders, 95)), cell),
            ]
        )

    table = Table(
        table_rows,
        colWidths=[doc.width * 0.05, doc.width * 0.11, doc.width * 0.07, doc.width * 0.13, doc.width * 0.21, doc.width * 0.20, doc.width * 0.23],
        repeatRows=1,
    )
    table_style: list[tuple[Any, ...]] = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for idx, patient in enumerate(patients, start=1):
        status = _normalize(patient.get("status", "")).upper()
        if status == "CRITICAL":
            table_style.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor("#fef2f2")))
        elif patient.get("red_flag"):
            table_style.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor("#fffbeb")))
    table.setStyle(TableStyle(table_style))
    story.append(table)

    story.append(Spacer(1, 1.2 * mm))
    story.append(Paragraph("Prepared for tele-round handover and WhatsApp sharing.", sub))

    doc.build(story)
    return output_path.read_bytes(), output_path
