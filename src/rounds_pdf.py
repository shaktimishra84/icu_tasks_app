from __future__ import annotations

import re
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except Exception:
    colors = None
    A4 = None
    ParagraphStyle = None
    getSampleStyleSheet = None
    mm = None
    PageBreak = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
CARDS_PER_PAGE = 3

STATUS_ORDER = {
    "CRITICAL": 0,
    "SICK": 1,
    "SERIOUS": 2,
    "STABLE": 3,
    "DECEASED": 4,
}

STATUS_COLOR = {
    "CRITICAL": "#b91c1c",
    "SICK": "#c2410c",
    "SERIOUS": "#a16207",
    "STABLE": "#64748b",
    "DECEASED": "#374151",
}

GENERIC_PHRASES = [
    "reassess abcs",
    "reassess airway breathing circulation",
    "continue supportive care",
    "monitor closely",
    "clinical correlation",
]


def _require_reportlab() -> None:
    if SimpleDocTemplate is None:
        raise RuntimeError("PDF export requires `reportlab`. Install with: pip install reportlab")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalized_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority.lower(), 3)


def _status_group_for_pdf(row: dict[str, Any]) -> str:
    raw = _normalize_text(row.get("_status_group", "")).upper()
    if raw == "OTHER" or not raw:
        return "STABLE"
    if raw in STATUS_ORDER:
        return raw
    return "STABLE"


def _bed_sort_value(value: Any) -> tuple[int, int | str]:
    text = _normalize_text(value)
    match = re.search(r"\d+", text)
    if match:
        return (0, int(match.group(0)))
    if text:
        return (1, text)
    return (2, "")


def _status_sort_key(row: dict[str, Any]) -> tuple[int, tuple[int, int | str], str]:
    return (
        STATUS_ORDER.get(_status_group_for_pdf(row), 99),
        _bed_sort_value(row.get("Bed", "")),
        _normalize_text(row.get("Patient ID", "")),
    )


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _normalize_text(value)
        if not cleaned:
            continue
        key = _normalized_key(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _is_generic_phrase(text: str) -> bool:
    lower = _normalize_text(text).lower()
    return any(phrase in lower for phrase in GENERIC_PHRASES) if lower else True


def _shorten(text: str, limit: int) -> str:
    cleaned = _normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _parse_missing_items(text: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for line in str(text or "").splitlines():
        cleaned = re.sub(r"^[\-\*\u2022]+\s*", "", line).strip()
        if not cleaned:
            continue
        match = re.match(r"(?i)(high|medium|low)\s*:\s*(.+)", cleaned)
        if match:
            priority = match.group(1).title()
            name = _normalize_text(match.group(2))
        else:
            priority = "Medium"
            name = _normalize_text(cleaned)
        if name and not _is_generic_phrase(name):
            items.append((priority, name))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for priority, name in items:
        key = _normalized_key(name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((priority, name))
    return deduped


def _missing_category_items(row: dict[str, Any], field_name: str, max_items: int = 2) -> list[str]:
    parsed = _parse_missing_items(str(row.get(field_name, "")))
    parsed.sort(key=lambda item: (_priority_rank(item[0]), item[1].lower()))
    return [name for _priority, name in parsed][:max_items]


def _split_items(text: str, max_items: int = 2) -> list[str]:
    parts: list[str] = []
    for chunk in re.split(r"\n|;|\|", str(text or "")):
        cleaned = re.sub(r"^[\-\*\u2022]+\s*", "", chunk).strip()
        if not cleaned:
            continue
        if cleaned.lower() in {"-", "none", "nil", "na", "n/a"}:
            continue
        if _is_generic_phrase(cleaned):
            continue
        parts.append(cleaned)
    return _dedupe(parts)[:max_items]


def _pending_items(row: dict[str, Any], max_items: int = 2) -> list[str]:
    return _split_items(row.get("Pending (verbatim)", ""), max_items=max_items)


def _care_check_items(row: dict[str, Any], max_items: int = 2) -> list[str]:
    parsed = _parse_missing_items(str(row.get("Care checks (deterministic)", "")))
    parsed.sort(key=lambda item: (_priority_rank(item[0]), item[1].lower()))
    return [name for _priority, name in parsed][:max_items]


def _support_badges(row: dict[str, Any]) -> list[str]:
    badges: list[str] = []
    if row.get("_is_mv"):
        badges.append("MV")
    if row.get("_is_niv"):
        badges.append("NIV")
    if row.get("_is_vaso"):
        badges.append("VASO")
    if row.get("_is_rrt"):
        badges.append("RRT")
    if row.get("_is_o2"):
        badges.append("O2")
    return badges


def _build_pending_tracker_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    """
    Keep deterministic pending extraction available for tests/debug:
    sorted by bed ascending, then patient id.
    """
    tracker_rows: list[list[str]] = []
    for row in sorted(rows, key=lambda current: (_bed_sort_value(current.get("Bed", "")), _normalize_text(current.get("Patient ID", "")))):
        if _status_group_for_pdf(row) == "DECEASED":
            continue
        for pending in _pending_items(row, max_items=50):
            tracker_rows.append(
                [
                    _normalize_text(row.get("Bed", "")),
                    _normalize_text(row.get("Patient ID", "")),
                    pending,
                    "Pending",
                    "",
                ]
            )
    return tracker_rows


def _status_counts(rows: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    critical = sum(1 for row in rows if _status_group_for_pdf(row) == "CRITICAL")
    mv = sum(1 for row in rows if bool(row.get("_is_mv")))
    vaso = sum(1 for row in rows if bool(row.get("_is_vaso")))
    pending = sum(1 for row in rows if _pending_items(row, max_items=50))
    return critical, mv, vaso, pending


def _safe_line(text: str) -> str:
    return escape(_normalize_text(text))


def _card_table(row: dict[str, Any], width: float, styles: dict[str, Any]) -> Table:
    status = _status_group_for_pdf(row)
    bed = _normalize_text(row.get("Bed", ""))
    patient_id = _normalize_text(row.get("Patient ID", ""))
    diagnosis = _shorten(str(row.get("Diagnosis", "")), 90)
    system = _normalize_text(row.get("System tag", ""))
    matched = _normalize_text(row.get("Matched algorithms", ""))
    trend = _normalize_text(row.get("Round trend", ""))
    trend_reason = _shorten(_normalize_text(row.get("Deterioration reasons", "")), 50)
    key_labs = _shorten(_normalize_text(row.get("Key labs/imaging (1 line)", "")) or "-", 110)

    badges = _support_badges(row)
    badge_text = " ".join(f"[{badge}]" for badge in badges) if badges else ""
    trend_text = ""
    if trend:
        trend_text = f" | Trend: {trend}"
        if trend_reason:
            trend_text += f" ({trend_reason})"
    header_line = f"<b>Bed {escape(bed)} | {escape(patient_id)}</b> <font color='{STATUS_COLOR[status]}'><b>[{status}]</b></font>"
    if badge_text:
        header_line += f" {escape(badge_text)}"
    if trend_text:
        header_line += f" <font color='#64748b'>{escape(trend_text)}</font>"

    flowables: list[Any] = [
        Paragraph(header_line, styles["card_header"]),
        Paragraph(f"Diagnosis: {escape(diagnosis or '-')}", styles["line"]),
        Paragraph(
            f"<font color='#64748b'>System: {escape(system or '-')} | Matched: {escape(matched or '-')}</font>",
            styles["meta"],
        ),
    ]

    if status != "DECEASED":
        missing_tests = _missing_category_items(row, "Missing Tests", max_items=2)
        missing_imaging = _missing_category_items(row, "Missing Imaging", max_items=2)
        missing_consults = _missing_category_items(row, "Missing Consults", max_items=2)
        care_checks = _care_check_items(row, max_items=2)
        pending = _pending_items(row, max_items=2)

        if missing_tests or missing_imaging or missing_consults:
            flowables.append(Paragraph("A) Missing (High priority first)", styles["section"]))
            if missing_tests:
                flowables.append(Paragraph(f"Tests: {escape('; '.join(missing_tests))}", styles["line"]))
            if missing_imaging:
                flowables.append(Paragraph(f"Imaging: {escape('; '.join(missing_imaging))}", styles["line"]))
            if missing_consults:
                flowables.append(Paragraph(f"Consults: {escape('; '.join(missing_consults))}", styles["line"]))

        if care_checks:
            flowables.append(Paragraph("B) Care checks", styles["section"]))
            flowables.append(Paragraph(escape("; ".join(care_checks)), styles["line"]))

        if pending:
            flowables.append(Paragraph("C) Pending (verbatim)", styles["section"]))
            flowables.append(Paragraph(escape("; ".join(pending)), styles["line"]))

    flowables.append(Paragraph(f"<font color='#334155'>Key labs/imaging: {escape(key_labs)}</font>", styles["footer"]))

    strip_width = 4 * mm
    table = Table([[ "", flowables ]], colWidths=[strip_width, width - strip_width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(STATUS_COLOR[status])),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 6),
                ("RIGHTPADDING", (1, 0), (1, 0), 6),
                ("TOPPADDING", (1, 0), (1, 0), 5),
                ("BOTTOMPADDING", (1, 0), (1, 0), 5),
            ]
        )
    )
    return table


def generate_rounds_pdf(
    rows: list[dict[str, Any]],
    shift: str,
    run_date: date | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[bytes, Path]:
    _require_reportlab()

    normalized_shift = "Morning" if str(shift).strip().lower() == "morning" else "Evening"
    use_date = run_date or datetime.now().date()
    file_name = f"ICU_Rounds_{use_date.isoformat()}_{normalized_shift}.pdf"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=9 * mm,
        rightMargin=9 * mm,
        topMargin=9 * mm,
        bottomMargin=9 * mm,
    )

    style_sheet = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=style_sheet["Heading1"],
            fontSize=13,
            leading=15,
            spaceAfter=2,
        ),
        "meta_top": ParagraphStyle(
            "meta_top",
            parent=style_sheet["Normal"],
            fontSize=8.5,
            leading=10,
            textColor=colors.HexColor("#334155"),
            spaceAfter=5,
        ),
        "card_header": ParagraphStyle(
            "card_header",
            parent=style_sheet["Normal"],
            fontSize=8.5,
            leading=10,
            spaceAfter=2,
        ),
        "meta": ParagraphStyle(
            "meta",
            parent=style_sheet["Normal"],
            fontSize=7.2,
            leading=8.4,
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section",
            parent=style_sheet["Normal"],
            fontSize=7.8,
            leading=9.2,
            spaceBefore=1,
            spaceAfter=1,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0f172a"),
        ),
        "line": ParagraphStyle(
            "line",
            parent=style_sheet["Normal"],
            fontSize=7.6,
            leading=8.9,
            spaceAfter=1,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=style_sheet["Normal"],
            fontSize=7.2,
            leading=8.2,
            spaceBefore=2,
        ),
    }

    sorted_rows = sorted(rows, key=_status_sort_key)
    critical_count, mv_count, vaso_count, pending_count = _status_counts(sorted_rows)

    story: list[Any] = []
    story.append(Paragraph(f"ICU Rounds - {use_date.isoformat()} - {normalized_shift}", styles["title"]))
    story.append(
        Paragraph(
            f"Total beds: {len(sorted_rows)} | CRITICAL: {critical_count} | MV: {mv_count} | "
            f"Vasopressor: {vaso_count} | Pending reports: {pending_count}",
            styles["meta_top"],
        )
    )

    for index, row in enumerate(sorted_rows):
        if index and index % CARDS_PER_PAGE == 0:
            story.append(PageBreak())
            story.append(Paragraph(f"ICU Rounds - {use_date.isoformat()} - {normalized_shift}", styles["title"]))
            story.append(
                Paragraph(
                    f"Total beds: {len(sorted_rows)} | CRITICAL: {critical_count} | MV: {mv_count} | "
                    f"Vasopressor: {vaso_count} | Pending reports: {pending_count}",
                    styles["meta_top"],
                )
            )

        story.append(_card_table(row, doc.width, styles))
        if (index + 1) % CARDS_PER_PAGE != 0:
            story.append(Spacer(1, 3 * mm))

    doc.build(story)
    return output_path.read_bytes(), output_path
