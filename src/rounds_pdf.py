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


def _bed_sort_key_for_pdf(row: dict[str, Any]) -> tuple[tuple[int, int | str], str]:
    return (
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
    raw_text = str(text or "").replace("\r", "\n")
    raw_text = re.sub(r"(?:(?<=^)|(?<=\s))\d+[.)](?=\s*[A-Za-z])", "\n", raw_text)
    for chunk in re.split(r"\n|;|\|", raw_text):
        cleaned = re.sub(r"^[\-\*\u2022]+\s*", "", chunk).strip()
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned).strip()
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


def _source_highlights(row: dict[str, Any], max_items: int = 4) -> list[str]:
    source_text = " | ".join(
        [
            str(row.get("_raw_plan_next_12h", "")),
            str(row.get("_raw_actions_done", "")),
            str(row.get("_raw_new_issues", "")),
        ]
    )
    return _split_items(source_text, max_items=max_items)


def _build_pending_tracker_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    """
    Keep deterministic pending extraction available for tests/debug:
    sorted by bed ascending, then patient id.
    """
    tracker_rows: list[list[str]] = []
    for row in sorted(rows, key=_bed_sort_key_for_pdf):
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


PROCEDURE_HINTS = [
    "ugie",
    "endoscopy",
    "cect",
    "ct ",
    "ctpa",
    "mri",
    "hrct",
    "echo",
    "tracheostomy",
    "debridement",
    "bronchoscopy",
    "line",
]


def _is_new_admission(row: dict[str, Any]) -> bool:
    trend = _normalize_text(row.get("Round trend", "")).upper()
    if trend == "NEW ADMISSION":
        return True
    all_text = " ".join(
        [
            str(row.get("Diagnosis", "")),
            str(row.get("_raw_new_issues", "")),
            str(row.get("_raw_actions_done", "")),
            str(row.get("_raw_plan_next_12h", "")),
        ]
    ).lower()
    return "new admission" in all_text


def _is_procedure_case(row: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(row.get("_raw_plan_next_12h", "")),
            str(row.get("Pending (verbatim)", "")),
            str(row.get("_raw_actions_done", "")),
        ]
    ).lower()
    return any(hint in text for hint in PROCEDURE_HINTS)


def _is_closed_case(row: dict[str, Any]) -> bool:
    if _status_group_for_pdf(row) == "DECEASED":
        return True
    flags = [str(flag).lower() for flag in (row.get("_flags", []) or [])]
    if any("dama" in flag for flag in flags):
        return True
    transfer_text = " ".join(
        [
            str(row.get("Status", "")),
            str(row.get("Diagnosis", "")),
            str(row.get("_raw_plan_next_12h", "")),
        ]
    ).lower()
    return any(token in transfer_text for token in ["for discharge", "for transfer", "step down", "shift to ward"])


def _triage_rank(row: dict[str, Any]) -> int:
    status = _status_group_for_pdf(row)
    if _is_closed_case(row):
        return 5
    if status == "CRITICAL" or bool(row.get("_is_mv")) or bool(row.get("_is_vaso")):
        return 0
    if _is_new_admission(row):
        return 1
    if str(row.get("Deterioration since last round", "")).strip().upper() == "YES":
        return 2
    if _is_procedure_case(row):
        return 3
    return 4


def _triage_bucket(row: dict[str, Any]) -> str:
    rank = _triage_rank(row)
    if rank == 5:
        return "CLOSED"
    if rank <= 2:
        return "RED"
    if rank == 3:
        return "AMBER"
    return "GREEN"


def _triage_sort_key(row: dict[str, Any]) -> tuple[int, int, tuple[int, int | str], str]:
    status_order = {"CRITICAL": 0, "SICK": 1, "SERIOUS": 2, "STABLE": 3, "DECEASED": 4}
    status = _status_group_for_pdf(row)
    return (
        _triage_rank(row),
        status_order.get(status, 9),
        _bed_sort_value(row.get("Bed", "")),
        _normalize_text(row.get("Patient ID", "")),
    )


def _parse_priority_lines(text: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in str(text or "").splitlines():
        cleaned = re.sub(r"^[\-\*\u2022]+\s*", "", line).strip()
        if not cleaned:
            continue
        match = re.match(r"(?i)(high|medium|low)\s*:\s*(.+)", cleaned)
        if match:
            priority = match.group(1).title()
            item = _normalize_text(match.group(2))
        else:
            priority = "Medium"
            item = _normalize_text(cleaned)
        if item and not _is_generic_phrase(item):
            out.append((priority, item))
    out.sort(key=lambda item: (_priority_rank(item[0]), item[1].lower()))
    return out


def _collect_now(row: dict[str, Any], max_items: int = 2) -> list[str]:
    items: list[str] = []
    if _is_new_admission(row):
        items.append("New admission")
    if str(row.get("Deterioration since last round", "")).strip().upper() == "YES":
        reason = _normalize_text(row.get("Deterioration reasons", ""))
        items.append(f"Deteriorated: {reason}" if reason else "Deteriorated since previous round")
    items.extend(_split_items(str(row.get("_raw_new_issues", "")), max_items=max_items))
    if not items:
        trend = _normalize_text(row.get("Round trend", ""))
        if trend:
            items.append(f"Trend: {trend}")
    return _dedupe(items)[:max_items]


def _collect_today(row: dict[str, Any], max_items: int = 3) -> list[str]:
    plan_items = _split_items(str(row.get("_raw_plan_next_12h", "")), max_items=12)
    candidate: list[str] = [item for item in plan_items if not _is_generic_phrase(item)]
    for field_name in ["Missing Tests", "Missing Imaging", "Missing Consults", "Care checks (deterministic)"]:
        for priority, item in _parse_priority_lines(row.get(field_name, "")):
            candidate.append(f"{priority}: {item}")

    pending_keys = {_normalized_key(item) for item in _pending_items(row, max_items=30)}
    clean: list[str] = []
    for item in _dedupe(candidate):
        key = _normalized_key(item)
        key_wo = _normalized_key(re.sub(r"^(high|medium|low)\s*:\s*", "", item, flags=re.IGNORECASE))
        if key in pending_keys or key_wo in pending_keys:
            continue
        clean.append(item)
    return clean[:max_items]


def _collect_watch(row: dict[str, Any], max_items: int = 2) -> list[str]:
    watch: list[str] = []
    watch.extend(_split_items(str(row.get("Deterioration reasons", "")), max_items=max_items))
    flags = [str(flag).strip() for flag in (row.get("_flags", []) or []) if str(flag).strip()]
    watch.extend(flags)
    for issue in _split_items(str(row.get("_raw_new_issues", "")), max_items=6):
        if any(token in issue.lower() for token in ["shock", "hypotension", "peak pressure", "decline", "seizure"]):
            watch.append(issue)
    labs = _split_items(str(row.get("Key labs/imaging (1 line)", "")), max_items=6)
    for lab in labs:
        if any(token in lab.lower() for token in ["lactate", "creat", "potassium", "sodium", "abg", "platelet"]):
            watch.append(lab)
    return _dedupe(watch)[:max_items]


def _support_text(row: dict[str, Any]) -> str:
    supports: list[str] = []
    if bool(row.get("_is_mv")):
        supports.append("MV")
    if bool(row.get("_is_niv")):
        supports.append("NIV")
    if bool(row.get("_is_vaso")):
        supports.append("Vasopressor")
    if bool(row.get("_is_rrt")):
        supports.append("Dialysis")
    return " / ".join(supports) if supports else "-"


def _acuity_label(row: dict[str, Any]) -> str:
    bucket = _triage_bucket(row)
    if bucket == "RED":
        return "Red"
    if bucket == "AMBER":
        return "Amber"
    if bucket == "GREEN":
        return "Green"
    return "Grey"


def _change_line(row: dict[str, Any]) -> str:
    if _status_group_for_pdf(row) == "DECEASED":
        return "Declared deceased"
    if _is_new_admission(row):
        return "New admission"
    if str(row.get("Deterioration since last round", "")).strip().upper() == "YES":
        reason = _normalize_text(row.get("Deterioration reasons", ""))
        return f"Deteriorated: {reason}" if reason else "Deteriorated"
    trend = _normalize_text(row.get("Round trend", ""))
    if trend and trend.upper() not in {"STABLE", "NEW ADMISSION"}:
        return trend
    now = _collect_now(row, max_items=1)
    return now[0] if now else "No major change"


def _must_do_line(row: dict[str, Any]) -> str:
    if _status_group_for_pdf(row) == "DECEASED":
        return "-"
    tasks = _collect_today(row, max_items=2)
    return " | ".join(tasks) if tasks else "-"


def _pending_line(row: dict[str, Any]) -> str:
    pending = _pending_items(row, max_items=2)
    return " | ".join(pending) if pending else "-"


def _abnormal_lab_item(item: str) -> bool:
    text = _normalize_text(item)
    if not text:
        return False
    lower = text.lower()
    if any(token in lower for token in ["high", "low", "critical", "elevated", "abnormal", "pending", "report"]):
        return True

    checks: list[tuple[re.Pattern[str], float, float]] = [
        (re.compile(r"\b(?:k|potassium)\s*[-:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE), 3.5, 5.3),
        (re.compile(r"\b(?:na|sodium)\s*[-:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE), 130.0, 150.0),
        (re.compile(r"\b(?:creat(?:inine)?)\s*[-:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE), 0.0, 1.5),
        (re.compile(r"\blactate\s*[-:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE), 0.0, 2.0),
        (re.compile(r"\b(?:tlc|wbc)\s*[-:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE), 4.0, 12.0),
    ]
    for pattern, min_ok, max_ok in checks:
        match = pattern.search(text)
        if not match:
            continue
        value = float(match.group(1))
        return value < min_ok or value > max_ok

    return any(token in lower for token in ["abg", "ct", "mri", "ctpa", "hrct", "echo"])


def _key_labs_top3(row: dict[str, Any]) -> str:
    raw = str(row.get("Key labs/imaging (1 line)", "")).strip()
    if not raw:
        return "-"
    parts = _split_items(raw, max_items=12)
    abnormal = [item for item in parts if _abnormal_lab_item(item)]
    if not abnormal:
        return "-"
    return " | ".join(abnormal[:3])


def _escalation_line(row: dict[str, Any]) -> str:
    if _status_group_for_pdf(row) == "DECEASED":
        return "-"
    watch = _collect_watch(row, max_items=1)
    return watch[0] if watch else "-"


def _card_block(row: dict[str, Any], width: float, styles: dict[str, Any]) -> Table:
    status = _status_group_for_pdf(row)
    bucket = _triage_bucket(row)
    bed = _normalize_text(row.get("Bed", ""))
    patient_id = _normalize_text(row.get("Patient ID", ""))
    diagnosis = _shorten(str(row.get("Diagnosis", "")) or "-", 105)
    acuity = _acuity_label(row)
    supports = _support_text(row)
    change = _shorten(_change_line(row), 95)
    must_do = _shorten(_must_do_line(row), 100)
    pending = _shorten(_pending_line(row), 100)
    key_labs = _shorten(_key_labs_top3(row), 75)
    escalation = _shorten(_escalation_line(row), 60)

    lines = [
        Paragraph(
            f"<b>Bed {escape(bed)} | {escape(patient_id)}</b> "
            f"<font color='{STATUS_COLOR.get(status, '#334155')}'><b>[{escape(status)}]</b></font>",
            styles["card_head"],
        ),
        Paragraph(f"<b>Acuity / Supports:</b> {escape(acuity)} / {escape(supports)}", styles["card_line"]),
        Paragraph(f"<b>Diagnosis:</b> {escape(diagnosis)}", styles["card_line"]),
        Paragraph(f"<b>Change since last round:</b> {escape(change)}", styles["card_line"]),
        Paragraph(f"<b>Must do before evening:</b> {escape(must_do)}", styles["card_line"]),
        Paragraph(f"<b>Pending:</b> {escape(pending)}", styles["card_line"]),
        Paragraph(f"<b>Key labs (3) / Escalation:</b> {escape(key_labs)} / {escape(escalation)}", styles["card_line"]),
    ]

    strip_color = {
        "RED": "#ef4444",
        "AMBER": "#f59e0b",
        "GREEN": "#16a34a",
        "CLOSED": "#6b7280",
    }.get(bucket, "#64748b")
    fill_color = {
        "RED": "#fff1f2",
        "AMBER": "#fff7ed",
        "GREEN": "#f7fee7",
        "CLOSED": "#f3f4f6",
    }.get(bucket, "#ffffff")

    strip_width = 3.6 * mm
    card = Table([["", lines]], colWidths=[strip_width, width - strip_width])
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(strip_color)),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(fill_color)),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 5),
                ("RIGHTPADDING", (1, 0), (1, 0), 5),
                ("TOPPADDING", (1, 0), (1, 0), 4),
                ("BOTTOMPADDING", (1, 0), (1, 0), 4),
            ]
        )
    )
    return card


def _dashboard_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "critical": sum(1 for row in rows if _status_group_for_pdf(row) == "CRITICAL"),
        "mv": sum(1 for row in rows if bool(row.get("_is_mv"))),
        "vaso": sum(1 for row in rows if bool(row.get("_is_vaso"))),
        "new_admissions": sum(1 for row in rows if _is_new_admission(row)),
        "deteriorated": sum(
            1 for row in rows if str(row.get("Deterioration since last round", "")).strip().upper() == "YES"
        ),
        "pending_reports": sum(1 for row in rows if _pending_items(row, max_items=30)),
        "procedures": sum(1 for row in rows if _is_procedure_case(row)),
        "deaths": sum(1 for row in rows if _status_group_for_pdf(row) == "DECEASED"),
    }


def _comparison_table_rows(changes: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for change in sorted(changes, key=lambda item: (_bed_sort_value(item.get("bed", "")), str(item.get("patient_id", "")))):
        supports_added = list(change.get("supports_added", []) or [])
        supports_removed = list(change.get("supports_removed", []) or [])
        supports_parts: list[str] = []
        if supports_added:
            supports_parts.append("+" + ", ".join(supports_added))
        if supports_removed:
            supports_parts.append("-" + ", ".join(supports_removed))
        supports_delta = " ; ".join(supports_parts) if supports_parts else "-"

        pending_new = list(change.get("pending_new", []) or [])
        pending_resolved = list(change.get("pending_resolved", []) or [])
        pending_parts: list[str] = []
        if pending_new:
            pending_parts.append("new: " + " | ".join(pending_new[:2]))
        if pending_resolved:
            pending_parts.append("resolved: " + " | ".join(pending_resolved[:2]))
        pending_delta = " ; ".join(pending_parts) if pending_parts else "-"

        previous_status = _normalize_text(change.get("previous_status_group", ""))
        current_status = _normalize_text(change.get("current_status_group", "")) or "-"
        status_change = f"{previous_status} -> {current_status}" if previous_status else f"NEW -> {current_status}"

        summary_lines = [str(line).strip() for line in list(change.get("summary_lines", [])) if str(line).strip()]
        key_change = " ; ".join(summary_lines[:2]) if summary_lines else "-"

        rows.append(
            [
                _normalize_text(change.get("bed", "")),
                _normalize_text(change.get("patient_id", "")),
                _normalize_text(change.get("trend", "")),
                status_change,
                supports_delta,
                pending_delta,
                key_change,
            ]
        )
    return rows


def generate_rounds_pdf(
    rows: list[dict[str, Any]],
    shift: str,
    detail_level: str = "max",
    run_date: date | None = None,
    output_dir: Path = OUTPUT_DIR,
    comparison_changes: list[dict[str, Any]] | None = None,
    comparison_older_label: str = "",
    comparison_newer_label: str = "",
) -> tuple[bytes, Path]:
    _require_reportlab()
    simple_mode = str(detail_level).strip().lower() in {"simple", "max", "default"}

    normalized_shift = "Morning" if str(shift).strip().lower() == "morning" else "Evening"
    use_date = run_date or datetime.now().date()
    file_name = f"ICU_Rounds_{use_date.isoformat()}_{normalized_shift}.pdf"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=6 * mm,
        rightMargin=6 * mm,
        topMargin=7 * mm,
        bottomMargin=7 * mm,
    )

    style_sheet = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=style_sheet["Heading1"],
            fontSize=14 if simple_mode else 13,
            leading=16 if simple_mode else 15,
            spaceAfter=4,
        ),
        "subhead": ParagraphStyle(
            "subhead",
            parent=style_sheet["Normal"],
            fontSize=8.7 if simple_mode else 8.0,
            leading=10.2 if simple_mode else 9.4,
            textColor=colors.HexColor("#334155"),
            spaceAfter=4,
        ),
        "section": ParagraphStyle(
            "section",
            parent=style_sheet["Normal"],
            fontSize=9,
            leading=10.2,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=2,
        ),
        "card_head": ParagraphStyle(
            "card_head",
            parent=style_sheet["Normal"],
            fontSize=8.6 if simple_mode else 7.8,
            leading=9.8 if simple_mode else 8.9,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0f172a"),
        ),
        "card_line": ParagraphStyle(
            "card_line",
            parent=style_sheet["Normal"],
            fontSize=7.4 if simple_mode else 6.8,
            leading=8.6 if simple_mode else 7.8,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=0.4,
        ),
        "cmp_head": ParagraphStyle(
            "cmp_head",
            parent=style_sheet["Normal"],
            fontSize=7.3 if simple_mode else 6.8,
            leading=8.4 if simple_mode else 7.8,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0f172a"),
        ),
        "cmp_cell": ParagraphStyle(
            "cmp_cell",
            parent=style_sheet["Normal"],
            fontSize=6.9 if simple_mode else 6.4,
            leading=7.8 if simple_mode else 7.2,
            textColor=colors.HexColor("#0f172a"),
        ),
    }

    sorted_rows = sorted(rows, key=_bed_sort_key_for_pdf)
    counts = _dashboard_counts(sorted_rows)
    bucket_counts = {
        "red": sum(1 for row in sorted_rows if _triage_bucket(row) == "RED"),
        "amber": sum(1 for row in sorted_rows if _triage_bucket(row) == "AMBER"),
        "green": sum(1 for row in sorted_rows if _triage_bucket(row) == "GREEN"),
    }

    story: list[Any] = []
    story.append(Paragraph(f"ICU Rounds - {use_date.isoformat()} - {normalized_shift}", styles["title"]))
    story.append(
        Paragraph(
            (
                f"Beds {counts['total']} | Critical {counts['critical']} | MV {counts['mv']} | "
                f"Vaso {counts['vaso']} | Red {bucket_counts['red']} | Amber {bucket_counts['amber']} | "
                f"Green {bucket_counts['green']} | Pending {counts['pending_reports']}"
            ),
            styles["subhead"],
        )
    )
    story.append(
        Paragraph(
            "Order: bed number ascending",
            styles["subhead"],
        )
    )

    for idx, row in enumerate(sorted_rows):
        story.append(_card_block(row, doc.width, styles))
        if idx < len(sorted_rows) - 1:
            story.append(Spacer(1, 1.6 * mm))

    comparison_rows = _comparison_table_rows(comparison_changes or [])
    if comparison_rows:
        story.append(PageBreak())
        story.append(Paragraph("Round Comparison Table", styles["title"]))
        if _normalize_text(comparison_older_label) and _normalize_text(comparison_newer_label):
            story.append(
                Paragraph(
                    f"{escape(_normalize_text(comparison_older_label))} -> {escape(_normalize_text(comparison_newer_label))}",
                    styles["subhead"],
                )
            )
        else:
            story.append(Paragraph("Bed-wise delta overview", styles["subhead"]))

        headers = [
            "Bed",
            "Patient ID",
            "Trend",
            "Status change",
            "Supports delta",
            "Pending delta",
            "Key change",
        ]
        table_data: list[list[Any]] = [
            [Paragraph(f"<b>{escape(col)}</b>", styles["cmp_head"]) for col in headers]
        ]
        for row in comparison_rows:
            table_data.append([Paragraph(escape(_shorten(cell, 120)), styles["cmp_cell"]) for cell in row])

        comparison_table = Table(
            table_data,
            colWidths=[
                doc.width * 0.06,
                doc.width * 0.11,
                doc.width * 0.08,
                doc.width * 0.16,
                doc.width * 0.18,
                doc.width * 0.20,
                doc.width * 0.21,
            ],
            repeatRows=1,
        )
        style_ops: list[tuple[Any, ...]] = [
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        for row_index, row in enumerate(comparison_rows, start=1):
            trend = _normalize_text(row[2]).upper()
            if trend == "DETERIORATED":
                style_ops.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#fee2e2")))
            elif trend == "IMPROVED":
                style_ops.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#dcfce7")))
            elif trend == "NEW ADMISSION":
                style_ops.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#dbeafe")))
        comparison_table.setStyle(TableStyle(style_ops))
        story.append(comparison_table)

    doc.build(story)
    return output_path.read_bytes(), output_path
