from __future__ import annotations

import re
from typing import Any

from .extractors import ExtractionError, extract_pdf_pages

RMO_FIELDS = [
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

DATE_ROW_RE = re.compile(r"^\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", re.IGNORECASE)
DATE_TOKEN_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", re.IGNORECASE)
TIME_TOKEN_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:AM|PM|M|N)?\b", re.IGNORECASE)
MEDICAL_DIAG_KEYWORDS = [
    "CKD",
    "AKI",
    "COPD",
    "PNEUMONIA",
    "SEPSIS",
    "SEPTIC",
    "SHOCK",
    "CVA",
    "STROKE",
    "MI",
    "ENCEPHALOPATHY",
    "PTB",
    "LIVER",
    "ABSCESS",
    "PAROTITIS",
    "UGIB",
    "VARICEAL",
    "ARDS",
    "MALIGNANCY",
    "T2DM",
    "HTN",
    "UREMIA",
]
STATUS_WORDS = ["CRITICAL", "STABLE", "GUARDED", "SICK", "SERIOUS"]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _split_blocks(full_text: str) -> list[str]:
    parts = re.split(r"ICU RMO REPORTING FORMAT", full_text, flags=re.IGNORECASE)
    blocks = [part for part in parts[1:] if "SECTION 1" in part and "SECTION 2" in part]
    if blocks:
        return blocks
    if "SECTION 1" in full_text and "SECTION 2" in full_text:
        return [full_text]
    return []


def _extract_section(block: str, section_no: int) -> str:
    if section_no >= 13:
        pattern = rf"SECTION\s*{section_no}\s*:\s*[\s\S]*"
    else:
        pattern = rf"SECTION\s*{section_no}\s*:\s*[\s\S]*?(?=SECTION\s*{section_no + 1}\s*:)"
    match = re.search(pattern, block, flags=re.IGNORECASE)
    return match.group(0) if match else ""


def _extract_first_row_block(section_text: str) -> str:
    if not section_text:
        return ""
    lines = [line.rstrip() for line in section_text.splitlines()]
    start_index = -1
    for idx, line in enumerate(lines):
        if DATE_ROW_RE.match(line):
            start_index = idx
            break
    if start_index < 0:
        return ""

    row_lines = [lines[start_index].strip()]
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if DATE_ROW_RE.match(stripped):
            break
        if re.search(r"SECTION\s*\d+\s*:", stripped, flags=re.IGNORECASE):
            break
        row_lines.append(stripped)
    return _normalize_space(" ".join(row_lines))


def _split_columns(row_block: str) -> list[str]:
    if not row_block:
        return []
    lines = [line for line in row_block.split("\n") if line.strip()]
    if not lines:
        return []
    first = lines[0]
    columns = [col.strip() for col in re.split(r"\s{2,}", first.strip()) if col.strip()]
    if not columns:
        columns = [first.strip()]
    for extra in lines[1:]:
        extra_text = extra.strip()
        if not extra_text:
            continue
        if columns:
            columns[-1] = _normalize_space(f"{columns[-1]} {extra_text}")
        else:
            columns.append(extra_text)
    return columns


def _strip_datetime_prefix(text: str) -> str:
    value = _normalize_space(text)
    value = re.sub(
        r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?:\s+(?:AT\s+)?)?(?:\d{1,2}[:.]\d{2}\s*(?:AM|PM|M|N)?)?",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return _normalize_space(value)


def _extract_patient_id(section1_text: str) -> str:
    match = re.search(r"\b20\d{10}\b", section1_text)
    if match:
        return match.group(0)
    fallback = re.search(r"\b\d{10,14}\b", section1_text)
    return fallback.group(0) if fallback else ""


def _extract_bed(section1_text: str) -> str:
    after_age = re.search(
        r"\b\d{1,3}\s*(?:YRS|YEARS)?\s*/\s*(?:MALE|FEMALE|M|F)\b",
        section1_text,
        flags=re.IGNORECASE,
    )
    if after_age:
        tail = section1_text[after_age.end() :]
        match = re.search(r"\b(\d{1,2})\b\s+(?=\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", tail)
        if match:
            return match.group(1)

    match = re.search(r"\b(\d{1,2})\b\s+(?=\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", section1_text)
    if match:
        return match.group(1)
    return ""


def _extract_diagnosis(section1_text: str, summary_row: str) -> str:
    compact = _normalize_space(section1_text)
    keyword_pattern = r"\b(" + "|".join(re.escape(word) for word in MEDICAL_DIAG_KEYWORDS) + r")\b"
    match = re.search(keyword_pattern, compact, flags=re.IGNORECASE)
    candidate = ""
    if match:
        candidate = compact[match.start() : match.start() + 180]

    if not candidate:
        candidate = summary_row

    candidate = DATE_TOKEN_RE.sub(" ", candidate)
    candidate = TIME_TOKEN_RE.sub(" ", candidate)
    candidate = re.sub(r"\b20\d{10}\b", " ", candidate)
    candidate = re.sub(r"\bDR\.?\s*[A-Z.\s]{2,25}", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b(ICU|HDU|SECTION|PATIENT IDENTIFICATION|PRIMARY CONSULTANT|PRIMARY DIAGNOSIS)\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b\d{1,2}\b", " ", candidate)
    candidate = _normalize_space(candidate).strip(" ,;-")
    if not candidate:
        return "ICU case from RMO sheet"
    return candidate[:180]


def _extract_status(summary_row: str, supports: str, red_flag_row: str) -> str:
    upper = summary_row.upper()
    for word in STATUS_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", upper):
            return word
    supports_upper = supports.upper()
    if "MV" in supports_upper or "VASOPRESSOR" in supports_upper:
        return "CRITICAL"
    if re.search(r"\bY\b", red_flag_row.upper()):
        return "SICK"
    return "SERIOUS"


def _extract_supports(section3_row: str, section4_row: str, section6_row: str, section9_row: str, section12_row: str) -> str:
    text = " ".join([section3_row, section4_row, section6_row, section9_row, section12_row]).upper()
    supports: list[str] = []
    if any(token in text for token in [" MV ", "VENT", "PRVC", "PCV", "VOLUME CONTROL", "INTUB"]):
        supports.append("MV")
    if any(token in text for token in ["NIV", "BIPAP", "BI PAP", "CPAP"]):
        supports.append("NIV")
    if any(token in text for token in ["NORAD", "NOREPI", "ADRENALINE", "EPINEPHRINE", "VASOPRESSOR", "INOTROPE", "DOBUT", "DOPAMINE"]):
        supports.append("Vasopressor")
    if any(token in text for token in ["HD", "DIALYSIS", "SLED", "RRT", "MHD"]):
        supports.append("Dialysis")
    if any(token in text for token in [" O2 ", "HFNC", "NRBM", "VENTURI", "OXYGEN"]):
        supports.append("O2")
    if not supports:
        return ""
    deduped = []
    seen = set()
    for item in supports:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return " | ".join(deduped)


def _extract_red_flag_issue(section10_row: str) -> str:
    if not section10_row:
        return ""
    text = _strip_datetime_prefix(section10_row)
    text = re.sub(r"^\b(Y|N)\b", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\b(Y|N)\b\s*$", "", text, flags=re.IGNORECASE).strip()
    return _normalize_space(text)[:180]


def _extract_investigation_actions(section9_row: str, section11_row: str, section12_row: str) -> tuple[str, str, str]:
    section9_columns = _split_columns(section9_row)
    pending = ""
    actions_done = ""
    planned = ""

    if len(section9_columns) >= 4:
        pending = _strip_datetime_prefix(section9_columns[1])
        actions_done = _strip_datetime_prefix(section9_columns[2])
        planned = _strip_datetime_prefix(section9_columns[3])
    elif len(section9_columns) >= 2:
        pending = _strip_datetime_prefix(section9_columns[-1])

    pending = "" if pending.upper() in {"NIL", "NA", "N/A", "-"} else pending
    actions_done = "" if actions_done.upper() in {"NIL", "NA", "N/A", "-"} else actions_done
    planned = "" if planned.upper() in {"NIL", "NA", "N/A", "-"} else planned

    summary_payload = _strip_datetime_prefix(section11_row)
    advice_payload = _strip_datetime_prefix(section12_row)

    if summary_payload and summary_payload not in planned:
        planned = _normalize_space(f"{planned} | {summary_payload}" if planned else summary_payload)
    if advice_payload and advice_payload not in planned:
        planned = _normalize_space(f"{planned} | {advice_payload}" if planned else advice_payload)
    if advice_payload and advice_payload not in actions_done:
        actions_done = _normalize_space(f"{actions_done} | {advice_payload}" if actions_done else advice_payload)

    if not pending:
        pending_keywords = re.findall(
            r"(?:PENDING|REPORT|FOLLOW[- ]?UP|TO DO|PLAN FOR|ADVISED FOR)\s+[A-Z0-9/ .,-]{3,80}",
            f"{section9_row} {section11_row} {section12_row}",
            flags=re.IGNORECASE,
        )
        if pending_keywords:
            pending = " | ".join(_normalize_space(item) for item in pending_keywords[:3])

    return actions_done[:200], planned[:220], pending[:200]


def _extract_key_labs(section2_row: str, section3_row: str, section4_row: str, section6_row: str, section7_row: str) -> str:
    snippets: list[str] = []
    for row in [section2_row, section3_row, section4_row, section6_row, section7_row]:
        clean = _strip_datetime_prefix(row)
        if not clean:
            continue
        if any(token in clean.upper() for token in ["ABG", "LACTATE", "CREAT", "RBS", "WBC", "CRP", "PROCALCITONIN", "K", "NA", "SPO2"]):
            snippets.append(clean)
    if not snippets:
        return ""
    compact = " | ".join(snippets[:3])
    return _normalize_space(compact)[:220]


def _parse_block(block: str) -> dict[str, str]:
    section1 = _extract_section(block, 1)
    section2 = _extract_section(block, 2)
    section3 = _extract_section(block, 3)
    section4 = _extract_section(block, 4)
    section6 = _extract_section(block, 6)
    section7 = _extract_section(block, 7)
    section9 = _extract_section(block, 9)
    section10 = _extract_section(block, 10)
    section11 = _extract_section(block, 11)
    section12 = _extract_section(block, 12)

    sec2_row = _extract_first_row_block(section2)
    sec3_row = _extract_first_row_block(section3)
    sec4_row = _extract_first_row_block(section4)
    sec6_row = _extract_first_row_block(section6)
    sec7_row = _extract_first_row_block(section7)
    sec9_row = _extract_first_row_block(section9)
    sec10_row = _extract_first_row_block(section10)
    sec11_row = _extract_first_row_block(section11)
    sec12_row = _extract_first_row_block(section12)

    bed = _extract_bed(section1)
    patient_id = _extract_patient_id(section1)
    supports = _extract_supports(sec3_row, sec4_row, sec6_row, sec9_row, sec12_row)
    status = _extract_status(sec11_row, supports, sec10_row)
    diagnosis = _extract_diagnosis(section1, sec11_row)
    new_issues = _extract_red_flag_issue(sec10_row)
    actions_done, plan_next_12h, pending = _extract_investigation_actions(sec9_row, sec11_row, sec12_row)
    key_labs = _extract_key_labs(sec2_row, sec3_row, sec4_row, sec6_row, sec7_row)

    return {
        "bed": bed,
        "patient_id": patient_id,
        "diagnosis": diagnosis,
        "status": status,
        "supports": supports,
        "new_issues": new_issues,
        "actions_done": actions_done,
        "plan_next_12h": plan_next_12h,
        "pending": pending,
        "key_labs_imaging": key_labs,
    }


def parse_combined_rmo_text(full_text: str) -> dict[str, Any]:
    blocks = _split_blocks(full_text)
    table_rows: list[dict[str, str]] = []
    warnings: list[str] = []

    if not blocks:
        return {
            "table_rows": [],
            "blocks_detected": 0,
            "warnings": ["No RMO formatted blocks detected."],
            "debug_blocks": [],
        }

    for idx, block in enumerate(blocks, start=1):
        parsed = _parse_block(block)
        if not parsed.get("bed"):
            warnings.append(f"Block {idx}: bed could not be parsed and was skipped.")
            continue
        table_rows.append(parsed)

    if not table_rows:
        warnings.append("No patient rows parsed from RMO PDF blocks.")

    return {
        "table_rows": table_rows,
        "blocks_detected": len(blocks),
        "warnings": warnings,
        "debug_blocks": [_normalize_space(block)[:1200] for block in blocks[:2]],
    }


def parse_combined_rmo_pdf(filename: str, data: bytes) -> dict[str, Any]:
    if not str(filename or "").lower().endswith(".pdf"):
        raise ExtractionError("Combined RMO parser supports PDF only.")

    pages = extract_pdf_pages(data)
    if not pages:
        raise ExtractionError("No extractable text found in PDF.")

    full_text = "\n\n".join(page_text for _page_no, page_text in pages)
    result = parse_combined_rmo_text(full_text)
    if result.get("blocks_detected", 0) <= 0:
        raise ExtractionError("Uploaded PDF is not in expected combined RMO format.")
    return result

