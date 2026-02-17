from __future__ import annotations

import re
from typing import Any

STATUS_SEVERITY = {
    "OTHER": 0,
    "SERIOUS": 1,
    "SICK": 2,
    "CRITICAL": 3,
    "DECEASED": 4,
}

SUPPORT_ORDER = ["MV", "NIV", "VASO", "RRT", "O2"]
SUPPORT_SCORE = {
    "MV": 2.0,
    "NIV": 1.0,
    "VASO": 2.0,
    "RRT": 2.0,
    "O2": 0.5,
}


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def split_field_items(value: Any) -> list[str]:
    text = _normalized_text(value)
    if not text:
        return []

    items: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"\n|;|\s\|\s|,", text):
        clean = re.sub(r"^[\-\*\u2022\s]+", "", chunk).strip()
        if not clean:
            continue
        if clean.lower() in {"-", "none", "nil", "na", "n/a"}:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(clean)
    return items


def status_group_from_text(status: Any) -> str:
    lower = _normalized_text(status).lower()
    if "deceased" in lower or "dead" in lower:
        return "DECEASED"
    if "critical" in lower or "crit" in lower:
        return "CRITICAL"
    if "sick" in lower:
        return "SICK"
    if "serious" in lower:
        return "SERIOUS"
    return "OTHER"


def _status_severity(status: Any) -> int:
    return STATUS_SEVERITY.get(status_group_from_text(status), 0)


def support_labels_from_state(state_row: dict[str, Any]) -> list[str]:
    source = " ".join(
        [
            _normalized_text(state_row.get("supports", "")),
            _normalized_text(state_row.get("actions_done", "")),
            _normalized_text(state_row.get("plan_next_12h", "")),
            _normalized_text(state_row.get("pending", "")),
        ]
    ).lower()

    labels: list[str] = []
    if any(term in source for term in ["mv", "ventilator", "mechanical ventilation"]):
        labels.append("MV")
    if any(term in source for term in ["niv", "bipap", "bi pap", "cpap"]):
        labels.append("NIV")
    if any(
        term in source
        for term in [
            "norad",
            "norepinephrine",
            "adrenaline",
            "epinephrine",
            "vasopressor",
            "vaso",
        ]
    ):
        labels.append("VASO")
    if any(term in source for term in ["rrt", "hd", "sled", "dialysis", "cvvh"]):
        labels.append("RRT")
    if any(term in source for term in ["o2", "oxygen", "hfnc", "nrbm"]):
        labels.append("O2")

    labels = sorted(set(labels), key=lambda label: SUPPORT_ORDER.index(label))
    return labels


def _support_score(labels: list[str]) -> float:
    return sum(SUPPORT_SCORE.get(label, 0.0) for label in labels)


def _issue_phrases(state_row: dict[str, Any]) -> list[str]:
    joined = " | ".join(
        [
            _normalized_text(state_row.get("diagnosis", "")),
            _normalized_text(state_row.get("new_issues", "")),
        ]
    )
    return split_field_items(joined)


def _pending_items(state_row: dict[str, Any]) -> list[str]:
    return split_field_items(state_row.get("pending", ""))


def _patient_label(state_row: dict[str, Any]) -> str:
    bed = _normalized_text(state_row.get("bed", "")) or "-"
    patient_id = _normalized_text(state_row.get("patient_id", "")) or "No ID"
    return f"Bed {bed} | {patient_id}"


def _change_summary_lines(change: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    status_before = change.get("previous_status_group", "")
    status_after = change.get("current_status_group", "")
    if status_before and status_before != status_after:
        lines.append(f"Status: {status_before} -> {status_after}")

    added = change.get("supports_added", []) or []
    removed = change.get("supports_removed", []) or []
    if added:
        lines.append(f"Supports added: {', '.join(added)}")
    if removed:
        lines.append(f"Supports removed: {', '.join(removed)}")

    pending_new = change.get("pending_new", []) or []
    pending_resolved = change.get("pending_resolved", []) or []
    if pending_new or pending_resolved:
        parts = []
        if pending_new:
            parts.append(f"new {len(pending_new)}")
        if pending_resolved:
            parts.append(f"resolved {len(pending_resolved)}")
        lines.append(f"Pending delta: {', '.join(parts)}")

    issues_added = change.get("issues_added", []) or []
    issues_removed = change.get("issues_removed", []) or []
    if issues_added:
        lines.append(f"New issue phrases: {', '.join(issues_added[:3])}")
    elif issues_removed:
        lines.append(f"Issue phrases removed: {', '.join(issues_removed[:3])}")

    if not lines:
        lines.append("No major change vs previous snapshot")
    return lines[:4]


def _patient_change(current_row: dict[str, Any], previous_row: dict[str, Any] | None) -> dict[str, Any]:
    current_status = status_group_from_text(current_row.get("status", ""))
    previous_status = status_group_from_text(previous_row.get("status", "")) if previous_row else ""

    current_supports = support_labels_from_state(current_row)
    previous_supports = support_labels_from_state(previous_row) if previous_row else []

    current_support_set = set(current_supports)
    previous_support_set = set(previous_supports)
    supports_added = [label for label in SUPPORT_ORDER if label in (current_support_set - previous_support_set)]
    supports_removed = [label for label in SUPPORT_ORDER if label in (previous_support_set - current_support_set)]

    current_pending = _pending_items(current_row)
    previous_pending = _pending_items(previous_row) if previous_row else []

    current_pending_set = {item.lower(): item for item in current_pending}
    previous_pending_set = {item.lower(): item for item in previous_pending}

    pending_new = [
        current_pending_set[key]
        for key in current_pending_set
        if key not in previous_pending_set
    ]
    pending_resolved = [
        previous_pending_set[key]
        for key in previous_pending_set
        if key not in current_pending_set
    ]
    pending_unresolved = [
        current_pending_set[key]
        for key in current_pending_set
        if key in previous_pending_set
    ]

    current_issues = _issue_phrases(current_row)
    previous_issues = _issue_phrases(previous_row) if previous_row else []
    current_issue_set = {item.lower(): item for item in current_issues}
    previous_issue_set = {item.lower(): item for item in previous_issues}

    issues_added = [
        current_issue_set[key]
        for key in current_issue_set
        if key not in previous_issue_set
    ]
    issues_removed = [
        previous_issue_set[key]
        for key in previous_issue_set
        if key not in current_issue_set
    ]

    previous_severity = _status_severity(previous_row.get("status", "")) if previous_row else -1
    current_severity = _status_severity(current_row.get("status", ""))
    previous_support_score = _support_score(previous_supports)
    current_support_score = _support_score(current_supports)

    if previous_row is None:
        trend = "NEW ADMISSION"
    elif current_severity > previous_severity or current_support_score > previous_support_score:
        trend = "DETERIORATED"
    elif current_severity < previous_severity or current_support_score < previous_support_score:
        trend = "IMPROVED"
    else:
        trend = "STABLE"

    change = {
        "patient_key": str(current_row.get("patient_key", "")),
        "patient_label": _patient_label(current_row),
        "patient_id": _normalized_text(current_row.get("patient_id", "")),
        "bed": _normalized_text(current_row.get("bed", "")),
        "trend": trend,
        "current_status_group": current_status,
        "previous_status_group": previous_status,
        "supports_current": current_supports,
        "supports_previous": previous_supports,
        "supports_added": supports_added,
        "supports_removed": supports_removed,
        "pending_new": pending_new,
        "pending_resolved": pending_resolved,
        "pending_unresolved": pending_unresolved,
        "issues_added": issues_added,
        "issues_removed": issues_removed,
        "current_row": current_row,
        "previous_row": previous_row,
    }
    change["summary_lines"] = _change_summary_lines(change)
    return change


def compute_snapshot_changes(
    current_rows: list[dict[str, Any]],
    previous_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    previous_map: dict[str, dict[str, Any]] = {}
    for row in previous_rows:
        key = str(row.get("patient_key", "")).strip()
        if key:
            previous_map[key] = row

    changes: list[dict[str, Any]] = []
    for row in current_rows:
        key = str(row.get("patient_key", "")).strip()
        if not key:
            continue
        changes.append(_patient_change(row, previous_map.get(key)))
    return changes


def group_changes(changes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "deteriorated": [item for item in changes if item.get("trend") == "DETERIORATED"],
        "improved": [item for item in changes if item.get("trend") == "IMPROVED"],
        "pending_resolved": [item for item in changes if item.get("pending_resolved")],
        "new_pending": [item for item in changes if item.get("pending_new")],
        "support_escalations": [item for item in changes if item.get("supports_added")],
    }
