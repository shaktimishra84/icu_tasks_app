from __future__ import annotations

import os
import re
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
try:
    import pandas as pd
except Exception:
    pd = None

from src.analysis_engine import ClinicalTaskAdvisor
from src.batch_mode import (
    build_all_beds_outputs,
    infer_round_file_order,
)
from src.course_docx import generate_course_docx_safe
from src.extractors import ExtractionError, extract_text
from src.history_store import (
    ICUHistoryStore,
    hash_payload,
    patient_key_from_source_row,
    patient_key_from_output_row,
)
from src.knowledge_base import KnowledgeBase
from src.rounds_tracker import (
    compute_snapshot_changes,
    group_changes,
    split_field_items,
    status_group_from_text,
    support_labels_from_state,
)
from src.rounds_pdf import generate_rounds_pdf


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
RESOURCE_STORE = DATA_DIR / "resources_index.json"
TRACKER_DB_DIR = DATA_DIR
RESOURCES_ROOT = APP_DIR / "resources"
ENV_FILE = APP_DIR / ".env"
ICU_UNITS = [
    "MICU",
    "CCM ICU",
    "ICU 1",
    "ICU 2",
    "ICU 3",
    "ICU 4",
    "ICU 5",
    "ICU 6",
    "ICU 7",
    "Berhampur ICU",
]
MISSING_PATIENT_OUTCOME_OPTIONS = ["Select...", "Death", "DAMA", "Discharge", "Shifted"]


def _truthy_env(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


ENABLE_ACCESS_GATE = _truthy_env("ENABLE_ACCESS_GATE", default=False)
AUTO_BUILD_INDEX_ON_START = _truthy_env("AUTO_BUILD_INDEX_ON_START", default=True)


def _load_local_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (key not in os.environ or not os.environ.get(key, "").strip()):
            os.environ[key] = value

DATA_DIR.mkdir(parents=True, exist_ok=True)
_load_local_env_file()


def _split_emails(raw: str) -> set[str]:
    emails: set[str] = set()
    for part in str(raw or "").split(","):
        value = part.strip().lower()
        if value:
            emails.add(value)
    return emails


def _allowed_users() -> set[str]:
    env_allowed = _split_emails(os.getenv("ALLOWED_USERS", ""))
    if env_allowed:
        return env_allowed

    try:
        secrets_value = st.secrets.get("ALLOWED_USERS", "")
    except Exception:
        secrets_value = ""

    if isinstance(secrets_value, (list, tuple, set)):
        return {str(item).strip().lower() for item in secrets_value if str(item).strip()}
    return _split_emails(str(secrets_value))


def _streamlit_user_obj() -> Any:
    if hasattr(st, "user"):
        try:
            return st.user
        except Exception:
            return None
    if hasattr(st, "experimental_user"):
        try:
            return st.experimental_user
        except Exception:
            return None
    return None


def _user_email(user_obj: Any) -> str:
    if user_obj is None:
        return ""
    if isinstance(user_obj, dict):
        return str(user_obj.get("email", "")).strip().lower()
    try:
        if hasattr(user_obj, "get"):
            value = user_obj.get("email", "")
            if value:
                return str(value).strip().lower()
    except Exception:
        pass
    value = getattr(user_obj, "email", "")
    return str(value or "").strip().lower()


def _user_logged_in(user_obj: Any) -> bool:
    if user_obj is None:
        return False
    for attr in ("is_logged_in", "logged_in"):
        value = getattr(user_obj, attr, None)
        if isinstance(value, bool):
            return value
    if isinstance(user_obj, dict):
        value = user_obj.get("is_logged_in")
        if isinstance(value, bool):
            return value
    return bool(_user_email(user_obj))


def _auth_secrets_configured() -> bool:
    try:
        auth = st.secrets.get("auth", {})
    except Exception:
        auth = {}
    if not isinstance(auth, dict):
        return False
    required = ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url")
    return all(str(auth.get(key, "")).strip() for key in required)


def _enforce_allowed_users() -> None:
    allowed = _allowed_users()
    if not allowed:
        return
    if not _auth_secrets_configured():
        st.sidebar.warning("Access control skipped: `[auth]` secrets are incomplete.")
        return

    user_obj = _streamlit_user_obj()
    email = _user_email(user_obj)
    logged_in = _user_logged_in(user_obj)

    if not logged_in:
        st.title("ICU Task Assistant")
        st.warning("Sign-in required. Access is restricted to allowed users.")
        if hasattr(st, "login"):
            if st.button("Sign in", type="primary", use_container_width=True):
                try:
                    st.login()
                except Exception as error:
                    st.error(f"Login is not configured correctly: {error}")
        else:
            st.error("This Streamlit runtime does not support `st.login`.")
        st.stop()

    if email not in allowed:
        st.error(f"Access denied for `{email or 'unknown user'}`.")
        if hasattr(st, "logout") and st.button("Sign out", use_container_width=True):
            st.logout()
        st.stop()

    st.sidebar.caption(f"Signed in: {email}")
    if hasattr(st, "logout"):
        if st.sidebar.button("Sign out", use_container_width=True):
            st.logout()


def _status_chip(status_group: str) -> str:
    tone = {
        "CRITICAL": "critical",
        "SICK": "sick",
        "SERIOUS": "serious",
        "DECEASED": "deceased",
    }.get(status_group, "other")
    return f"<span class='icu-chip {tone}'>{escape(status_group)}</span>"


def _status_color(status_group: str) -> str:
    return {
        "CRITICAL": "#b91c1c",
        "SICK": "#b45309",
        "SERIOUS": "#854d0e",
        "DECEASED": "#374151",
        "OTHER": "#334155",
    }.get(status_group, "#334155")


def _badge(label: str, color: str = "#0f172a") -> str:
    _ = color
    return f"<span class='icu-pill'>{escape(label)}</span>"


def _support_badges(record: dict[str, Any]) -> list[str]:
    badges: list[str] = []
    if record.get("_is_mv"):
        badges.append("MV")
    if record.get("_is_niv"):
        badges.append("NIV/BiPAP")
    if record.get("_is_vaso"):
        badges.append("VASO")
    if record.get("_is_rrt"):
        badges.append("RRT/HD/SLED")
    if record.get("_is_o2"):
        badges.append("O2")
    return badges


def _decision_labs_line(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "-"
    parts = [part.strip() for part in re.split(r"\s\|\s|;|\n", cleaned) if part.strip()]
    if not parts:
        return cleaned

    decision_keywords = [
        "abnormal",
        "high",
        "low",
        "elevated",
        "critical",
        "worsening",
        "lactate",
        "creat",
        "potassium",
        "sodium",
        "ct",
        "mri",
        "hrct",
        "ctpa",
        "ugie",
        "report",
        "edema",
        "bleed",
    ]
    selected = [part for part in parts if any(keyword in part.lower() for keyword in decision_keywords)]
    if selected:
        return " | ".join(selected[:3])
    return parts[0]


def _inject_dashboard_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

        :root {
            --icu-bg-a: #0b1220;
            --icu-bg-b: #101f2e;
            --icu-bg-c: #16273b;
            --icu-panel: #121c2a;
            --icu-panel-elev: #172638;
            --icu-ink: #e5eef8;
            --icu-muted: #9ab0c7;
            --icu-line: #273f56;
            --icu-critical: #dc4f4f;
            --icu-sick: #e79a3b;
            --icu-serious: #d4b257;
            --icu-deceased: #7e8fa5;
            --icu-other: #63a8de;
            --icu-pill: #2b74a8;
            --icu-accent: #39c0c3;
        }

        .stApp, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 12% -2%, var(--icu-bg-c) 0%, transparent 48%),
                radial-gradient(circle at 86% -8%, #22344b 0%, transparent 42%),
                linear-gradient(180deg, var(--icu-bg-a) 0%, var(--icu-bg-b) 100%);
            color: var(--icu-ink);
            font-family: "Space Grotesk", "Avenir Next", "Trebuchet MS", sans-serif;
        }

        h1, h2, h3, h4, p, li, div, span, label {
            color: var(--icu-ink);
            font-family: "Space Grotesk", "Avenir Next", "Trebuchet MS", sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e1725 0%, #132132 100%);
            border-right: 1px solid var(--icu-line);
        }

        [data-testid="stSidebar"] * {
            color: var(--icu-ink) !important;
        }

        [data-baseweb="input"] input,
        [data-baseweb="select"] input,
        textarea,
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stDateInput input {
            background: #1a2a3d !important;
            border-color: #36516a !important;
            color: var(--icu-ink) !important;
        }

        [data-baseweb="select"] svg,
        .stSelectbox svg,
        .stDateInput svg {
            fill: var(--icu-muted) !important;
        }

        .stButton button,
        .stDownloadButton button {
            background: #1c2f45 !important;
            color: #e8f2fb !important;
            border: 1px solid #3a5975 !important;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            background: #24405c !important;
            border-color: #4d7396 !important;
        }

        .stAlert {
            background: #18283a !important;
            border: 1px solid #2f4963 !important;
            color: var(--icu-ink) !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(18, 28, 42, 0.92), rgba(12, 20, 31, 0.95));
            border: 1px solid var(--icu-line) !important;
            box-shadow: 0 12px 28px rgba(2, 6, 12, 0.36) !important;
        }

        [data-testid="stTabs"] [role="tab"] {
            border-radius: 999px;
            background: #1a2a3b;
            border: 1px solid #314d66;
            margin-right: 6px;
            padding: 8px 12px;
            color: #d5e4f3 !important;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: #27808a !important;
            color: #ffffff !important;
            border-color: #35a7b2 !important;
        }

        .icu-metric-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(120px, 1fr));
            gap: 10px;
            margin: 8px 0 16px 0;
        }

        .icu-metric-card {
            border: 1px solid var(--icu-line);
            border-radius: 14px;
            background: linear-gradient(145deg, var(--icu-panel-elev), var(--icu-panel));
            padding: 10px 12px;
            box-shadow: 0 10px 26px rgba(1, 5, 10, 0.34);
        }

        .icu-metric-label {
            font-size: 11px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--icu-muted);
        }

        .icu-metric-value {
            font-size: 24px;
            line-height: 1.1;
            font-weight: 700;
            margin-top: 3px;
        }

        .icu-chip {
            display: inline-block;
            color: #fff;
            border-radius: 999px;
            padding: 2px 9px;
            font-size: 11px;
            letter-spacing: 0.02em;
            margin-left: 6px;
            vertical-align: middle;
        }

        .icu-chip.critical { background: var(--icu-critical); }
        .icu-chip.sick { background: var(--icu-sick); }
        .icu-chip.serious { background: var(--icu-serious); }
        .icu-chip.deceased { background: var(--icu-deceased); }
        .icu-chip.other { background: var(--icu-other); }

        .icu-pill {
            display: inline-block;
            background: var(--icu-pill);
            color: #fff;
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 11px;
            margin-left: 6px;
        }

        .icu-lab-note {
            font-family: "IBM Plex Mono", "Menlo", monospace;
            font-size: 12px;
            color: #bfe7f0;
            background: #132838;
            border: 1px solid #2c5a72;
            border-radius: 10px;
            padding: 6px 8px;
            margin-top: 8px;
        }

        .icu-column-head {
            margin: 6px 0 10px 0;
            padding: 6px 8px;
            background: rgba(22, 36, 52, 0.86);
            border: 1px dashed #365775;
            border-radius: 10px;
            font-weight: 700;
            font-size: 13px;
            color: #d9e8f8;
        }

        .stCaption, .stMarkdown p, .stMarkdown li {
            color: var(--icu-muted) !important;
        }

        @media (max-width: 1100px) {
            .icu-metric-grid {
                grid-template-columns: repeat(3, minmax(120px, 1fr));
            }
        }
        @media (max-width: 700px) {
            .icu-metric-grid {
                grid-template-columns: repeat(2, minmax(120px, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_tiles(records: list[dict[str, Any]]) -> None:
    total_beds = len(records)
    critical = sum(1 for row in records if row.get("_status_group") == "CRITICAL")
    mv_count = sum(1 for row in records if row.get("_is_mv"))
    niv_count = sum(1 for row in records if row.get("_is_niv"))
    vaso_count = sum(1 for row in records if row.get("_is_vaso"))
    pending_count = sum(1 for row in records if str(row.get("_pending_verbatim", "")).strip())
    metric_cards = [
        ("Beds online", total_beds),
        ("Critical", critical),
        ("Mechanical vent", mv_count),
        ("NIV/BiPAP", niv_count),
        ("Vasopressor", vaso_count),
        ("Pending reports", pending_count),
    ]
    cards_html = "".join(
        (
            "<div class='icu-metric-card'>"
            f"<div class='icu-metric-label'>{escape(label)}</div>"
            f"<div class='icu-metric-value'>{value}</div>"
            "</div>"
        )
        for label, value in metric_cards
    )
    st.markdown(f"<div class='icu-metric-grid'>{cards_html}</div>", unsafe_allow_html=True)


def _patient_id_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _extract_source_output_discrepancies(
    source_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    source_by_pid: dict[str, str] = {}
    output_by_pid: dict[str, str] = {}
    pid_display: dict[str, str] = {}

    for row in source_rows:
        patient_id = str(row.get("patient_id", "")).strip()
        bed = str(row.get("bed", "")).strip()
        key = _patient_id_key(patient_id)
        if not key:
            continue
        pid_display.setdefault(key, patient_id)
        if key not in source_by_pid:
            source_by_pid[key] = bed

    for row in output_rows:
        patient_id = str(row.get("Patient ID", "")).strip()
        bed = str(row.get("Bed", "")).strip()
        key = _patient_id_key(patient_id)
        if not key:
            continue
        pid_display.setdefault(key, patient_id)
        if key not in output_by_pid:
            output_by_pid[key] = bed

    missing_in_output: list[str] = []
    extra_in_output: list[str] = []
    bed_mismatch: list[str] = []

    for key, source_bed in source_by_pid.items():
        if key not in output_by_pid:
            missing_in_output.append(f"{pid_display.get(key, key)} (Bed {source_bed or '-'})")
            continue
        output_bed = output_by_pid.get(key, "")
        if source_bed and output_bed and source_bed != output_bed:
            bed_mismatch.append(
                f"{pid_display.get(key, key)}: Word Bed {source_bed} -> PDF Bed {output_bed}"
            )

    for key, output_bed in output_by_pid.items():
        if key not in source_by_pid:
            extra_in_output.append(f"{pid_display.get(key, key)} (Bed {output_bed or '-'})")

    missing_in_output.sort()
    extra_in_output.sort()
    bed_mismatch.sort()
    return {
        "missing_in_output": missing_in_output,
        "extra_in_output": extra_in_output,
        "bed_mismatch": bed_mismatch,
    }


def _render_discrepancy_audit(source_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]]) -> bool:
    if not source_rows or not output_rows:
        return False
    discrepancies = _extract_source_output_discrepancies(source_rows, output_rows)
    missing_in_output = discrepancies["missing_in_output"]
    extra_in_output = discrepancies["extra_in_output"]
    bed_mismatch = discrepancies["bed_mismatch"]
    has_discrepancy = bool(missing_in_output or extra_in_output or bed_mismatch)

    if has_discrepancy:
        st.error(
            "Source-vs-output mismatch detected. Please review before downloading PDF."
        )
        with st.expander("Discrepancy audit (Word table vs generated output)", expanded=True):
            st.markdown(f"- Missing in PDF output: **{len(missing_in_output)}**")
            for item in missing_in_output[:50]:
                st.markdown(f"  - {item}")
            st.markdown(f"- Present in PDF but missing in Word: **{len(extra_in_output)}**")
            for item in extra_in_output[:50]:
                st.markdown(f"  - {item}")
            st.markdown(f"- Bed number mismatches: **{len(bed_mismatch)}**")
            for item in bed_mismatch[:50]:
                st.markdown(f"  - {item}")
    else:
        st.success("Discrepancy audit passed: Word table and generated output patient mapping are aligned.")
    return has_discrepancy


def _apply_filters(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    st.sidebar.subheader("Round Filters")
    status_filter = st.sidebar.selectbox(
        "Status",
        ["ALL", "CRITICAL", "SICK", "SERIOUS"],
        index=0,
        key="round_status_filter",
    )

    st.sidebar.markdown("**Supports**")
    f_mv = st.sidebar.checkbox("MV", value=False, key="f_mv")
    f_niv = st.sidebar.checkbox("NIV/BiPAP", value=False, key="f_niv")
    f_vaso = st.sidebar.checkbox("Vasopressor", value=False, key="f_vaso")
    f_rrt = st.sidebar.checkbox("RRT/HD/SLED", value=False, key="f_rrt")

    st.sidebar.markdown("**Flags**")
    f_dama = st.sidebar.checkbox("DAMA mentioned", value=False, key="f_dama")
    f_deceased = st.sidebar.checkbox("DECEASED", value=False, key="f_deceased")
    f_pending_imaging = st.sidebar.checkbox("Pending imaging", value=False, key="f_pending_imaging")
    f_pending_consult = st.sidebar.checkbox("Pending consult", value=False, key="f_pending_consult")

    search_query = st.sidebar.text_input("Search patient id or diagnosis", key="f_search").strip().lower()

    def keep(row: dict[str, Any]) -> bool:
        if status_filter != "ALL" and row.get("_status_group") != status_filter:
            return False

        support_filters = []
        if f_mv:
            support_filters.append(bool(row.get("_is_mv")))
        if f_niv:
            support_filters.append(bool(row.get("_is_niv")))
        if f_vaso:
            support_filters.append(bool(row.get("_is_vaso")))
        if f_rrt:
            support_filters.append(bool(row.get("_is_rrt")))
        if support_filters and not any(support_filters):
            return False

        flags = set(row.get("_flags", []))
        if f_dama and not any("dama" in str(flag).lower() for flag in flags):
            return False
        if f_deceased and row.get("_status_group") != "DECEASED":
            return False
        if f_pending_imaging and "Pending critical imaging" not in flags:
            return False
        if f_pending_consult and "Pending consult" not in flags:
            return False

        if search_query:
            patient_id = str(row.get("Patient ID", "")).lower()
            diagnosis = str(row.get("Diagnosis", "")).lower()
            if search_query not in patient_id and search_query not in diagnosis:
                return False
        return True

    return [row for row in records if keep(row)]


def _render_bed_card(
    record: dict[str, Any],
    *,
    key_prefix: str = "default",
    show_course_button: bool = False,
    collapsible: bool = False,
) -> None:
    def _short_text(value: Any, limit: int = 70) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "..."

    def _render_bed_card_body(
        *,
        record: dict[str, Any],
        key_prefix: str,
        show_course_button: bool,
        patient_key: str,
        status_group: str,
    ) -> None:
        flags = record.get("_flags", [])
        severity = str(record.get("_flag_severity", "NONE"))
        trend = str(record.get("Round trend", "")).strip()
        deterioration_reason = str(record.get("Deterioration reasons", "")).strip()
        is_deteriorated = str(record.get("Deterioration since last round", "")).upper() == "YES"
        system_tags = [str(x) for x in (record.get("_system_tags", []) or [])]
        matched_algorithms = [str(x) for x in (record.get("_matched_algorithms", []) or [])]
        tracker_summary_lines = [str(item) for item in (record.get("_tracker_summary_lines", []) or [])]

        if is_deteriorated:
            st.markdown(
                f"<div style='font-weight:700;color:#b91c1c;margin-bottom:4px;'>"
                f"Deteriorated since last round"
                f"{': ' + escape(deterioration_reason) if deterioration_reason else ''}</div>",
                unsafe_allow_html=True,
            )
        elif trend:
            st.caption(f"Trend vs previous round: {trend}")
        if tracker_summary_lines:
            st.markdown("**Change summary (vs previous round):**")
            for line in tracker_summary_lines[:4]:
                st.markdown(f"- {line}")

        if flags:
            color = "#b91c1c" if severity == "RED" else "#b45309"
            st.markdown(
                f"<div style='font-weight:600;color:{color};margin-bottom:4px;'>"
                f"Flags ({escape(severity)}): {escape(', '.join(flags))}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(f"**Diagnosis:** {record.get('Diagnosis', '-') or '-'}")
        if show_course_button and patient_key:
            if st.button("View course", key=f"view_course_{key_prefix}_{patient_key}"):
                st.session_state[f"selected_patient_key_{key_prefix}"] = patient_key
                st.caption("Patient selected for course view and correction.")
        if system_tags:
            friendly_system = " | ".join(tag.split("_", 1)[-1].replace("_", " ").upper() for tag in system_tags)
            matched_text = ", ".join(matched_algorithms) if matched_algorithms else "None"
            st.markdown(
                f"<div style='color:#a7bfd7;font-size:12px;'>System: <strong>{escape(friendly_system)}</strong> "
                f"| Matched: {escape(matched_text)}</div>",
                unsafe_allow_html=True,
            )

        if status_group == "DECEASED":
            st.markdown("**DECEASED**")
            pending_admin = record.get("Pending (verbatim)", "")
            if pending_admin:
                _render_safe_items(pending_admin, max_items=2)
            return

        st.markdown("**A) Missing (High priority first)**")
        st.markdown("**Tests:**")
        _render_safe_items(record.get("Missing Tests", ""), max_items=6)
        st.markdown("**Imaging:**")
        _render_safe_items(record.get("Missing Imaging", ""), max_items=6)
        st.markdown("**Consults:**")
        _render_safe_items(record.get("Missing Consults", ""), max_items=6)
        st.markdown("**Care checks (deterministic):**")
        _render_safe_items(record.get("Care checks (deterministic)", ""), max_items=6)

        if st.session_state.get("show_already_covered", False):
            st.markdown("**B) Already covered**")
            st.markdown("**Tests:**")
            _render_safe_items(record.get("_covered_tests", ""), max_items=6)
            st.markdown("**Imaging:**")
            _render_safe_items(record.get("_covered_imaging", ""), max_items=6)
            st.markdown("**Consults:**")
            _render_safe_items(record.get("_covered_consults", ""), max_items=6)
            st.markdown("**Care checks:**")
            _render_safe_items(record.get("_covered_care_checks", ""), max_items=6)

        st.markdown("**C) Pending (verbatim)**")
        _render_safe_items(record.get("Pending (verbatim)", ""), max_items=6)
        unresolved_pending = str(record.get("_unresolved_pending", "")).strip()
        if unresolved_pending:
            st.markdown("**Unresolved pending (carry-forward):**")
            _render_safe_items(unresolved_pending, max_items=6)

        st.markdown(
            f"<div class='icu-lab-note'>Key labs/imaging: "
            f"{escape(_decision_labs_line(str(record.get('Key labs/imaging (1 line)', ''))))}</div>",
            unsafe_allow_html=True,
        )

    status_group = str(record.get("_status_group", "OTHER"))
    status_color = _status_color(status_group)
    support_badges = _support_badges(record)
    patient_key = str(record.get("_patient_key", "")).strip() or patient_key_from_output_row(record)
    pending_count = len(_split_display_items(record.get("Pending (verbatim)", "")))
    trend = str(record.get("Round trend", "")).strip() or "NEW/UNCHANGED"

    with st.container(border=True):
        st.markdown(
            f"<div style='height:4px;background:{status_color};border-radius:4px;margin-bottom:8px;'></div>",
            unsafe_allow_html=True,
        )
        badges_html = "".join(_badge(tag, "#1d4ed8") for tag in support_badges)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
            f"<strong>Bed {escape(str(record.get('Bed', '')))} | {escape(str(record.get('Patient ID', '')))}</strong>"
            f"{_status_chip(status_group)}{badges_html}</div>",
            unsafe_allow_html=True,
        )
        if collapsible:
            st.caption(
                f"Dx: {_short_text(record.get('Diagnosis', '-'), limit=64) or '-'} | "
                f"Trend: {trend} | Pending: {pending_count}"
            )
            with st.expander("Open details", expanded=False):
                _render_bed_card_body(
                    record=record,
                    key_prefix=key_prefix,
                    show_course_button=show_course_button,
                    patient_key=patient_key,
                    status_group=status_group,
                )
        else:
            _render_bed_card_body(
                record=record,
                key_prefix=key_prefix,
                show_course_button=show_course_button,
                patient_key=patient_key,
                status_group=status_group,
            )


def _status_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    order = {"CRITICAL": 0, "SICK": 1, "SERIOUS": 2, "DECEASED": 3, "OTHER": 4}
    return order.get(str(record.get("_status_group", "OTHER")), 9), str(record.get("Bed", ""))


def _bed_sort_value(value: Any) -> tuple[int, int | str]:
    text = str(value or "").strip()
    match = re.search(r"\d+", text)
    if match:
        return (0, int(match.group(0)))
    if text:
        return (1, text)
    return (2, "")


def _bed_sort_key(record: dict[str, Any]) -> tuple[tuple[int, int | str], str]:
    return _bed_sort_value(record.get("Bed", "")), str(record.get("Patient ID", ""))


def _split_display_items(text: Any) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"\n|;|\s\|\s", str(text or "")):
        cleaned = re.sub(r"^[\-\*\u2022]+\s*", "", chunk).strip()
        if not cleaned:
            continue
        if cleaned.lower() in {"-", "none", "nil", "na", "n/a"}:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(cleaned)
    return values


def _render_safe_items(text: Any, max_items: int = 6) -> None:
    items = _split_display_items(text)[:max_items]
    if not items:
        st.markdown("<div style='color:#9ab0c7;'>-</div>", unsafe_allow_html=True)
        return
    for item in items:
        st.markdown(
            f"<div style='font-size:14px;line-height:1.35;margin:1px 0;'>• {escape(item)}</div>",
            unsafe_allow_html=True,
        )

def _default_round_shift() -> str:
    return "Morning" if datetime.now().hour < 16 else "Evening"


def _unit_slug(unit_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(unit_name or "").lower()).strip("_")
    return slug or "default"


def _state_key_fragment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "")).strip("_") or "x"


def _history_db_path_for_unit(unit_name: str) -> Path:
    return TRACKER_DB_DIR / f"icu_{_unit_slug(unit_name)}.db"


def _clear_tracker_state() -> None:
    for key in [
        "tracker_output",
        "tracker_source_rows",
        "tracker_changes",
        "tracker_current_rows",
        "tracker_current_snapshot",
        "tracker_previous_snapshot",
        "tracker_context",
        "tracker_autosaved_signature",
    ]:
        st.session_state.pop(key, None)


def _infer_round_date_shift_from_filename(filename: str) -> tuple[str, str]:
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
        groups = match.groups()
        try:
            if len(groups[0]) == 4:
                parsed_date = date(int(groups[0]), int(groups[1]), int(groups[2]))
            else:
                parsed_date = date(int(groups[2]), int(groups[1]), int(groups[0]))
            break
        except ValueError:
            continue

    if any(token in lower_name for token in ["night", "_pm", "-pm", " pm", "evening"]):
        shift = "Night"
    elif any(token in lower_name for token in ["morning", "_am", "-am", " am"]):
        shift = "Morning"
    else:
        shift = _default_round_shift()

    return (parsed_date or date.today()).isoformat(), shift


def _safe_build_all_beds(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    try:
        return build_all_beds_outputs(rows)
    except RuntimeError as error:
        st.error(f"Batch generation failed: {error}")
        return []
    except Exception as error:
        st.error(f"Batch generation failed with unexpected error: {error}")
        with st.expander("Batch error details", expanded=False):
            st.exception(error)
        return []


def _show_extraction_error(filename: str, error: Exception) -> None:
    if isinstance(error, ExtractionError):
        st.error(f"File extraction failed for `{filename}`: {error}")
        return
    st.error(f"Unexpected extraction error for `{filename}`: {type(error).__name__}: {error}")
    with st.expander("Extraction error details", expanded=False):
        st.exception(error)


def _ensure_startup_index(knowledge_base: KnowledgeBase) -> None:
    if st.session_state.get("startup_index_checked", False):
        return
    st.session_state.startup_index_checked = True

    try:
        loaded = knowledge_base.load_from_store()
    except Exception as error:
        loaded = False
        st.warning(f"Index store load failed: {error}")

    if loaded and knowledge_base.chunk_count() > 0:
        st.session_state.index_ready = True
        st.session_state.startup_index_source = "store"
        return

    st.session_state.index_ready = False

    if not AUTO_BUILD_INDEX_ON_START:
        st.session_state.startup_index_source = "not_enabled"
        return

    has_pdfs = any(path.is_file() for path in RESOURCES_ROOT.rglob("*.pdf"))
    if not has_pdfs:
        st.session_state.startup_index_source = "no_pdfs"
        return

    with st.spinner("First run on this deployment: building PDF index..."):
        try:
            knowledge_base.build_from_resources()
        except Exception as error:
            st.session_state.startup_index_source = "build_failed"
            st.error(f"Automatic startup index build failed: {error}")
            with st.expander("Startup index build details", expanded=False):
                st.exception(error)
            return

    st.session_state.index_ready = knowledge_base.chunk_count() > 0
    st.session_state.startup_index_source = "auto_built"


def _render_deterioration_table(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    compare_columns = [
        "Bed",
        "Patient ID",
        "Diagnosis",
        "Status",
        "Round trend",
        "Deterioration since last round",
        "Deterioration reasons",
    ]
    rows = [{col: row.get(col, "") for col in compare_columns} for row in records]
    if pd is None:
        st.table(rows)
        return

    dataframe = pd.DataFrame(rows)

    def _row_style(row: Any) -> list[str]:
        is_deteriorated = str(row.get("Deterioration since last round", "")).upper() == "YES"
        if is_deteriorated:
            return ["background-color: #fee2e2; color: #7f1d1d; font-weight: 600;" for _ in row]
        return ["" for _ in row]

    st.dataframe(
        dataframe.style.apply(_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def _snapshot_label(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return "No round"
    return f"{snapshot.get('date', '-')} ({snapshot.get('shift', '-')})"


def _missing_patients_since_previous(
    previous_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not previous_rows:
        return []

    current_keys = {
        str(row.get("patient_key", "")).strip()
        for row in current_rows
        if str(row.get("patient_key", "")).strip()
    }

    missing: list[dict[str, Any]] = []
    for row in previous_rows:
        patient_key = str(row.get("patient_key", "")).strip()
        if not patient_key or patient_key in current_keys:
            continue
        bed = str(row.get("bed", "")).strip() or "-"
        patient_id = str(row.get("patient_id", "")).strip() or "No ID"
        missing.append(
            {
                "patient_key": patient_key,
                "patient_id": patient_id,
                "bed": bed,
                "status": str(row.get("status", "")).strip(),
                "label": f"Bed {bed} | {patient_id}",
            }
        )

    missing.sort(key=lambda item: (_bed_sort_value(item.get("bed", "")), str(item.get("patient_id", ""))))
    return missing


def _render_missing_patient_resolution(
    history_store: ICUHistoryStore,
    current_snapshot: dict[str, Any],
    missing_patients: list[dict[str, Any]],
    *,
    key_prefix: str,
) -> None:
    snapshot_id = int(current_snapshot.get("snapshot_id", 0))
    if snapshot_id <= 0:
        return

    st.markdown(f"### Missing from previous round ({len(missing_patients)})")
    if not missing_patients:
        st.caption("None")
        return

    saved_rows = history_store.get_missing_outcomes(snapshot_id)
    saved_by_key = {str(row.get("patient_key", "")).strip(): row for row in saved_rows}
    unresolved = [item for item in missing_patients if not str(saved_by_key.get(item["patient_key"], {}).get("outcome", "")).strip()]
    if unresolved:
        st.warning(
            f"{len(unresolved)} patient(s) missing in this round still need disposition. "
            "Please mark Death, DAMA, Discharge, or Shifted."
        )
    else:
        st.success("All missing patients have a saved disposition.")

    with st.form(key=f"missing_patient_resolution_{key_prefix}_{snapshot_id}"):
        rows_payload: list[dict[str, str]] = []
        for item in missing_patients:
            patient_key = str(item.get("patient_key", "")).strip()
            fragment = _state_key_fragment(patient_key)
            saved = saved_by_key.get(patient_key, {})
            default_outcome = str(saved.get("outcome", "")).strip()
            default_index = (
                MISSING_PATIENT_OUTCOME_OPTIONS.index(default_outcome)
                if default_outcome in MISSING_PATIENT_OUTCOME_OPTIONS
                else 0
            )
            default_notes = str(saved.get("notes", "")).strip()

            st.markdown(f"**{item.get('label', patient_key)}**")
            c1, c2, c3 = st.columns([1.2, 1.3, 2.5])
            with c1:
                st.caption(f"Last status: {item.get('status', '-') or '-'}")
            with c2:
                selected_outcome = st.selectbox(
                    "Outcome",
                    options=MISSING_PATIENT_OUTCOME_OPTIONS,
                    index=default_index,
                    key=f"missing_outcome_{key_prefix}_{snapshot_id}_{fragment}",
                )
            with c3:
                notes = st.text_input(
                    "Notes",
                    value=default_notes,
                    placeholder="Optional transfer details",
                    key=f"missing_notes_{key_prefix}_{snapshot_id}_{fragment}",
                )

            rows_payload.append(
                {
                    "patient_key": patient_key,
                    "patient_id": str(item.get("patient_id", "")).strip(),
                    "bed": str(item.get("bed", "")).strip(),
                    "status": str(item.get("status", "")).strip(),
                    "outcome": "" if selected_outcome == "Select..." else selected_outcome,
                    "notes": notes,
                }
            )

        submit = st.form_submit_button("Save missing-patient outcomes", use_container_width=True)

    if not submit:
        return

    saved_count = 0
    cleared_count = 0
    for row in rows_payload:
        patient_key = str(row.get("patient_key", "")).strip()
        if not patient_key:
            continue
        outcome = str(row.get("outcome", "")).strip()
        if outcome:
            history_store.save_missing_outcome(
                snapshot_id=snapshot_id,
                patient_key=patient_key,
                patient_id=str(row.get("patient_id", "")).strip(),
                bed=str(row.get("bed", "")).strip(),
                last_status=str(row.get("status", "")).strip(),
                outcome=outcome,
                notes=str(row.get("notes", "")).strip(),
            )
            saved_count += 1
        else:
            history_store.delete_missing_outcome(snapshot_id=snapshot_id, patient_key=patient_key)
            cleared_count += 1

    st.success(
        f"Missing-patient outcomes saved for {saved_count} patient(s). "
        f"Cleared outcome for {cleared_count} patient(s)."
    )
    try:
        st.rerun()
    except Exception:
        pass


def _change_sort_key(change: dict[str, Any]) -> tuple[tuple[int, int | str], str]:
    return _bed_sort_value(change.get("bed", "")), str(change.get("patient_id", ""))


def _render_change_group(title: str, entries: list[dict[str, Any]]) -> None:
    st.markdown(f"### {title} ({len(entries)})")
    if not entries:
        st.caption("None")
        return
    for change in sorted(entries, key=_change_sort_key):
        with st.container(border=True):
            st.markdown(f"**{change.get('patient_label', '-') or '-'}**")
            for line in list(change.get("summary_lines", []))[:4]:
                st.markdown(f"- {line}")
            unresolved = list(change.get("pending_unresolved", []))
            if unresolved:
                st.caption("Unresolved pending: " + " | ".join(unresolved[:4]))


def _render_changes_tab(
    changes: list[dict[str, Any]],
    previous_snapshot: dict[str, Any] | None,
    *,
    history_store: ICUHistoryStore,
    current_snapshot: dict[str, Any] | None,
    current_rows: list[dict[str, Any]],
    key_prefix: str = "round_tracker",
) -> None:
    if not changes:
        st.info("No tracker changes available yet.")
        return
    if previous_snapshot is None:
        st.info("This is the first stored round. Upload the next rounds sheet to see deltas.")
        return

    st.caption(f"Comparing against previous round: {_snapshot_label(previous_snapshot)}")
    grouped = group_changes(changes)
    _render_change_group("Deteriorated", grouped["deteriorated"])
    _render_change_group("Improved", grouped["improved"])
    _render_change_group("Pending resolved", grouped["pending_resolved"])
    _render_change_group("New pending", grouped["new_pending"])
    _render_change_group("Support escalations", grouped["support_escalations"])

    if current_snapshot is None:
        return
    previous_rows = history_store.get_snapshot_rows(int(previous_snapshot["snapshot_id"]))
    missing_patients = _missing_patients_since_previous(previous_rows, current_rows)
    _render_missing_patient_resolution(
        history_store,
        current_snapshot,
        missing_patients,
        key_prefix=key_prefix,
    )


def _source_rows_from_state_rows(state_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    source_rows: list[dict[str, str]] = []
    for row in state_rows:
        source_rows.append(
            {
                "bed": str(row.get("bed", "")).strip(),
                "patient_id": str(row.get("patient_id", "")).strip(),
                "diagnosis": str(row.get("diagnosis", "")).strip(),
                "status": str(row.get("status", "")).strip(),
                "supports": str(row.get("supports", "")).strip(),
                "new_issues": str(row.get("new_issues", "")).strip(),
                "actions_done": str(row.get("actions_done", "")).strip(),
                "plan_next_12h": str(row.get("plan_next_12h", "")).strip(),
                "pending": str(row.get("pending", "")).strip(),
                "key_labs_imaging": str(row.get("key_labs_imaging", "")).strip(),
            }
        )
    return source_rows


def _hydrate_tracker_from_snapshot(
    history_store: ICUHistoryStore,
    snapshot: dict[str, Any],
    *,
    unit_name: str,
) -> bool:
    snapshot_id = int(snapshot.get("snapshot_id", 0))
    if snapshot_id <= 0:
        return False

    current_state_rows = history_store.get_snapshot_rows(snapshot_id)
    if not current_state_rows:
        return False

    source_rows = _source_rows_from_state_rows(current_state_rows)
    current_output = _safe_build_all_beds(source_rows)
    if not current_output:
        return False

    previous_snapshot = history_store.get_previous_snapshot(snapshot_id)
    previous_state_rows = (
        history_store.get_snapshot_rows(int(previous_snapshot["snapshot_id"]))
        if previous_snapshot
        else []
    )
    changes = compute_snapshot_changes(current_state_rows, previous_state_rows)
    _annotate_output_with_changes(current_output, changes)

    st.session_state["tracker_output"] = current_output
    st.session_state["tracker_source_rows"] = source_rows
    st.session_state["tracker_changes"] = changes
    st.session_state["tracker_current_rows"] = current_state_rows
    st.session_state["tracker_current_snapshot"] = snapshot
    st.session_state["tracker_previous_snapshot"] = previous_snapshot
    st.session_state["tracker_context"] = f"Loaded saved round for {unit_name} from local DB."
    return True


def _render_tracker_views(
    *,
    history_store: ICUHistoryStore,
    knowledge_base: KnowledgeBase,
    selected_icu_unit: str,
) -> None:
    tracker_output = st.session_state.get("tracker_output", [])
    tracker_source_rows = st.session_state.get("tracker_source_rows", [])
    tracker_changes = st.session_state.get("tracker_changes", [])
    tracker_current_rows = st.session_state.get("tracker_current_rows", [])
    tracker_current_snapshot = st.session_state.get("tracker_current_snapshot")
    tracker_previous_snapshot = st.session_state.get("tracker_previous_snapshot")
    tracker_context_saved = str(st.session_state.get("tracker_context", "")).strip()

    if not tracker_output:
        return

    if tracker_context_saved:
        st.caption(tracker_context_saved)
    if tracker_current_snapshot:
        st.caption(
            f"Unit: {selected_icu_unit} | Current round: {_snapshot_label(tracker_current_snapshot)} | "
            f"Previous round: {_snapshot_label(tracker_previous_snapshot)}"
        )

    dashboard_tab, course_tab, changes_tab = st.tabs(
        ["Dashboard", "Patient course", "Changes"]
    )
    with dashboard_tab:
        _render_all_beds_panel(
            tracker_output,
            key_prefix="round_tracker",
            source_rows=tracker_source_rows,
            show_course_button=True,
        )
    with course_tab:
        _render_patient_course_tab(
            history_store,
            knowledge_base,
            tracker_current_snapshot,
            tracker_source_rows,
            tracker_output,
            selected_icu_unit,
            key_prefix="round_tracker",
        )
    with changes_tab:
        _render_changes_tab(
            tracker_changes,
            tracker_previous_snapshot,
            history_store=history_store,
            current_snapshot=tracker_current_snapshot,
            current_rows=tracker_current_rows,
            key_prefix="round_tracker",
        )


def _find_source_row_index(source_rows: list[dict[str, Any]], patient_key: str) -> int:
    if not patient_key:
        return -1
    for index, row in enumerate(source_rows):
        if patient_key_from_source_row(row) == patient_key:
            return index
    return -1


def _round_label(state_row: dict[str, Any]) -> str:
    return f"{state_row.get('date', '-')} ({state_row.get('shift', '-')})"


def _selected_output_row(output_rows: list[dict[str, Any]], patient_key: str) -> dict[str, Any] | None:
    for row in output_rows:
        row_key = str(row.get("_patient_key", "")).strip() or patient_key_from_output_row(row)
        if row_key == patient_key:
            return row
    return None


def _course_support_timeline(chronological_rows: list[dict[str, Any]]) -> list[str]:
    if not chronological_rows:
        return []
    labels = ["MV", "NIV", "VASO", "RRT", "O2"]
    lines: list[str] = []
    for label in labels:
        first_index = -1
        last_index = -1
        for index, row in enumerate(chronological_rows):
            supports = support_labels_from_state(row)
            if label in supports:
                if first_index < 0:
                    first_index = index
                last_index = index
        if first_index < 0:
            continue
        first_label = _round_label(chronological_rows[first_index])
        if last_index == len(chronological_rows) - 1:
            lines.append(f"{label}: started {first_label}, ongoing")
        else:
            last_label = _round_label(chronological_rows[last_index])
            lines.append(f"{label}: started {first_label}, stopped by {last_label}")
    return lines


def _course_pending_summary(chronological_rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    if not chronological_rows:
        return [], []
    latest_pending = split_field_items(chronological_rows[-1].get("pending", ""))
    latest_pending_keys = {item.lower() for item in latest_pending}
    all_previous: dict[str, str] = {}
    for row in chronological_rows[:-1]:
        for item in split_field_items(row.get("pending", "")):
            all_previous.setdefault(item.lower(), item)
    resolved = [value for key, value in all_previous.items() if key not in latest_pending_keys]
    return latest_pending, resolved


def _render_course_since_admission(chronological_rows: list[dict[str, Any]], selected_output: dict[str, Any] | None) -> None:
    if not chronological_rows:
        st.info("No round history available yet.")
        return

    admission = chronological_rows[0]
    current = chronological_rows[-1]
    current_status = status_group_from_text(current.get("status", ""))
    admission_status = status_group_from_text(admission.get("status", ""))

    st.markdown("### Course Since Admission")
    st.markdown(
        f"- **Admission baseline:** {_round_label(admission)} | Status `{admission_status}` | "
        f"Dx: {admission.get('diagnosis', '-') or '-'}"
    )
    st.markdown(
        f"- **Current state:** {_round_label(current)} | Status `{current_status}` | "
        f"Supports: {', '.join(support_labels_from_state(current)) or '-'}"
    )

    turning_points: list[str] = []
    for index in range(1, len(chronological_rows)):
        previous = chronological_rows[index - 1]
        now = chronological_rows[index]
        change_set = compute_snapshot_changes([now], [previous])
        if not change_set:
            continue
        change = change_set[0]
        if change.get("trend") in {"DETERIORATED", "IMPROVED"} or change.get("supports_added") or change.get("supports_removed"):
            summary = ", ".join(change.get("summary_lines", [])[:2])
            turning_points.append(f"{_round_label(now)}: {summary}")
    if turning_points:
        st.markdown("**Major turning points**")
        for item in turning_points[:8]:
            st.markdown(f"- {item}")

    support_lines = _course_support_timeline(chronological_rows)
    if support_lines:
        st.markdown("**Support trajectory**")
        for item in support_lines:
            st.markdown(f"- {item}")

    latest_pending, resolved_pending = _course_pending_summary(chronological_rows)
    st.markdown("**Pending timeline**")
    st.markdown(f"- Active pending now: {', '.join(latest_pending) if latest_pending else '-'}")
    st.markdown(f"- Resolved since admission: {', '.join(resolved_pending[:8]) if resolved_pending else '-'}")

    active_issues = split_field_items(
        " | ".join([str(current.get("diagnosis", "")), str(current.get("new_issues", ""))])
    )
    st.markdown(f"**Current active issues**: {', '.join(active_issues[:8]) if active_issues else '-'}")
    plan = split_field_items(current.get("plan_next_12h", ""))
    st.markdown(f"**Forward plan (next 24h)**: {'; '.join(plan[:6]) if plan else '-'}")

    st.markdown("### Crosscheck for Missed Items")
    if selected_output is None:
        st.info("No deterministic crosscheck output found for this patient.")
        return

    st.markdown("**Deterministic misses from current round**")
    st.markdown("Tests:")
    _render_safe_items(selected_output.get("Missing Tests", ""), max_items=8)
    st.markdown("Imaging:")
    _render_safe_items(selected_output.get("Missing Imaging", ""), max_items=8)
    st.markdown("Consults:")
    _render_safe_items(selected_output.get("Missing Consults", ""), max_items=8)
    st.markdown("Care checks:")
    _render_safe_items(selected_output.get("Care checks (deterministic)", ""), max_items=8)


def _render_resource_context_crosscheck(
    knowledge_base: KnowledgeBase,
    current_row: dict[str, Any],
    *,
    top_k: int = 4,
) -> list[tuple[Any, float]]:
    query = " ".join(
        [
            str(current_row.get("diagnosis", "")),
            str(current_row.get("new_issues", "")),
            str(current_row.get("supports", "")),
            str(current_row.get("key_labs_imaging", "")),
            str(current_row.get("pending", "")),
        ]
    ).strip()
    if not query:
        st.caption("No enough clinical text to query indexed resources.")
        return []
    if knowledge_base.chunk_count() == 0:
        st.caption("No indexed resources loaded for crosscheck context.")
        return []

    retrieved = knowledge_base.retrieve(query=query, top_k=top_k, only_neuro=False)
    if not retrieved:
        st.caption("No matched indexed resources for this course context.")
        return []

    st.markdown("**Indexed resource context (manual crosscheck)**")
    for chunk, score in retrieved:
        st.markdown(f"- `{chunk.file_name}` p.{chunk.page_number} (score {score:.3f})")
    return retrieved


def _render_patient_course_tab(
    history_store: ICUHistoryStore,
    knowledge_base: KnowledgeBase,
    current_round: dict[str, Any] | None,
    source_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    unit_name: str,
    *,
    key_prefix: str = "round_tracker",
) -> None:
    if not output_rows:
        latest_snapshot = history_store.get_latest_snapshot()
        if latest_snapshot is not None:
            latest_state_rows = history_store.get_snapshot_rows(int(latest_snapshot["snapshot_id"]))
            latest_source_rows = _source_rows_from_state_rows(latest_state_rows)
            latest_output_rows = _safe_build_all_beds(latest_source_rows)
            if latest_output_rows:
                previous_snapshot = history_store.get_previous_snapshot(int(latest_snapshot["snapshot_id"]))
                previous_rows = (
                    history_store.get_snapshot_rows(int(previous_snapshot["snapshot_id"]))
                    if previous_snapshot
                    else []
                )
                changes = compute_snapshot_changes(latest_state_rows, previous_rows)
                _annotate_output_with_changes(latest_output_rows, changes)
                output_rows = latest_output_rows
                source_rows = latest_source_rows
                current_round = latest_snapshot
        if not output_rows:
            st.info("No saved patient course available yet for this ICU.")
            return

    options: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for row in output_rows:
        patient_key = str(row.get("_patient_key", "")).strip() or patient_key_from_output_row(row)
        if not patient_key or patient_key in seen_keys:
            continue
        seen_keys.add(patient_key)
        bed = str(row.get("Bed", "")).strip() or "-"
        patient_id = str(row.get("Patient ID", "")).strip() or "No ID"
        options.append(
            {
                "patient_key": patient_key,
                "bed": bed,
                "label": f"Bed {bed} | {patient_id}",
            }
        )

    if not options:
        st.info("No patient records available for course view.")
        return

    options.sort(key=lambda item: (_bed_sort_value(item["bed"]), item["label"]))
    keys = [item["patient_key"] for item in options]
    labels_by_key = {item["patient_key"]: item["label"] for item in options}
    selected_state_key = f"selected_patient_key_{key_prefix}"
    selected_key = str(st.session_state.get(selected_state_key, "")).strip()
    if selected_key not in labels_by_key:
        selected_key = keys[0]

    selected_key = st.selectbox(
        "Selected patient",
        options=keys,
        index=keys.index(selected_key),
        format_func=lambda key: labels_by_key.get(key, key),
        key=f"course_selector_{key_prefix}",
    )
    st.session_state[selected_state_key] = selected_key

    history_limit = st.slider(
        "Course depth",
        min_value=3,
        max_value=14,
        value=7,
        key=f"history_depth_{key_prefix}",
    )
    history_rows = history_store.get_patient_history(selected_key, limit=history_limit)
    if not history_rows:
        source_index = _find_source_row_index(source_rows, selected_key)
        if source_index < 0:
            st.info("No course history found for this patient yet.")
            return
        fallback_row = dict(source_rows[source_index])
        history_rows = [
            {
                "snapshot_id": int(current_round.get("snapshot_id", 0)) if current_round else 0,
                "date": str(current_round.get("date", datetime.now().strftime("%Y-%m-%d"))) if current_round else datetime.now().strftime("%Y-%m-%d"),
                "shift": str(current_round.get("shift", "-")) if current_round else "-",
                "patient_key": selected_key,
                "patient_id": str(fallback_row.get("patient_id", "")),
                "bed": str(fallback_row.get("bed", "")),
                "diagnosis": str(fallback_row.get("diagnosis", "")),
                "status": str(fallback_row.get("status", "")),
                "supports": str(fallback_row.get("supports", "")),
                "new_issues": str(fallback_row.get("new_issues", "")),
                "actions_done": str(fallback_row.get("actions_done", "")),
                "plan_next_12h": str(fallback_row.get("plan_next_12h", "")),
                "pending": str(fallback_row.get("pending", "")),
                "key_labs_imaging": str(fallback_row.get("key_labs_imaging", "")),
            }
        ]
        st.caption("History was not found in local DB for this patient. Using current round row for course and Word export.")

    chronological_rows = list(reversed(history_rows))
    selected_output = _selected_output_row(output_rows, selected_key)
    _render_course_since_admission(chronological_rows, selected_output)
    try:
        resource_matches = _render_resource_context_crosscheck(knowledge_base, chronological_rows[-1], top_k=4)
    except Exception as error:
        resource_matches = []
        st.caption(f"Resource crosscheck unavailable: {error}")

    st.markdown("### Word Output")
    course_docx_bytes, course_docx_name, export_warning = generate_course_docx_safe(
        unit_name=unit_name,
        patient_label=labels_by_key.get(selected_key, selected_key),
        patient_key=selected_key,
        chronological_rows=chronological_rows,
        selected_output=selected_output,
        resource_matches=resource_matches,
    )
    if export_warning:
        st.caption(f"Word export generated with fallback renderer: {export_warning}")
    st.download_button(
        "Download ICU Course Word (.docx)",
        data=course_docx_bytes,
        file_name=course_docx_name,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
        key=f"download_course_docx_{key_prefix}_{selected_key}",
    )

    st.markdown("### Course Trend View")
    chart_rows: list[dict[str, Any]] = []
    for row in chronological_rows:
        status_group = status_group_from_text(row.get("status", ""))
        supports = support_labels_from_state(row)
        pending_count = len(split_field_items(row.get("pending", "")))
        chart_rows.append(
            {
                "Round": f"{row.get('date', '-')} ({row.get('shift', '-')})",
                "Status severity": {"OTHER": 0, "SERIOUS": 1, "SICK": 2, "CRITICAL": 3, "DECEASED": 4}.get(
                    status_group,
                    0,
                ),
                "Pending count": pending_count,
                "Status": status_group,
                "Supports": " | ".join(supports) if supports else "-",
                "Pending": " | ".join(split_field_items(row.get("pending", ""))[:4]) or "-",
                "New issues": row.get("new_issues", "") or "-",
                "Key labs/imaging": row.get("key_labs_imaging", "") or "-",
            }
        )

    if pd is not None and chart_rows:
        frame = pd.DataFrame(chart_rows)
        st.line_chart(
            frame.set_index("Round")[["Status severity", "Pending count"]],
            use_container_width=True,
        )
        st.dataframe(
            frame[["Round", "Status", "Supports", "Pending", "New issues", "Key labs/imaging"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.table(chart_rows)

    source_index = _find_source_row_index(source_rows, selected_key)
    if source_index < 0:
        st.warning("Selected patient is not present in current editable round rows.")
        return

    editable = dict(source_rows[source_index])
    st.markdown("### Correct Missing Information")
    with st.form(key=f"edit_course_{key_prefix}_{selected_key}"):
        c1, c2 = st.columns(2)
        with c1:
            bed = st.text_input("Bed", value=str(editable.get("bed", "")))
            diagnosis = st.text_area("Diagnosis", value=str(editable.get("diagnosis", "")), height=70)
            supports = st.text_area("Supports", value=str(editable.get("supports", "")), height=70)
            actions_done = st.text_area("Actions done", value=str(editable.get("actions_done", "")), height=90)
            pending = st.text_area("Pending", value=str(editable.get("pending", "")), height=100)
        with c2:
            patient_id = st.text_input("Patient ID", value=str(editable.get("patient_id", "")))
            status = st.text_input("Status", value=str(editable.get("status", "")))
            new_issues = st.text_area("New issues", value=str(editable.get("new_issues", "")), height=70)
            plan_next_12h = st.text_area("Plan next 12h", value=str(editable.get("plan_next_12h", "")), height=90)
            key_labs = st.text_area(
                "Key labs/imaging",
                value=str(editable.get("key_labs_imaging", "")),
                height=100,
            )
        submitted = st.form_submit_button("Save corrections and refresh dashboard", use_container_width=True)

    if not submitted:
        return

    updated_row = dict(editable)
    updated_row["bed"] = bed.strip()
    updated_row["patient_id"] = patient_id.strip()
    updated_row["diagnosis"] = diagnosis.strip()
    updated_row["status"] = status.strip()
    updated_row["supports"] = supports.strip()
    updated_row["new_issues"] = new_issues.strip()
    updated_row["actions_done"] = actions_done.strip()
    updated_row["plan_next_12h"] = plan_next_12h.strip()
    updated_row["pending"] = pending.strip()
    updated_row["key_labs_imaging"] = key_labs.strip()

    updated_rows = [dict(row) for row in source_rows]
    updated_rows[source_index] = updated_row
    updated_output = _safe_build_all_beds(updated_rows)
    if not updated_output:
        return

    try:
        round_date = (
            str(current_round.get("date", "")).strip()
            if current_round is not None
            else date.today().isoformat()
        ) or date.today().isoformat()
        round_shift = (
            str(current_round.get("shift", "")).strip()
            if current_round is not None
            else _default_round_shift()
        ) or _default_round_shift()
        file_hash = (
            str(current_round.get("file_hash", "")).strip()
            if current_round is not None
            else "manual-edit"
        ) or "manual-edit"

        snapshot_id = history_store.save_snapshot(
            snapshot_date=round_date,
            shift=round_shift,
            file_hash=file_hash,
            table_rows=updated_rows,
        )
        current_snapshot = history_store.get_snapshot(snapshot_id)
        previous_snapshot = history_store.get_previous_snapshot(snapshot_id)
        current_state_rows = history_store.get_snapshot_rows(snapshot_id)
        previous_state_rows = (
            history_store.get_snapshot_rows(int(previous_snapshot["snapshot_id"]))
            if previous_snapshot
            else []
        )
        changes = compute_snapshot_changes(current_state_rows, previous_state_rows)
        _annotate_output_with_changes(updated_output, changes)
    except Exception as error:
        st.error(f"Could not save corrections: {error}")
        with st.expander("Correction error details", expanded=False):
            st.exception(error)
        return

    st.session_state["tracker_output"] = updated_output
    st.session_state["tracker_source_rows"] = updated_rows
    st.session_state["tracker_changes"] = changes
    st.session_state["tracker_current_rows"] = current_state_rows
    st.session_state["tracker_current_snapshot"] = current_snapshot
    st.session_state["tracker_previous_snapshot"] = previous_snapshot
    new_selected_key = patient_key_from_source_row(updated_row)
    st.session_state[selected_state_key] = new_selected_key or selected_key
    st.success("Corrections saved and dashboard refreshed.")
    try:
        st.rerun()
    except Exception:
        pass


def _annotate_output_with_changes(
    output_rows: list[dict[str, Any]],
    changes: list[dict[str, Any]],
) -> None:
    changes_by_key = {str(item.get("patient_key", "")): item for item in changes}
    for row in output_rows:
        patient_key = patient_key_from_output_row(row)
        row["_patient_key"] = patient_key
        if not patient_key:
            continue
        change = changes_by_key.get(patient_key)
        if not change:
            continue

        trend = str(change.get("trend", ""))
        row["Round trend"] = trend
        row["Deterioration since last round"] = "YES" if trend == "DETERIORATED" else "NO"
        reasons = [line for line in list(change.get("summary_lines", [])) if "No major change" not in line]
        row["Deterioration reasons"] = " | ".join(reasons[:2])
        row["_tracker_summary_lines"] = list(change.get("summary_lines", []))[:4]
        row["_unresolved_pending"] = " | ".join(list(change.get("pending_unresolved", [])))


def _render_all_beds_panel(
    all_beds_output: list[dict[str, Any]],
    key_prefix: str = "default",
    source_rows: list[dict[str, Any]] | None = None,
    show_course_button: bool = False,
) -> None:
    if not all_beds_output:
        return

    all_beds_output = sorted(all_beds_output, key=_bed_sort_key)
    _render_summary_tiles(all_beds_output)
    has_discrepancy = _render_discrepancy_audit(source_rows or [], all_beds_output)

    has_round_trend = any(
        str(row.get("Round trend", "")).strip()
        and str(row.get("Round trend", "")).strip().upper() != "NEW ADMISSION"
        for row in all_beds_output
    )
    if has_round_trend:
        deteriorated_count = sum(
            1 for row in all_beds_output if str(row.get("Deterioration since last round", "")).upper() == "YES"
        )
        st.markdown("### Change Since Previous Round")
        st.caption(f"Deteriorated patients: {deteriorated_count}")
        _render_deterioration_table(all_beds_output)

    show_covered_key = f"show_covered_toggle_{key_prefix}"
    st.session_state["show_already_covered"] = st.toggle(
        "Show already covered items",
        value=st.session_state.get("show_already_covered", False),
        key=show_covered_key,
    )

    layout_mode = st.radio(
        "Dashboard layout",
        options=["Bed grid", "Detailed list"],
        horizontal=True,
        key=f"dashboard_layout_{key_prefix}",
    )

    filtered_records = sorted(_apply_filters(all_beds_output), key=_bed_sort_key)
    st.caption(f"Showing {len(filtered_records)} of {len(all_beds_output)} beds after filters")
    if layout_mode == "Detailed list":
        st.caption("Detailed list: bed-wise ascending order.")
        for row in filtered_records:
            _render_bed_card(
                row,
                key_prefix=key_prefix,
                show_course_button=show_course_button,
                collapsible=False,
            )
    else:
        st.caption("Bed grid: compact bed-wise cards. Click each card to open full details.")
        if not filtered_records:
            st.info("No beds match the current filters.")
        else:
            cards_per_row = 4 if len(filtered_records) >= 12 else 3
            for start in range(0, len(filtered_records), cards_per_row):
                columns = st.columns(cards_per_row)
                row_slice = filtered_records[start : start + cards_per_row]
                for col_idx, record in enumerate(row_slice):
                    with columns[col_idx]:
                        _render_bed_card(
                            record,
                            key_prefix=key_prefix,
                            show_course_button=show_course_button,
                            collapsible=True,
                        )

    shift_options = ["Morning", "Evening"]
    default_shift = _default_round_shift()
    default_shift_index = 0 if default_shift == "Morning" else 1
    rounds_shift = st.radio(
        "Rounds slot",
        options=shift_options,
        index=default_shift_index,
        horizontal=True,
        key=f"rounds_shift_{key_prefix}",
    )

    st.markdown("### Rounds PDF")
    pdf_bytes_key = f"rounds_pdf_bytes_{key_prefix}"
    pdf_file_key = f"rounds_pdf_file_{key_prefix}"
    if st.button(
        "Generate Rounds PDF",
        use_container_width=True,
        key=f"generate_rounds_pdf_{key_prefix}",
        disabled=has_discrepancy,
    ):
        try:
            pdf_bytes, pdf_path = generate_rounds_pdf(all_beds_output, shift=rounds_shift)
        except RuntimeError as error:
            st.error(f"Rounds PDF generation failed: {error}")
        except Exception as error:
            st.error(f"Rounds PDF generation failed: {error}")
        else:
            st.session_state[pdf_bytes_key] = pdf_bytes
            st.session_state[pdf_file_key] = pdf_path.name
            st.success(f"Rounds PDF generated: `{pdf_path}`")
    if has_discrepancy:
        st.caption("PDF generation is disabled until source/output discrepancies are resolved.")

    if st.session_state.get(pdf_bytes_key):
        st.download_button(
            "Download PDF (bed cards)",
            data=st.session_state[pdf_bytes_key],
            file_name=st.session_state.get(pdf_file_key, "ICU_Rounds.pdf"),
            mime="application/pdf",
            use_container_width=True,
            key=f"download_rounds_pdf_{key_prefix}",
        )


st.set_page_config(page_title="ICU Task Assistant", layout="wide")
_inject_dashboard_theme()
if ENABLE_ACCESS_GATE:
    try:
        _enforce_allowed_users()
    except Exception as error:
        st.sidebar.error("Authentication setup error. Running without access gate.")
        st.sidebar.code(str(error))

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase(
        resources_root=RESOURCES_ROOT,
        store_path=RESOURCE_STORE,
    )
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "advisor" not in st.session_state:
    st.session_state.advisor = ClinicalTaskAdvisor()
if "history_store_cache" not in st.session_state:
    st.session_state.history_store_cache = {}
if "active_tracker_unit" not in st.session_state:
    st.session_state.active_tracker_unit = ICU_UNITS[0]

knowledge_base: KnowledgeBase = st.session_state.knowledge_base
advisor: ClinicalTaskAdvisor = st.session_state.advisor

st.title("ICU Task Assistant")
st.caption("Round-ready bed cards with deterministic rules, round-over-round course tracking, and change detection.")
st.warning(
    "Clinical decision support only. A licensed clinician must verify every recommendation before use.",
    icon="⚠️",
)
_ensure_startup_index(knowledge_base)

st.sidebar.header("Settings")
default_unit = str(st.session_state.get("active_tracker_unit", ICU_UNITS[0]))
unit_index = ICU_UNITS.index(default_unit) if default_unit in ICU_UNITS else 0
selected_icu_unit = st.sidebar.selectbox(
    "Navigate ICU",
    options=ICU_UNITS,
    index=unit_index,
    key="selected_icu_unit",
)
if selected_icu_unit != st.session_state.get("active_tracker_unit"):
    _clear_tracker_state()
    st.session_state.active_tracker_unit = selected_icu_unit

history_store_cache: dict[str, ICUHistoryStore] = st.session_state.history_store_cache
if selected_icu_unit not in history_store_cache:
    history_store_cache[selected_icu_unit] = ICUHistoryStore(_history_db_path_for_unit(selected_icu_unit))
history_store: ICUHistoryStore = history_store_cache[selected_icu_unit]

st.sidebar.caption(f"Tracking unit: {selected_icu_unit}")
st.sidebar.caption(f"Local DB: `{_history_db_path_for_unit(selected_icu_unit).name}`")
with st.sidebar.expander("Unit history maintenance", expanded=False):
    confirm_reset = st.checkbox(
        f"Confirm clear {selected_icu_unit} history",
        value=False,
        key=f"confirm_clear_{_unit_slug(selected_icu_unit)}",
    )
    if st.button("Clear selected ICU history", use_container_width=True, key=f"clear_unit_{_unit_slug(selected_icu_unit)}"):
        if not confirm_reset:
            st.warning("Tick confirmation first to clear this ICU history.")
        else:
            try:
                history_store.clear_all()
            except Exception as error:
                st.error(f"Could not clear ICU history: {error}")
            else:
                _clear_tracker_state()
                st.session_state["tracker_context"] = f"Cleared saved history for {selected_icu_unit}."
                st.success(f"Cleared saved history for {selected_icu_unit}.")

model_name = st.sidebar.text_input("LLM model", value=os.getenv("OPENAI_MODEL", advisor.model))
top_k = st.sidebar.slider("Top chunks to retrieve", min_value=1, max_value=12, value=6)
use_only_neuro = st.sidebar.toggle("Use only Neuro resources", value=True)

if model_name != advisor.model:
    st.session_state.advisor = ClinicalTaskAdvisor(model=model_name)
    advisor = st.session_state.advisor

if advisor.llm_available:
    st.sidebar.success("OPENAI_API_KEY detected")
else:
    st.sidebar.warning("OPENAI_API_KEY missing - fallback mode")

if st.sidebar.button("Rebuild startup index", use_container_width=True):
    with st.spinner("Indexing resources/**/*.pdf ..."):
        knowledge_base.build_from_resources()
        st.session_state.index_ready = True
        st.session_state.startup_index_source = "manual_rebuild"
    st.sidebar.success("Index rebuilt from resources/**/*.pdf")

resources_tab, case_tab = st.tabs(["Indexed Resources", "Case Review"])

with resources_tab:
    st.subheader("Startup PDF Index")
    st.write(f"Root: `{RESOURCES_ROOT}`")
    fs_pdf_paths = sorted(path for path in RESOURCES_ROOT.rglob("*.pdf") if path.is_file())
    startup_source = str(st.session_state.get("startup_index_source", "")).strip()
    if startup_source == "store":
        st.caption("Startup index: loaded from saved local store.")
    elif startup_source == "auto_built":
        st.caption("Startup index: auto-built on first run for this deployment.")
    elif startup_source == "build_failed":
        st.caption("Startup index: automatic build failed. Use manual rebuild.")
    elif startup_source == "not_enabled":
        st.caption("Startup index: auto-build is disabled by configuration.")
    elif startup_source == "no_pdfs":
        st.caption("Startup index: no PDF files found under resources.")
    elif startup_source == "manual_rebuild":
        st.caption("Startup index: manually rebuilt.")
    st.write(f"PDF files detected on disk: **{len(fs_pdf_paths)}**")
    st.write(f"Indexed PDF files: **{knowledge_base.file_count()}**")
    st.write(f"Indexed chunks: **{knowledge_base.chunk_count()}**")
    if st.button("Build/Rebuild index now", use_container_width=True, key="build_index_in_tab"):
        with st.spinner("Indexing resources/**/*.pdf ..."):
            knowledge_base.build_from_resources()
            st.session_state.index_ready = True
            st.session_state.startup_index_source = "manual_rebuild"
        st.success("Index built successfully.")
    if knowledge_base.chunk_count() == 0:
        st.info("Index not built yet on this deployment. Click `Build/Rebuild index now` above.")
    indexed_files = knowledge_base.list_files()
    if not indexed_files:
        st.info("No indexed files yet. Build the index to enable retrieval.")
    else:
        for file_path in indexed_files:
            st.write(f"- `{file_path}`")

with case_tab:
    st.subheader("ICU Tracker (DOCX rounds dashboard)")
    st.caption(f"Current ICU navigation: **{selected_icu_unit}**")
    if not st.session_state.get("tracker_output"):
        latest_snapshot = history_store.get_latest_snapshot()
        if latest_snapshot is not None:
            _hydrate_tracker_from_snapshot(history_store, latest_snapshot, unit_name=selected_icu_unit)
    round_files = st.file_uploader(
        "Upload 1 or 2 DOCX round sheets",
        type=["docx"],
        accept_multiple_files=True,
        key="rounds_docx_upload",
    )

    if round_files:
        selected_files = list(round_files[:2])
        if len(round_files) > 2:
            st.warning("Only first 2 files are used for comparison in this version.")

        previous_upload_signature = str(st.session_state.get("round_upload_signature", ""))

        parsed_docs: list[dict[str, Any]] = []
        extraction_failed = False
        for upload in selected_files:
            upload_bytes = upload.getvalue()
            try:
                extracted = extract_text(upload.name, upload_bytes)
            except Exception as error:
                _show_extraction_error(upload.name, error)
                extraction_failed = True
                continue

            if not isinstance(extracted, dict):
                st.warning(f"`{upload.name}` is not a structured table DOCX.")
                parsed_docs.append({"name": upload.name, "rows": [], "debug_raw_rows": [], "bytes": upload_bytes})
                continue

            rows = extracted.get("table_rows", [])
            table_rows = rows if isinstance(rows, list) else []
            raw_rows = extracted.get("debug_raw_rows", [])
            debug_rows = raw_rows if isinstance(raw_rows, list) else []
            parse_warnings_raw = extracted.get("parse_warnings", [])
            parse_warnings = parse_warnings_raw if isinstance(parse_warnings_raw, list) else []
            selected_table_index = extracted.get("table_index")
            parsed_docs.append(
                {
                    "name": upload.name,
                    "rows": table_rows,
                    "debug_raw_rows": debug_rows,
                    "parse_warnings": [str(item) for item in parse_warnings],
                    "table_index": selected_table_index,
                    "bytes": upload_bytes,
                }
            )

        for doc in parsed_docs:
            st.write(f"`{doc['name']}` -> Beds parsed = {len(doc['rows'])}")
            if doc.get("table_index") is not None:
                st.caption(f"Selected table index: {doc['table_index']}")
            if doc.get("parse_warnings"):
                with st.expander(f"Parser warnings: {doc['name']}", expanded=False):
                    for warning in doc["parse_warnings"]:
                        st.markdown(f"- {warning}")

        if not extraction_failed and parsed_docs:
            upload_signature = "|".join(
                f"{doc.get('name', '')}:{len(bytes(doc.get('bytes', b'')))}:{hash_payload(bytes(doc.get('bytes', b'')))[:12]}"
                for doc in parsed_docs
            )
            is_new_round_upload = previous_upload_signature != upload_signature
            st.session_state["round_upload_signature"] = upload_signature

            current_doc = parsed_docs[0]
            tracker_context = f"Current round source: `{current_doc['name']}` | Unit: {selected_icu_unit}."

            if len(parsed_docs) == 2:
                file_names = [str(doc["name"]) for doc in parsed_docs]
                older_idx, newer_idx, reason = infer_round_file_order(file_names)
                older_doc = parsed_docs[older_idx]
                newer_doc = parsed_docs[newer_idx]
                current_doc = newer_doc
                tracker_context = (
                    f"Detected older `{older_doc['name']}` and newer `{newer_doc['name']}` "
                    f"(inferred by {reason}). Newer file is used as current round for {selected_icu_unit}."
                )
                st.info(tracker_context)
                with st.expander("Debug parsed records", expanded=False):
                    st.markdown(f"**Older:** `{older_doc['name']}`")
                    st.json(older_doc["rows"][:2])
                    st.markdown(f"**Newer (used):** `{newer_doc['name']}`")
                    st.json(newer_doc["rows"][:2])
            else:
                with st.expander("Debug parsed records", expanded=False):
                    st.json(current_doc["rows"][:2])

            missing_pid_count = sum(1 for row in current_doc.get("rows", []) if not str(row.get("patient_id", "")).strip())
            if missing_pid_count:
                st.caption(
                    f"Rows without Patient ID: {missing_pid_count}. "
                    "Course matching for these rows uses bed fallback only."
                )
            inferred_round_date, inferred_round_shift = _infer_round_date_shift_from_filename(
                str(current_doc.get("name", ""))
            )
            st.caption(
                f"Auto-save target for this upload: {selected_icu_unit} | "
                f"{inferred_round_date} ({inferred_round_shift}). "
                "Re-uploading same date+shift updates that round."
            )

            def _save_current_upload(auto_triggered: bool = False) -> None:
                if not current_doc.get("rows"):
                    st.error("No parsed bed rows found. Upload a valid rounds table DOCX.")
                    return
                current_output = _safe_build_all_beds(list(current_doc["rows"]))
                if not current_output:
                    return
                try:
                    tracker_round_date, tracker_round_shift = _infer_round_date_shift_from_filename(
                        str(current_doc.get("name", ""))
                    )
                    snapshot_id = history_store.save_snapshot(
                        snapshot_date=tracker_round_date,
                        shift=tracker_round_shift,
                        file_hash=hash_payload(bytes(current_doc.get("bytes", b""))),
                        table_rows=list(current_doc["rows"]),
                    )
                    current_snapshot = history_store.get_snapshot(snapshot_id)
                    previous_snapshot = history_store.get_previous_snapshot(snapshot_id)
                    current_state_rows = history_store.get_snapshot_rows(snapshot_id)
                    previous_state_rows = (
                        history_store.get_snapshot_rows(int(previous_snapshot["snapshot_id"]))
                        if previous_snapshot
                        else []
                    )
                    changes = compute_snapshot_changes(current_state_rows, previous_state_rows)
                    _annotate_output_with_changes(current_output, changes)
                except Exception as error:
                    st.error(f"Round save or delta computation failed: {error}")
                    with st.expander("Tracker error details", expanded=False):
                        st.exception(error)
                    return

                st.session_state["tracker_output"] = current_output
                st.session_state["tracker_source_rows"] = list(current_doc["rows"])
                st.session_state["tracker_changes"] = changes
                st.session_state["tracker_current_rows"] = current_state_rows
                st.session_state["tracker_current_snapshot"] = current_snapshot
                st.session_state["tracker_previous_snapshot"] = previous_snapshot
                st.session_state["tracker_context"] = tracker_context
                st.session_state["tracker_autosaved_signature"] = upload_signature
                if auto_triggered:
                    st.success(
                        f"Uploaded round auto-saved for {selected_icu_unit}. "
                        f"Current round: {_snapshot_label(current_snapshot)}"
                    )
                else:
                    st.success(
                        f"Dashboard refreshed for {selected_icu_unit}. "
                        f"Current round: {_snapshot_label(current_snapshot)}"
                    )

            last_autosaved_signature = str(st.session_state.get("tracker_autosaved_signature", ""))
            if is_new_round_upload and upload_signature and last_autosaved_signature != upload_signature:
                _save_current_upload(auto_triggered=True)

            if st.button(
                "Refresh dashboard from current upload",
                type="primary",
                use_container_width=True,
                key="save_snapshot_generate_tracker",
            ):
                _save_current_upload(auto_triggered=False)

    _render_tracker_views(
        history_store=history_store,
        knowledge_base=knowledge_base,
        selected_icu_unit=selected_icu_unit,
    )

    st.divider()
    st.subheader("Patient Note Input")
    patient_file = st.file_uploader(
        "Patient document",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=False,
        key="single_note_uploader",
    )
    pasted_text = st.text_area("Paste patient note", height=220)

    extracted_raw_text = ""
    extracted_table_rows: list[dict[str, str]] = []
    debug_raw_rows: list[list[str]] = []
    extracted_parse_warnings: list[str] = []
    extracted_table_index: int | None = None

    if patient_file is not None:
        file_signature = f"{patient_file.name}:{getattr(patient_file, 'size', 0)}"
        if st.session_state.get("all_beds_file_signature") != file_signature:
            st.session_state.pop("all_beds_output", None)
            st.session_state.pop("all_beds_source_rows", None)
            st.session_state["all_beds_file_signature"] = file_signature

        try:
            extracted = extract_text(patient_file.name, patient_file.getvalue())
        except Exception as error:
            _show_extraction_error(patient_file.name, error)
            extracted = None

        if isinstance(extracted, dict):
            extracted_raw_text = str(extracted.get("raw_text", ""))
            rows = extracted.get("table_rows", [])
            extracted_table_rows = rows if isinstance(rows, list) else []
            raw_rows = extracted.get("debug_raw_rows", [])
            debug_raw_rows = raw_rows if isinstance(raw_rows, list) else []
            parse_warnings_raw = extracted.get("parse_warnings", [])
            if isinstance(parse_warnings_raw, list):
                extracted_parse_warnings = [str(item) for item in parse_warnings_raw]
            table_index_raw = extracted.get("table_index")
            if isinstance(table_index_raw, int):
                extracted_table_index = table_index_raw
        elif isinstance(extracted, str):
            extracted_raw_text = extracted

    if extracted_table_rows and not round_files:
        st.info("Structured DOCX bed table detected. Single-note path is disabled for this file.")
        st.write(f"Beds parsed = {len(extracted_table_rows)}")
        if extracted_table_index is not None:
            st.caption(f"Selected table index: {extracted_table_index}")
        if extracted_parse_warnings:
            with st.expander("Parser warnings", expanded=False):
                for warning in extracted_parse_warnings:
                    st.markdown(f"- {warning}")

        with st.expander("Debug parsed records", expanded=False):
            st.json(extracted_table_rows[:2])

        if st.button(
            "Generate output for ALL beds",
            type="primary",
            use_container_width=True,
            key="generate_all_beds_single_note",
        ):
            single_note_output = _safe_build_all_beds(extracted_table_rows)
            if single_note_output:
                st.session_state.all_beds_output = single_note_output
                st.session_state.all_beds_source_rows = list(extracted_table_rows)

        all_beds_output = st.session_state.get("all_beds_output", [])
        source_rows = st.session_state.get("all_beds_source_rows", [])
        if all_beds_output:
            _render_all_beds_panel(all_beds_output, key_prefix="single_note", source_rows=source_rows)
    elif patient_file is not None and patient_file.name.lower().endswith(".docx") and not round_files:
        st.write("Beds parsed = 0")
        with st.expander("Debug parsed records", expanded=False):
            st.json([])
        with st.expander("Debug raw table rows (first 5)", expanded=False):
            st.json(debug_raw_rows[:5])
    elif not round_files:
        if st.button("Analyze Case", type="primary", use_container_width=True, key="analyze_single_case"):
            combined_text = pasted_text.strip()
            if extracted_raw_text:
                combined_text = f"{combined_text}\n\n{extracted_raw_text}".strip() if combined_text else extracted_raw_text

            if not combined_text:
                st.error("Provide patient text or upload a patient file.")
            else:
                if knowledge_base.chunk_count() == 0:
                    loaded = knowledge_base.load_from_store()
                    if loaded:
                        st.session_state.index_ready = True
                    else:
                        st.error("Knowledge index is empty. Click `Rebuild startup index` in the sidebar first.")
                        combined_text = ""

                if combined_text:
                    retrieved = knowledge_base.retrieve(
                        query=combined_text,
                        top_k=top_k,
                        only_neuro=use_only_neuro,
                    )
                    matched_chunks = [chunk for chunk, _ in retrieved]

                    with st.spinner("Generating recommendations..."):
                        result_markdown = advisor.analyze(combined_text, matched_chunks)

                    st.markdown(result_markdown)
                    st.subheader("Top sources used")
                    if not retrieved:
                        st.info("No matching indexed chunks found for this note.")
                    else:
                        for chunk, score in retrieved:
                            st.markdown(f"- `{chunk.file_name}` | page **{chunk.page_number}** | score `{score:.3f}`")
