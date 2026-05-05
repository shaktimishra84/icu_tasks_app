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
from src.rmo_pdf import parse_combined_rmo_pdf, parse_combined_rmo_text
from src.tele_rounds import generate_whatsapp_round_pdf, parse_docx_patient_blocks, process_icu_report


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
PRIMARY_ICU_UNIT = os.getenv("PRIMARY_ICU_UNIT", "MICU").strip() or "MICU"
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


def _uploaded_file_fingerprint(uploaded_file: Any) -> str:
    name = str(getattr(uploaded_file, "name", "")).strip()
    size = int(getattr(uploaded_file, "size", 0) or 0)
    try:
        data = uploaded_file.getvalue()
    except Exception:
        data = b""
    digest = hash_payload(data)[:12] if data else "nohash"
    return f"{name}:{size}:{digest}"


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
        return " | ".join(selected[:4])
    return " | ".join(parts[:2]) if len(parts) > 1 else parts[0]


def _inject_dashboard_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #0c0e19;
            --bg-card: #141828;
            --bg-card-h: #1b2038;
            --bg-sidebar: #090b14;
            --bg-header: #0d0f1c;
            --bg-input: #1a1e30;
            --border: rgba(255,255,255,0.07);
            --border-s: rgba(255,255,255,0.13);
            --text: #dde3f2;
            --text-2: #8892ae;
            --text-3: #4a5270;
            --accent: #3b82f6;
            --critical: #ef4444;
            --guarded: #f59e0b;
            --serious: #d4b257;
            --deceased: #6b7280;
            --radius: 10px;
        }

        .stApp, [data-testid="stAppViewContainer"] {
            background: var(--bg) !important;
            color: var(--text) !important;
            font-family: "IBM Plex Sans", system-ui, sans-serif !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stSidebar"] {
            background: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--text) !important;
            font-family: "IBM Plex Sans", system-ui, sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6, p, li, label, div, span {
            color: var(--text);
            font-family: "IBM Plex Sans", system-ui, sans-serif;
        }

        [data-baseweb="input"] input,
        [data-baseweb="select"] input,
        textarea,
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stDateInput input {
            background: var(--bg-input) !important;
            border-color: var(--border-s) !important;
            color: var(--text) !important;
        }

        [data-baseweb="select"] svg,
        .stSelectbox svg,
        .stDateInput svg {
            fill: var(--text-2) !important;
        }

        .stButton button,
        .stDownloadButton button {
            width: 100%;
            border-radius: 8px !important;
            border: 1px solid var(--border-s) !important;
            background: var(--bg-card) !important;
            color: var(--text) !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            font-family: "IBM Plex Sans", system-ui, sans-serif !important;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            background: var(--bg-card-h) !important;
            border-color: var(--accent) !important;
        }

        .stAlert {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-s) !important;
            color: var(--text) !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            box-shadow: none !important;
        }

        [data-testid="stTabs"] [role="tablist"] {
            border-bottom: 1px solid var(--border);
            gap: 2px;
        }

        [data-testid="stTabs"] [role="tab"] {
            border: none !important;
            border-radius: 0 !important;
            background: transparent !important;
            color: var(--text-2) !important;
            font-weight: 600;
            padding: 10px 6px 12px 6px;
            margin-right: 20px;
        }

        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: var(--text) !important;
            box-shadow: inset 0 -2px 0 0 var(--critical);
        }

        .icu-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 14px;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--bg-header);
            margin-bottom: 12px;
        }

        .icu-topbar-title {
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.01em;
            color: var(--text);
        }

        .icu-topbar-subtitle {
            font-size: 12px;
            color: var(--text-2);
            margin-top: 2px;
        }

        .icu-unit-chip {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            border: 1px solid var(--border-s);
            font-size: 11px;
            font-weight: 600;
            color: var(--text);
            background: var(--bg-card);
            white-space: nowrap;
        }

        .icu-warning-banner {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid rgba(245, 158, 11, 0.28);
            background: rgba(245, 158, 11, 0.12);
            color: #fcd34d;
            font-size: 13px;
            margin-bottom: 12px;
        }

        .icu-metric-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(120px, 1fr));
            gap: 10px;
            margin: 6px 0 16px 0;
        }

        .icu-metric-card {
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--bg-card);
            padding: 10px 12px;
        }

        .icu-metric-label {
            font-size: 11px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--text-3);
        }

        .icu-metric-value {
            font-size: 24px;
            line-height: 1.1;
            font-weight: 700;
            margin-top: 3px;
            color: var(--text);
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

        .icu-chip.critical { background: var(--critical); }
        .icu-chip.sick { background: var(--guarded); }
        .icu-chip.serious { background: var(--serious); color: #111827; }
        .icu-chip.deceased { background: var(--deceased); }
        .icu-chip.other { background: #3b82f6; }

        .icu-pill {
            display: inline-block;
            background: #1e3a8a;
            color: #dbeafe;
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 11px;
            margin-left: 6px;
        }

        .stCaption, .stMarkdown p, .stMarkdown li {
            color: var(--text-2) !important;
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
            .icu-topbar {
                flex-direction: column;
                align-items: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_app_header(selected_icu_unit: str) -> None:
    st.markdown(
        (
            "<div class='icu-topbar'>"
            "<div>"
            "<div class='icu-topbar-title'>ICU Task Assistant</div>"
            "<div class='icu-topbar-subtitle'>Round-ready bed cards · change detection · course tracking</div>"
            "</div>"
            f"<span class='icu-unit-chip'>{escape(selected_icu_unit)}</span>"
            "</div>"
            "<div class='icu-warning-banner'>"
            "⚠ Clinical decision support only. A licensed clinician must verify every recommendation before use."
            "</div>"
        ),
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


_GENERIC_ACTION_SNIPPETS = [
    "reassess abcs",
    "reassess airway",
    "continue supportive care",
    "monitor closely",
    "clinical correlation",
]

_PROCEDURE_HINTS = [
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

_WATCH_HINTS = [
    "worsening",
    "hypotension",
    "shock",
    "low gcs",
    "high peak pressure",
    "rising",
    "drop",
    "decline",
    "seizure",
]

_DISCHARGE_TRANSFER_HINTS = [
    "for discharge",
    "discharge planned",
    "for transfer",
    "transfer to ward",
    "shift to ward",
    "step down",
]


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority.lower(), 3)


def _is_generic_action(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return True
    return any(snippet in normalized for snippet in _GENERIC_ACTION_SNIPPETS)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        key = _normalized_key(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _split_priority_items(text: Any) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for line in str(text or "").splitlines():
        cleaned = re.sub(r"^[\-\*\u2022]+\s*", "", line).strip()
        if not cleaned:
            continue
        match = re.match(r"(?i)(high|medium|low)\s*:\s*(.+)", cleaned)
        if match:
            priority = match.group(1).title()
            item = match.group(2).strip()
        else:
            priority = "Medium"
            item = cleaned
        if item and not _is_generic_action(item):
            parsed.append((priority, item))
    parsed.sort(key=lambda pair: (_priority_rank(pair[0]), pair[1].lower()))
    return parsed


def _is_new_admission(record: dict[str, Any]) -> bool:
    trend = str(record.get("Round trend", "")).strip().upper()
    if trend == "NEW ADMISSION":
        return True
    combined = " ".join(
        [
            str(record.get("Diagnosis", "")),
            str(record.get("_raw_new_issues", "")),
            str(record.get("_raw_actions_done", "")),
            str(record.get("_raw_plan_next_12h", "")),
        ]
    ).lower()
    return "new admission" in combined


def _is_procedure_patient(record: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(record.get("_raw_plan_next_12h", "")),
            str(record.get("Pending (verbatim)", "")),
            str(record.get("_raw_actions_done", "")),
        ]
    ).lower()
    return any(hint in text for hint in _PROCEDURE_HINTS)


def _is_discharge_transfer_candidate(record: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(record.get("Status", "")),
            str(record.get("Diagnosis", "")),
            str(record.get("_raw_plan_next_12h", "")),
        ]
    ).lower()
    return any(hint in text for hint in _DISCHARGE_TRANSFER_HINTS)


def _is_closed_case(record: dict[str, Any]) -> bool:
    if str(record.get("_status_group", "")).upper() == "DECEASED":
        return True
    flags = [str(flag).lower() for flag in (record.get("_flags", []) or [])]
    if any("dama" in flag for flag in flags):
        return True
    return _is_discharge_transfer_candidate(record)


def _collect_now_items(record: dict[str, Any], max_items: int = 3) -> list[str]:
    items: list[str] = []
    trend = str(record.get("Round trend", "")).strip()
    deteriorated_reason = str(record.get("Deterioration reasons", "")).strip()
    tracker_lines = [str(line).strip() for line in (record.get("_tracker_summary_lines", []) or []) if str(line).strip()]
    new_issues = _split_display_items(record.get("_raw_new_issues", ""))
    actions_done = _split_display_items(record.get("_raw_actions_done", ""))

    if _is_new_admission(record):
        items.append("New admission")
    if str(record.get("Deterioration since last round", "")).upper() == "YES":
        if deteriorated_reason:
            items.append(f"Deteriorated: {deteriorated_reason}")
        else:
            items.append("Deteriorated since previous round")
    elif trend and trend.upper() not in {"STABLE", "NEW ADMISSION"}:
        items.append(f"Trend: {trend}")

    items.extend(new_issues[:2])
    items.extend(actions_done[:1])
    items.extend(tracker_lines[:2])
    filtered = [item for item in _dedupe_keep_order(items) if not _is_generic_action(item)]
    return filtered[:max_items]


def _collect_pending_items(record: dict[str, Any], max_items: int = 3) -> list[str]:
    pending = _split_display_items(record.get("Pending (verbatim)", ""))
    unresolved = _split_display_items(record.get("_unresolved_pending", ""))
    merged = _dedupe_keep_order(pending + unresolved)
    return merged[:max_items]


def _collect_today_items(record: dict[str, Any], max_items: int = 3) -> list[str]:
    tasks: list[str] = []
    plan_items = _split_display_items(record.get("_raw_plan_next_12h", ""))
    tasks.extend([item for item in plan_items if not _is_generic_action(item)])

    missing_fields = [
        "Missing Tests",
        "Missing Imaging",
        "Missing Consults",
        "Care checks (deterministic)",
    ]
    for field_name in missing_fields:
        for priority, item in _split_priority_items(record.get(field_name, "")):
            tasks.append(f"{priority}: {item}")

    pending_keys = {_normalized_key(item) for item in _collect_pending_items(record, max_items=20)}
    filtered: list[str] = []
    for item in _dedupe_keep_order(tasks):
        key = _normalized_key(item)
        key_without_priority = _normalized_key(re.sub(r"^(high|medium|low)\s*:\s*", "", item, flags=re.IGNORECASE))
        if key in pending_keys or key_without_priority in pending_keys:
            continue
        filtered.append(item)
    return filtered[:max_items]


def _collect_watch_items(record: dict[str, Any], max_items: int = 2) -> list[str]:
    watch: list[str] = []
    reasons = _split_display_items(record.get("Deterioration reasons", ""))
    for reason in reasons:
        watch.append(reason)

    flags = [str(flag).strip() for flag in (record.get("_flags", []) or []) if str(flag).strip()]
    watch.extend(flags)

    new_issues = _split_display_items(record.get("_raw_new_issues", ""))
    for issue in new_issues:
        if any(hint in issue.lower() for hint in _WATCH_HINTS):
            watch.append(issue)

    key_labs = _split_display_items(record.get("Key labs/imaging (1 line)", ""))
    for lab in key_labs:
        lower_lab = lab.lower()
        if any(hint in lower_lab for hint in ["lactate", "creat", "potassium", "sodium", "platelet", "abg"]):
            watch.append(lab)
    return _dedupe_keep_order(watch)[:max_items]


def _triage_rank(record: dict[str, Any]) -> int:
    status_group = str(record.get("_status_group", "OTHER")).upper()
    if _is_closed_case(record):
        return 5
    if status_group == "CRITICAL" or bool(record.get("_is_mv")) or bool(record.get("_is_vaso")):
        return 0
    if _is_new_admission(record):
        return 1
    if str(record.get("Deterioration since last round", "")).upper() == "YES":
        return 2
    if _is_procedure_patient(record):
        return 3
    return 4


def _triage_bucket(record: dict[str, Any]) -> str:
    rank = _triage_rank(record)
    if rank == 5:
        return "CLOSED"
    if rank <= 2:
        return "RED"
    if rank == 3:
        return "AMBER"
    return "GREEN"


def _triage_sort_key(record: dict[str, Any]) -> tuple[int, int, tuple[int, int | str], str]:
    status_order = {"CRITICAL": 0, "SICK": 1, "SERIOUS": 2, "OTHER": 3, "DECEASED": 4}
    status_group = str(record.get("_status_group", "OTHER")).upper()
    return (
        _triage_rank(record),
        status_order.get(status_group, 9),
        _bed_sort_value(record.get("Bed", "")),
        str(record.get("Patient ID", "")),
    )


def _acuity_label(record: dict[str, Any]) -> str:
    bucket = _triage_bucket(record)
    if bucket == "RED":
        return "Red"
    if bucket == "AMBER":
        return "Amber"
    if bucket == "GREEN":
        return "Green"
    return "Grey"


def _supports_schema(record: dict[str, Any]) -> str:
    supports: list[str] = []
    if bool(record.get("_is_mv")):
        supports.append("MV")
    if bool(record.get("_is_niv")):
        supports.append("NIV")
    if bool(record.get("_is_vaso")):
        supports.append("Vasopressor")
    if bool(record.get("_is_rrt")):
        supports.append("Dialysis")
    if not supports:
        return "-"
    return " / ".join(supports)


def _change_since_round(record: dict[str, Any]) -> str:
    status_group = str(record.get("_status_group", "")).upper()
    if status_group == "DECEASED":
        return "Declared deceased"
    if _is_new_admission(record):
        return "New admission"
    if str(record.get("Deterioration since last round", "")).upper() == "YES":
        reason = str(record.get("Deterioration reasons", "")).strip()
        return f"Deteriorated: {reason}" if reason else "Deteriorated"
    trend = str(record.get("Round trend", "")).strip()
    if trend and trend.upper() not in {"STABLE", "NEW ADMISSION"}:
        return trend
    now_items = _collect_now_items(record, max_items=1)
    return now_items[0] if now_items else "No major change"


def _must_do_line(record: dict[str, Any]) -> str:
    if str(record.get("_status_group", "")).upper() == "DECEASED":
        return "-"
    tasks = _collect_today_items(record, max_items=2)
    return " | ".join(tasks) if tasks else "-"


def _pending_line(record: dict[str, Any]) -> str:
    pending = _collect_pending_items(record, max_items=2)
    return " | ".join(pending) if pending else "-"


def _abnormal_lab_item(item: str) -> bool:
    text = str(item or "").strip()
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


def _key_labs_top3(record: dict[str, Any]) -> str:
    raw = str(record.get("Key labs/imaging (1 line)", "")).strip()
    if not raw:
        return "-"
    parts = _split_display_items(raw)
    abnormal = [part for part in parts if _abnormal_lab_item(part)]
    selected = abnormal[:3]
    if not selected:
        return "-"
    return " | ".join(selected)


def _escalation_trigger(record: dict[str, Any]) -> str:
    if str(record.get("_status_group", "")).upper() == "DECEASED":
        return "-"
    watch = _collect_watch_items(record, max_items=1)
    return watch[0] if watch else "-"


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
        acuity = _acuity_label(record)
        diagnosis = _short_text(record.get("Diagnosis", "-"), limit=95) or "-"
        supports = _supports_schema(record)
        change = _change_since_round(record)
        must_do = _must_do_line(record)
        pending = _pending_line(record)
        key_labs = _key_labs_top3(record)
        escalation = _escalation_trigger(record)

        def _schema_line(label: str, value: str) -> None:
            st.markdown(
                (
                    "<div style='font-size:13.5px;line-height:1.3;margin:1px 0;'>"
                    f"<strong>{escape(label)}:</strong> {escape(value or '-')}</div>"
                ),
                unsafe_allow_html=True,
            )

        _schema_line("Acuity / Supports", f"{acuity} / {supports}")
        _schema_line("Diagnosis", diagnosis)
        _schema_line("Change since last round", change)
        _schema_line("Must do before evening", must_do)
        _schema_line("Pending", pending)
        _schema_line("Key labs (3) / Escalation", f"{key_labs} / {escalation}")

        if show_course_button and patient_key:
            if st.button("View course", key=f"view_course_{key_prefix}_{patient_key}"):
                st.session_state[f"selected_patient_key_{key_prefix}"] = patient_key
                st.caption("Patient selected for course view and correction.")

    status_group = str(record.get("_status_group", "OTHER"))
    status_color = _status_color(status_group)
    support_badges = _support_badges(record)
    patient_key = str(record.get("_patient_key", "")).strip() or patient_key_from_output_row(record)
    pending_count = len(_split_display_items(record.get("Pending (verbatim)", "")))
    must_do_preview = _must_do_line(record)

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
                f"Must do: {_short_text(must_do_preview, limit=52)} | Pending: {pending_count}"
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
    raw_text = str(text or "").replace("\r", "\n")
    raw_text = re.sub(r"(?:(?<=^)|(?<=\s))\d+[.)](?=\s*[A-Za-z])", "\n", raw_text)
    for chunk in re.split(r"\n|;|\s\|\s", raw_text):
        cleaned = re.sub(r"^[\-\*\u2022]+\s*", "", chunk).strip()
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned).strip()
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


def _clear_rmo_pdf_state() -> None:
    for key in [
        "rmo_pdf_signature",
        "rmo_pdf_docs",
        "rmo_pdf_rows",
        "rmo_pdf_previous_rows",
        "rmo_pdf_changes",
        "rmo_pdf_older_label",
        "rmo_pdf_newer_label",
        "rmo_pdf_compare_context",
        "rmo_pdf_output",
        "rmo_pdf_autobuilt_signature",
        "rmo_pdf_loaded_snapshot_id",
        "rmo_pdf_restore_context",
        "rounds_pdf_bytes_rmo_pdf",
        "rounds_pdf_file_rmo_pdf",
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


def _parse_rounds_document(filename: str, data: bytes) -> dict[str, Any]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_combined_rmo_pdf(filename, data)

    if ext != ".docx":
        raise ExtractionError("Rounds upload supports PDF and DOCX only.")

    try:
        block_rows, block_warnings = parse_docx_patient_blocks(data)
    except Exception as error:  # noqa: BLE001 - keep legacy DOCX extraction available for malformed files.
        block_rows = []
        block_warnings = [f"Patient-block DOCX parser skipped: {error}"]
    if block_rows:
        return {
            "table_rows": block_rows,
            "blocks_detected": len(block_rows),
            "warnings": block_warnings,
            "debug_blocks": block_rows[:2],
        }

    extracted = extract_text(filename, data)
    if not isinstance(extracted, dict):
        raise ExtractionError("Unexpected DOCX extraction result.")

    raw_text = str(extracted.get("raw_text", "") or "")
    parsed_from_sections = parse_combined_rmo_text(raw_text) if raw_text else {}
    section_rows_raw = parsed_from_sections.get("table_rows", []) if isinstance(parsed_from_sections, dict) else []
    if isinstance(section_rows_raw, list) and section_rows_raw:
        return {
            "table_rows": section_rows_raw,
            "blocks_detected": int(parsed_from_sections.get("blocks_detected", 0) or 0),
            "warnings": list(parsed_from_sections.get("warnings", []) or []),
            "debug_blocks": list(parsed_from_sections.get("debug_blocks", []) or []),
        }

    fallback_rows = extracted.get("table_rows", [])
    table_rows = fallback_rows if isinstance(fallback_rows, list) else []
    warnings = block_warnings + list(extracted.get("parse_warnings", []) or [])
    if not table_rows:
        warnings.append("No patient table rows parsed from DOCX.")
    else:
        warnings.append("Used DOCX table parser fallback; section-level RMO blocks were not detected.")

    return {
        "table_rows": table_rows,
        "blocks_detected": 0,
        "warnings": warnings,
        "debug_blocks": list(extracted.get("debug_raw_rows", []) or [])[:2],
    }


def _persist_rmo_docs_to_history(
    history_store: ICUHistoryStore,
    parsed_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    for doc in parsed_docs:
        rows = doc.get("rows", [])
        if not isinstance(rows, list) or not rows:
            continue
        snapshot_date = str(doc.get("snapshot_date", "")).strip()
        snapshot_shift = str(doc.get("snapshot_shift", "")).strip()
        file_hash = str(doc.get("file_hash", "")).strip() or "pdf-upload"
        if not snapshot_date:
            snapshot_date = date.today().isoformat()
        if not snapshot_shift:
            snapshot_shift = _default_round_shift()
        try:
            snapshot_id = history_store.save_snapshot(
                snapshot_date=snapshot_date,
                shift=snapshot_shift,
                file_hash=file_hash,
                table_rows=rows,
            )
        except Exception as error:
            doc["save_error"] = str(error)
            continue
        doc["saved_snapshot_id"] = snapshot_id
        snapshot = history_store.get_snapshot(snapshot_id)
        if snapshot:
            saved.append(snapshot)
    return saved


def _hydrate_rmo_from_snapshot(
    history_store: ICUHistoryStore,
    snapshot: dict[str, Any],
    *,
    context_prefix: str = "Loaded saved round",
) -> bool:
    snapshot_id = int(snapshot.get("snapshot_id", 0))
    if snapshot_id <= 0:
        return False

    current_state_rows = history_store.get_snapshot_rows(snapshot_id)
    if not current_state_rows:
        return False

    current_rows = _source_rows_from_state_rows(current_state_rows)
    output_rows = _safe_build_all_beds(current_rows)
    if not output_rows:
        return False

    previous_snapshot = history_store.get_previous_snapshot(snapshot_id)
    previous_state_rows = (
        history_store.get_snapshot_rows(int(previous_snapshot["snapshot_id"]))
        if previous_snapshot
        else []
    )
    previous_rows = _source_rows_from_state_rows(previous_state_rows)
    changes = compute_snapshot_changes(current_state_rows, previous_state_rows)
    _annotate_output_with_changes(output_rows, changes)

    st.session_state["rmo_pdf_rows"] = current_rows
    st.session_state["rmo_pdf_previous_rows"] = previous_rows
    st.session_state["rmo_pdf_changes"] = changes
    st.session_state["rmo_pdf_output"] = output_rows
    st.session_state["rmo_pdf_signature"] = f"snapshot:{snapshot_id}"
    st.session_state["rmo_pdf_autobuilt_signature"] = f"snapshot:{snapshot_id}"
    st.session_state["rmo_pdf_loaded_snapshot_id"] = snapshot_id
    st.session_state["rmo_pdf_older_label"] = _snapshot_label(previous_snapshot) if previous_snapshot else ""
    st.session_state["rmo_pdf_newer_label"] = _snapshot_label(snapshot)
    st.session_state["rmo_pdf_compare_context"] = (
        f"{context_prefix}: {_snapshot_label(snapshot)}"
    )
    st.session_state["rmo_pdf_restore_context"] = (
        f"{context_prefix}: {_snapshot_label(snapshot)} from local {history_store.db_path.name}"
    )
    return True


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


def _render_priority_attention_table(records: list[dict[str, Any]]) -> None:
    if not records:
        return

    rows: list[dict[str, Any]] = []
    for record in records:
        trend = str(record.get("Round trend", "")).strip().upper() or "STABLE"
        has_pending = bool(_collect_pending_items(record, max_items=2))
        is_critical = str(record.get("_status_group", "")).upper() == "CRITICAL"
        if not (trend == "DETERIORATED" or is_critical or has_pending):
            continue
        rank = 0 if trend == "DETERIORATED" else 1 if is_critical else 2
        rows.append(
            {
                "_rank": rank,
                "Bed": str(record.get("Bed", "")).strip(),
                "Patient ID": str(record.get("Patient ID", "")).strip(),
                "Acuity": _acuity_label(record),
                "Supports": _supports_schema(record),
                "Trend": str(record.get("Round trend", "")).strip() or "STABLE",
                "Change / Deterioration": _change_since_round(record),
                "Must do before evening": _must_do_line(record),
                "Pending": _pending_line(record),
            }
        )

    if not rows:
        st.caption("No priority-attention beds in current view.")
        return

    rows.sort(key=lambda row: (int(row.get("_rank", 9)), _bed_sort_value(row.get("Bed", "")), str(row.get("Patient ID", ""))))
    display_rows = [{k: v for k, v in row.items() if k != "_rank"} for row in rows]

    if pd is None:
        st.table(display_rows)
        return

    dataframe = pd.DataFrame(display_rows)

    def _row_style(row: Any) -> list[str]:
        trend = str(row.get("Trend", "")).upper()
        if trend == "DETERIORATED":
            return ["background-color: #fef2f2; color: #7f1d1d; font-weight: 600;" for _ in row]
        if str(row.get("Acuity", "")).lower() == "red":
            return ["background-color: #fffbeb; color: #92400e;" for _ in row]
        return ["" for _ in row]

    st.dataframe(
        dataframe.style.apply(_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def _state_rows_from_source_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state_rows: list[dict[str, Any]] = []
    for row in source_rows:
        patient_key = patient_key_from_source_row(row)
        if not patient_key:
            continue
        state_rows.append(
            {
                "patient_key": patient_key,
                "patient_id": str(row.get("patient_id", "")).strip(),
                "bed": str(row.get("bed", "")).strip(),
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
    return state_rows


def _render_round_comparison_table(
    changes: list[dict[str, Any]],
    *,
    older_label: str,
    newer_label: str,
) -> None:
    if not changes:
        return

    st.markdown("### Round Comparison Table")
    st.caption(f"Comparing `{older_label}` -> `{newer_label}`")

    rows: list[dict[str, Any]] = []
    for change in sorted(changes, key=lambda item: (_bed_sort_value(item.get("bed", "")), str(item.get("patient_id", "")))):
        supports_added = list(change.get("supports_added", []) or [])
        supports_removed = list(change.get("supports_removed", []) or [])
        supports_delta_parts: list[str] = []
        if supports_added:
            supports_delta_parts.append("+" + ", ".join(supports_added))
        if supports_removed:
            supports_delta_parts.append("-" + ", ".join(supports_removed))
        supports_delta = " ; ".join(supports_delta_parts) if supports_delta_parts else "-"

        pending_new = list(change.get("pending_new", []) or [])
        pending_resolved = list(change.get("pending_resolved", []) or [])
        pending_delta_parts: list[str] = []
        if pending_new:
            pending_delta_parts.append("new: " + " | ".join(pending_new[:2]))
        if pending_resolved:
            pending_delta_parts.append("resolved: " + " | ".join(pending_resolved[:2]))
        pending_delta = " ; ".join(pending_delta_parts) if pending_delta_parts else "-"

        previous_status = str(change.get("previous_status_group", "")).strip()
        current_status = str(change.get("current_status_group", "")).strip() or "-"
        status_change = f"{previous_status} -> {current_status}" if previous_status else f"NEW -> {current_status}"

        key_change = " ; ".join(str(line) for line in list(change.get("summary_lines", []))[:2]) or "-"

        rows.append(
            {
                "Bed": str(change.get("bed", "")).strip(),
                "Patient ID": str(change.get("patient_id", "")).strip(),
                "Trend": str(change.get("trend", "")).strip(),
                "Status change": status_change,
                "Supports delta": supports_delta,
                "Pending delta": pending_delta,
                "Key change": key_change,
            }
        )

    if not rows:
        return

    if pd is None:
        st.table(rows)
        return

    dataframe = pd.DataFrame(rows)

    def _row_style(row: Any) -> list[str]:
        trend = str(row.get("Trend", "")).upper()
        if trend == "DETERIORATED":
            return ["background-color: #fee2e2; color: #7f1d1d; font-weight: 600;" for _ in row]
        if trend == "IMPROVED":
            return ["background-color: #dcfce7; color: #14532d;" for _ in row]
        if trend == "NEW ADMISSION":
            return ["background-color: #dbeafe; color: #1e3a8a;" for _ in row]
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
        older_label = _snapshot_label(tracker_previous_snapshot) if tracker_previous_snapshot else ""
        newer_label = _snapshot_label(tracker_current_snapshot) if tracker_current_snapshot else ""
        _render_all_beds_panel(
            tracker_output,
            key_prefix="round_tracker",
            source_rows=tracker_source_rows,
            show_course_button=True,
            comparison_changes=tracker_changes if isinstance(tracker_changes, list) else [],
            comparison_older_label=older_label,
            comparison_newer_label=newer_label,
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
    comparison_changes: list[dict[str, Any]] | None = None,
    comparison_older_label: str = "",
    comparison_newer_label: str = "",
) -> None:
    if not all_beds_output:
        return

    all_beds_output = sorted(all_beds_output, key=_bed_sort_key)
    _render_summary_tiles(all_beds_output)
    st.markdown("### Priority Attention")
    st.caption("Deteriorated first, then critical, then beds with active pending items.")
    _render_priority_attention_table(all_beds_output)
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

    layout_mode = st.radio(
        "Dashboard layout",
        options=["Bed grid", "Detailed list"],
        horizontal=True,
        key=f"dashboard_layout_{key_prefix}",
    )
    filtered_records = sorted(_apply_filters(all_beds_output), key=_bed_sort_key)
    st.caption(
        f"Showing {len(filtered_records)} of {len(all_beds_output)} beds after filters "
        "(bed-wise ascending order)."
    )
    if layout_mode == "Detailed list":
        st.caption("Detailed list view: bed-wise cards with status color coding.")
        for row in filtered_records:
            _render_bed_card(
                row,
                key_prefix=key_prefix,
                show_course_button=show_course_button,
                collapsible=False,
            )
    else:
        st.caption("Bed grid view: bed-wise ascending cards. Click each card to open full details.")
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
    pdf_detail_mode = st.radio(
        "PDF format",
        options=["Consultant (recommended)", "Raw/Audit"],
        index=0,
        horizontal=True,
        key=f"rounds_pdf_detail_{key_prefix}",
    )
    pdf_detail_value = "raw" if pdf_detail_mode.startswith("Raw") else "consultant"

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
            pdf_bytes, pdf_path = generate_rounds_pdf(
                all_beds_output,
                shift=rounds_shift,
                detail_level=pdf_detail_value,
                comparison_changes=comparison_changes or [],
                comparison_older_label=comparison_older_label,
                comparison_newer_label=comparison_newer_label,
            )
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
            "Download PDF (rounds sheet)",
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
    st.session_state.active_tracker_unit = PRIMARY_ICU_UNIT

knowledge_base: KnowledgeBase = st.session_state.knowledge_base
advisor: ClinicalTaskAdvisor = st.session_state.advisor

_ensure_startup_index(knowledge_base)

selected_icu_unit = PRIMARY_ICU_UNIT
if selected_icu_unit != st.session_state.get("active_tracker_unit"):
    _clear_tracker_state()
    st.session_state.active_tracker_unit = selected_icu_unit

_render_app_header(selected_icu_unit)

history_store_cache: dict[str, ICUHistoryStore] = st.session_state.history_store_cache
if selected_icu_unit not in history_store_cache:
    history_store_cache[selected_icu_unit] = ICUHistoryStore(_history_db_path_for_unit(selected_icu_unit))
history_store: ICUHistoryStore = history_store_cache[selected_icu_unit]

model_name = advisor.model
if model_name != advisor.model:
    st.session_state.advisor = ClinicalTaskAdvisor(model=model_name)
    advisor = st.session_state.advisor

case_tab, resources_tab = st.tabs(["Case Review", "Indexed Resources"])

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
    with st.expander("Local round history", expanded=False):
        st.caption(f"Current ICU: {selected_icu_unit} | DB: `{_history_db_path_for_unit(selected_icu_unit).name}`")
        confirm_reset = st.checkbox(
            "Confirm clear local round history",
            value=False,
            key=f"confirm_clear_{_unit_slug(selected_icu_unit)}",
        )
        if st.button("Clear local round history", use_container_width=True, key=f"clear_unit_{_unit_slug(selected_icu_unit)}"):
            if not confirm_reset:
                st.warning("Tick confirmation first to clear history.")
            else:
                try:
                    history_store.clear_all()
                except Exception as error:
                    st.error(f"Could not clear history: {error}")
                else:
                    _clear_tracker_state()
                    _clear_rmo_pdf_state()
                    st.session_state["tracker_context"] = f"Cleared saved history for {selected_icu_unit}."
                    st.success("Local round history cleared.")
    if knowledge_base.chunk_count() == 0:
        st.info("Index not built yet on this deployment. Click `Build/Rebuild index now` above.")
    indexed_files = knowledge_base.list_files()
    if not indexed_files:
        st.info("No indexed files yet. Build the index to enable retrieval.")
    else:
        for file_path in indexed_files:
            st.write(f"- `{file_path}`")

with case_tab:
    st.subheader("ICU Tracker (rounds dashboard)")
    upload_col, load_col, clear_col = st.columns([2.4, 1, 1])
    with upload_col:
        rmo_pdf_uploads = st.file_uploader(
            "Upload 1 or 2 rounds files (PDF or DOCX; comparison appears when 2 files are uploaded)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="combined_rmo_pdf_upload",
        )
    with load_col:
        manual_load = st.button(
            "Load last saved round",
            use_container_width=True,
            key="load_last_saved_round",
        )
    with clear_col:
        if st.button("Clear current view", use_container_width=True, key="clear_rmo_pdf_view"):
            _clear_rmo_pdf_state()
            try:
                st.rerun()
            except Exception:
                pass

    if manual_load:
        latest_snapshot = history_store.get_latest_snapshot()
        if latest_snapshot and _hydrate_rmo_from_snapshot(
            history_store,
            latest_snapshot,
            context_prefix="Loaded last saved round",
        ):
            st.success(f"Loaded {_snapshot_label(latest_snapshot)} from local history.")
        else:
            st.warning("No saved rounds found yet for this ICU.")

    if not rmo_pdf_uploads:
        latest_snapshot = history_store.get_latest_snapshot()
        latest_snapshot_id = int(latest_snapshot.get("snapshot_id", 0)) if latest_snapshot else 0
        loaded_snapshot_id = int(st.session_state.get("rmo_pdf_loaded_snapshot_id", 0) or 0)
        has_current_output = bool(st.session_state.get("rmo_pdf_output", []))
        if latest_snapshot and (not has_current_output or loaded_snapshot_id != latest_snapshot_id):
            _hydrate_rmo_from_snapshot(
                history_store,
                latest_snapshot,
                context_prefix="Auto-restored last saved round",
            )

    restore_context = str(st.session_state.get("rmo_pdf_restore_context", "")).strip()
    if restore_context:
        st.caption(restore_context)

    if rmo_pdf_uploads:
        selected_pdf_files = list(rmo_pdf_uploads[:2])
        if len(rmo_pdf_uploads) > 2:
            st.warning("Only the first 2 files are used for comparison in this version.")

        rmo_signature = "|".join(_uploaded_file_fingerprint(upload) for upload in selected_pdf_files)
        if st.session_state.get("rmo_pdf_signature") != rmo_signature:
            _clear_rmo_pdf_state()
            st.session_state["rmo_pdf_signature"] = rmo_signature

            parsed_docs: list[dict[str, Any]] = []
            for upload in selected_pdf_files:
                payload = upload.getvalue()
                round_date, round_shift = _infer_round_date_shift_from_filename(upload.name)
                try:
                    parsed_rmo = _parse_rounds_document(upload.name, payload)
                except Exception as error:
                    st.error(f"Rounds document parse failed for `{upload.name}`: {error}")
                    continue

                if not isinstance(parsed_rmo, dict):
                    continue
                table_rows_raw = parsed_rmo.get("table_rows", [])
                parsed_rows = table_rows_raw if isinstance(table_rows_raw, list) else []
                warning_raw = parsed_rmo.get("warnings", [])
                debug_raw = parsed_rmo.get("debug_blocks", [])
                parsed_docs.append(
                    {
                        "name": upload.name,
                        "rows": parsed_rows,
                        "blocks": int(parsed_rmo.get("blocks_detected", 0) or 0),
                        "warnings": warning_raw if isinstance(warning_raw, list) else [],
                        "debug_blocks": debug_raw if isinstance(debug_raw, list) else [],
                        "file_hash": hash_payload(payload),
                        "snapshot_date": round_date,
                        "snapshot_shift": round_shift,
                    }
                )

            st.session_state["rmo_pdf_docs"] = parsed_docs

            current_doc: dict[str, Any] | None = None
            previous_rows: list[dict[str, str]] = []
            older_label = ""
            newer_label = ""
            compare_context = ""
            if parsed_docs:
                current_doc = parsed_docs[0]
                newer_label = str(current_doc.get("name", "Current file"))

            if len(parsed_docs) == 2:
                file_names = [str(doc.get("name", "")) for doc in parsed_docs]
                older_idx, newer_idx, reason = infer_round_file_order(file_names)
                older_doc = parsed_docs[older_idx]
                newer_doc = parsed_docs[newer_idx]
                if (
                    str(older_doc.get("snapshot_date", "")).strip()
                    and str(older_doc.get("snapshot_date", "")).strip() == str(newer_doc.get("snapshot_date", "")).strip()
                    and str(older_doc.get("snapshot_shift", "")).strip().lower() == str(newer_doc.get("snapshot_shift", "")).strip().lower()
                ):
                    older_doc["snapshot_shift"] = "Morning"
                    newer_doc["snapshot_shift"] = "Evening"
                current_doc = newer_doc
                previous_rows = list(older_doc.get("rows", []))
                older_label = str(older_doc.get("name", "Older file"))
                newer_label = str(newer_doc.get("name", "Newer file"))
                compare_context = f"Comparison order detected by {reason}."

            _persist_rmo_docs_to_history(history_store, parsed_docs)

            if current_doc:
                current_rows = list(current_doc.get("rows", []))
                st.session_state["rmo_pdf_rows"] = current_rows
                st.session_state["rmo_pdf_previous_rows"] = previous_rows
                st.session_state["rmo_pdf_older_label"] = older_label
                st.session_state["rmo_pdf_newer_label"] = newer_label
                st.session_state["rmo_pdf_compare_context"] = compare_context

                if current_rows:
                    auto_output = _safe_build_all_beds(current_rows)
                    if auto_output:
                        changes: list[dict[str, Any]] = []
                        if previous_rows:
                            changes = compute_snapshot_changes(
                                _state_rows_from_source_rows(current_rows),
                                _state_rows_from_source_rows(previous_rows),
                            )
                        _annotate_output_with_changes(auto_output, changes)
                        st.session_state["rmo_pdf_changes"] = changes
                        st.session_state["rmo_pdf_output"] = auto_output
                        st.session_state["rmo_pdf_autobuilt_signature"] = rmo_signature
                        st.session_state["rmo_pdf_restore_context"] = (
                            "Latest upload saved locally and loaded into dashboard."
                        )
                else:
                    st.session_state.pop("rmo_pdf_output", None)

        parsed_docs = st.session_state.get("rmo_pdf_docs", [])
        for doc in parsed_docs:
            doc_name = str(doc.get("name", "")).strip()
            doc_rows = doc.get("rows", [])
            row_count = len(doc_rows) if isinstance(doc_rows, list) else 0
            blocks = int(doc.get("blocks", 0) or 0)
            saved_id = int(doc.get("saved_snapshot_id", 0) or 0)
            save_error = str(doc.get("save_error", "")).strip()
            save_meta = f" | Saved snapshot ID: {saved_id}" if saved_id > 0 else ""
            st.write(f"`{doc_name}` -> RMO blocks detected = {blocks} | Beds parsed = {row_count}{save_meta}")
            if save_error:
                st.warning(f"Could not save `{doc_name}` to local history: {save_error}")
            warnings = doc.get("warnings", [])
            if isinstance(warnings, list) and warnings:
                with st.expander(f"RMO parser warnings: {doc_name}", expanded=False):
                    for warning in warnings:
                        st.markdown(f"- {warning}")

        compare_context = str(st.session_state.get("rmo_pdf_compare_context", "")).strip()
        if compare_context:
            older_label = str(st.session_state.get("rmo_pdf_older_label", "")).strip()
            newer_label = str(st.session_state.get("rmo_pdf_newer_label", "")).strip()
            st.caption(
                f"Comparison mode: older `{older_label}` -> newer `{newer_label}`. {compare_context}"
            )

        if parsed_docs:
            with st.expander("RMO parser debug (first 2 records per file)", expanded=False):
                for doc in parsed_docs:
                    st.markdown(f"**{doc.get('name', 'File')}**")
                    st.json((doc.get("debug_blocks", []) or [])[:2])
                    st.json((doc.get("rows", []) or [])[:2])

    rmo_rows = st.session_state.get("rmo_pdf_rows", [])
    if rmo_rows and st.button(
        "Regenerate output for ALL beds (from current snapshot)",
        type="primary",
        use_container_width=True,
        key="generate_all_beds_rmo_pdf",
    ):
        rmo_output = _safe_build_all_beds(rmo_rows)
        if rmo_output:
            previous_rows = st.session_state.get("rmo_pdf_previous_rows", [])
            changes: list[dict[str, Any]] = []
            if isinstance(previous_rows, list) and previous_rows:
                changes = compute_snapshot_changes(
                    _state_rows_from_source_rows(rmo_rows),
                    _state_rows_from_source_rows(previous_rows),
                )
            st.session_state["rmo_pdf_changes"] = changes
            _annotate_output_with_changes(rmo_output, changes)
            st.session_state["rmo_pdf_output"] = rmo_output

    rmo_output_rows = st.session_state.get("rmo_pdf_output", [])
    rmo_changes = st.session_state.get("rmo_pdf_changes", [])
    older_label = str(st.session_state.get("rmo_pdf_older_label", "")).strip()
    newer_label = str(st.session_state.get("rmo_pdf_newer_label", "")).strip()
    if rmo_output_rows:
        if str(st.session_state.get("rmo_pdf_autobuilt_signature", "")).startswith("snapshot:"):
            st.success("Loaded bed-wise output from your last saved round.")
        elif rmo_pdf_uploads:
            st.success("Auto-generated bed-wise output from uploaded rounds file.")
        _render_all_beds_panel(
            rmo_output_rows,
            key_prefix="rmo_pdf",
            source_rows=rmo_rows,
            comparison_changes=rmo_changes if isinstance(rmo_changes, list) else [],
            comparison_older_label=older_label,
            comparison_newer_label=newer_label,
        )
        if isinstance(rmo_changes, list) and rmo_changes and older_label and newer_label:
            _render_round_comparison_table(
                rmo_changes,
                older_label=older_label,
                newer_label=newer_label,
            )
    elif rmo_rows:
        st.warning("Beds were parsed but output table is empty. Click `Regenerate output for ALL beds`.")
    else:
        st.info("Upload a rounds PDF to start, or click `Load last saved round`.")

    st.markdown("---")
    st.subheader("Tele-Round Editing Assistant (DOCX/PDF)")
    st.caption("Structured handover editor: Section 2 clinical status + Section 11 concern/recommendation + new consultant orders.")

    tele_upload = st.file_uploader(
        "Upload ICU RMO report for tele-round editing",
        type=["docx", "pdf"],
        accept_multiple_files=False,
        key="tele_round_doc_upload",
    )

    if tele_upload is not None:
        tele_signature = _uploaded_file_fingerprint(tele_upload)
        if st.session_state.get("tele_round_signature") != tele_signature:
            st.session_state["tele_round_signature"] = tele_signature
            st.session_state.pop("tele_round_report", None)
            st.session_state.pop("tele_round_pdf_bytes", None)
            st.session_state.pop("tele_round_pdf_name", None)
            try:
                tele_report = process_icu_report(tele_upload.name, tele_upload.getvalue())
            except Exception as error:
                st.error(f"Tele-round parse failed for `{tele_upload.name}`: {error}")
            else:
                st.session_state["tele_round_report"] = tele_report

    tele_report = st.session_state.get("tele_round_report")
    if isinstance(tele_report, dict) and tele_report.get("patients"):
        report_date = str(tele_report.get("report_date", "")).strip() or date.today().isoformat()
        default_shift = str(tele_report.get("shift_label", "Morning (7:30 AM)")).strip() or "Morning (7:30 AM)"
        shift_options = ["Morning (7:30 AM)", "Evening (9:30 PM)"]
        shift_index = shift_options.index(default_shift) if default_shift in shift_options else 0
        selected_shift = st.radio(
            "Round slot for WhatsApp PDF",
            options=shift_options,
            index=shift_index,
            horizontal=True,
            key="tele_round_shift_label",
        )

        st.write(
            f"Report: `{tele_report.get('filename', '')}` | Date: **{report_date}** | "
            f"Beds parsed: **{len(tele_report.get('patients', []))}**"
        )

        tele_warnings = tele_report.get("warnings", [])
        if isinstance(tele_warnings, list) and tele_warnings:
            with st.expander("Tele-round parser warnings", expanded=False):
                for warning in tele_warnings:
                    st.markdown(f"- {warning}")

        red_flags = tele_report.get("red_flags", [])
        if isinstance(red_flags, list) and red_flags:
            st.error(f"Red Flag Alert: {len(red_flags)} patient(s) marked critical in Section 10.")
            for item in red_flags[:12]:
                st.markdown(
                    f"- Bed **{item.get('bed', '-') or '-'}** | `{item.get('patient_id', '-') or '-'}`: "
                    f"{item.get('issue', 'Red flag marked Y')}"
                )
        else:
            st.success("No Section 10 red-flag 'Y' entries detected.")

        st.markdown("### Bed-wise Tele-Round Input")
        patients = list(tele_report.get("patients", []))
        for patient in patients:
            bed = str(patient.get("bed", "")).strip() or "-"
            patient_id = str(patient.get("patient_id", "")).strip() or "Not documented"
            patient_key = str(patient.get("patient_key", "")).strip()
            if not patient_key:
                patient_key = _state_key_fragment(f"{bed}_{patient_id}")
                patient["patient_key"] = patient_key

            with st.expander(f"Bed {bed} | {patient_id}", expanded=False):
                st.caption(f"Diagnosis: {patient.get('diagnosis', 'Not documented')}")
                st.caption(f"Section 2 clinical status: {patient.get('clinical_status', 'Not documented')}")
                st.caption(f"Supports: {patient.get('supports', 'None documented')} | Acuity: {patient.get('status', 'Not documented')}")

                note_default = (
                    f"Major concern: {patient.get('major_concern', 'Not documented')}\n"
                    f"RMO recommendation: {patient.get('rmo_recommendation', 'Not documented')}"
                )
                st.text_area(
                    "Tele-round editable note",
                    value=note_default,
                    height=100,
                    key=f"tele_round_note_{patient_key}",
                )
                st.text_area(
                    "Section 12: New consultant orders",
                    value="",
                    height=80,
                    key=f"tele_round_orders_{patient_key}",
                )

        if st.button(
            "Export to WhatsApp PDF",
            type="primary",
            use_container_width=True,
            key="export_tele_round_pdf",
        ):
            orders_by_key: dict[str, str] = {}
            for patient in patients:
                patient_key = str(patient.get("patient_key", "")).strip()
                if not patient_key:
                    continue
                orders_by_key[patient_key] = str(
                    st.session_state.get(f"tele_round_orders_{patient_key}", "")
                ).strip()
            export_report = dict(tele_report)
            export_report["shift_label"] = selected_shift
            export_report["report_date"] = report_date
            try:
                logo_candidates = [
                    APP_DIR / "assets" / "hospital_logo.png",
                    APP_DIR / "assets" / "logo.png",
                    APP_DIR / "logo.png",
                ]
                logo_path = next((candidate for candidate in logo_candidates if candidate.exists()), None)
                pdf_bytes, pdf_path = generate_whatsapp_round_pdf(
                    export_report,
                    orders_by_key=orders_by_key,
                    logo_path=logo_path,
                )
            except Exception as error:
                st.error(f"WhatsApp PDF export failed: {error}")
            else:
                st.session_state["tele_round_pdf_bytes"] = pdf_bytes
                st.session_state["tele_round_pdf_name"] = pdf_path.name
                st.success(f"WhatsApp PDF generated: `{pdf_path}`")

        if st.session_state.get("tele_round_pdf_bytes"):
            st.download_button(
                "Download WhatsApp PDF",
                data=st.session_state["tele_round_pdf_bytes"],
                file_name=str(st.session_state.get("tele_round_pdf_name", "ICU_WhatsApp_Rounds.pdf")),
                mime="application/pdf",
                use_container_width=True,
                key="download_tele_round_pdf",
            )
