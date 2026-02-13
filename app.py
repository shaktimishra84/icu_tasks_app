from __future__ import annotations

import os
import re
from datetime import datetime
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
    compare_rounds_outputs,
    infer_round_file_order,
)
from src.extractors import ExtractionError, extract_text
from src.knowledge_base import KnowledgeBase
from src.rounds_pdf import generate_rounds_pdf


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
RESOURCE_STORE = DATA_DIR / "resources_index.json"
RESOURCES_ROOT = APP_DIR / "resources"
ENV_FILE = APP_DIR / ".env"


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
    color = {
        "CRITICAL": "#b91c1c",
        "SICK": "#b45309",
        "SERIOUS": "#854d0e",
        "DECEASED": "#374151",
    }.get(status_group, "#334155")
    return f"<span style='background:{color};color:white;padding:2px 8px;border-radius:999px;font-size:12px;'>{escape(status_group)}</span>"


def _status_color(status_group: str) -> str:
    return {
        "CRITICAL": "#b91c1c",
        "SICK": "#b45309",
        "SERIOUS": "#854d0e",
        "DECEASED": "#374151",
        "OTHER": "#334155",
    }.get(status_group, "#334155")


def _badge(label: str, color: str = "#0f172a") -> str:
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;border-radius:999px;"
        f"font-size:11px;margin-left:6px;'>{escape(label)}</span>"
    )


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


def _render_summary_tiles(records: list[dict[str, Any]]) -> None:
    total_beds = len(records)
    critical = sum(1 for row in records if row.get("_status_group") == "CRITICAL")
    mv_count = sum(1 for row in records if row.get("_is_mv"))
    niv_count = sum(1 for row in records if row.get("_is_niv"))
    vaso_count = sum(1 for row in records if row.get("_is_vaso"))
    pending_count = sum(1 for row in records if str(row.get("_pending_verbatim", "")).strip())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total beds", total_beds)
    c2.metric("CRITICAL", critical)
    c3.metric("On MV/vent", mv_count)
    c4.metric("On NIV/BiPAP", niv_count)
    c5.metric("Vasopressor", vaso_count)
    c6.metric("Pending reports", pending_count)


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


def _render_bed_card(record: dict[str, Any]) -> None:
    status_group = str(record.get("_status_group", "OTHER"))
    status_color = _status_color(status_group)
    flags = record.get("_flags", [])
    severity = str(record.get("_flag_severity", "NONE"))
    trend = str(record.get("Round trend", "")).strip()
    deterioration_reason = str(record.get("Deterioration reasons", "")).strip()
    is_deteriorated = str(record.get("Deterioration since last round", "")).upper() == "YES"
    support_badges = _support_badges(record)
    system_tags = [str(x) for x in (record.get("_system_tags", []) or [])]
    matched_algorithms = [str(x) for x in (record.get("_matched_algorithms", []) or [])]

    with st.container(border=True):
        st.markdown(
            f"<div style='height:4px;background:{status_color};border-radius:4px;margin-bottom:8px;'></div>",
            unsafe_allow_html=True,
        )
        if is_deteriorated:
            st.markdown(
                f"<div style='font-weight:700;color:#b91c1c;margin-bottom:4px;'>"
                f"Deteriorated since last round"
                f"{': ' + escape(deterioration_reason) if deterioration_reason else ''}</div>",
                unsafe_allow_html=True,
            )
        elif trend:
            st.caption(f"Trend vs previous round: {trend}")

        if flags:
            color = "#b91c1c" if severity == "RED" else "#b45309"
            st.markdown(
                f"<div style='font-weight:600;color:{color};margin-bottom:4px;'>"
                f"Flags ({escape(severity)}): {escape(', '.join(flags))}</div>",
                unsafe_allow_html=True,
            )

        badges_html = "".join(_badge(tag, "#1d4ed8") for tag in support_badges)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
            f"<strong>Bed {escape(str(record.get('Bed', '')))} | {escape(str(record.get('Patient ID', '')))}</strong>"
            f"{_status_chip(status_group)}{badges_html}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Diagnosis:** {record.get('Diagnosis', '-') or '-'}")
        if system_tags:
            friendly_system = " | ".join(tag.split("_", 1)[-1].replace("_", " ").upper() for tag in system_tags)
            matched_text = ", ".join(matched_algorithms) if matched_algorithms else "None"
            st.markdown(
                f"<div style='color:#475569;font-size:12px;'>System: <strong>{escape(friendly_system)}</strong> "
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

        st.markdown(f"**Key labs/imaging:** {_decision_labs_line(str(record.get('Key labs/imaging (1 line)', '')))}")


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
        st.markdown("<div style='color:#64748b;'>-</div>", unsafe_allow_html=True)
        return
    for item in items:
        st.markdown(
            f"<div style='font-size:14px;line-height:1.35;margin:1px 0;'>• {escape(item)}</div>",
            unsafe_allow_html=True,
        )

def _default_round_shift() -> str:
    return "Morning" if datetime.now().hour < 16 else "Evening"


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


def _render_all_beds_panel(all_beds_output: list[dict[str, Any]], key_prefix: str = "default") -> None:
    if not all_beds_output:
        return

    all_beds_output = sorted(all_beds_output, key=_bed_sort_key)
    _render_summary_tiles(all_beds_output)

    has_round_trend = any(str(row.get("Round trend", "")).strip() for row in all_beds_output)
    if has_round_trend:
        deteriorated_count = sum(
            1 for row in all_beds_output if str(row.get("Deterioration since last round", "")).upper() == "YES"
        )
        st.markdown("### Change Since Last Round")
        st.caption(f"Deteriorated patients: {deteriorated_count}")
        _render_deterioration_table(all_beds_output)

    show_covered_key = f"show_covered_toggle_{key_prefix}"
    st.session_state["show_already_covered"] = st.toggle(
        "Show already covered items",
        value=st.session_state.get("show_already_covered", False),
        key=show_covered_key,
    )

    filtered_records = sorted(_apply_filters(all_beds_output), key=_bed_sort_key)
    st.caption(f"Showing {len(filtered_records)} of {len(all_beds_output)} beds after filters")
    st.caption("Bed-wise order: ascending bed number. Status is color-coded on each card.")
    for row in filtered_records:
        _render_bed_card(row)

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
    if st.button("Generate Rounds PDF", use_container_width=True, key=f"generate_rounds_pdf_{key_prefix}"):
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

knowledge_base: KnowledgeBase = st.session_state.knowledge_base
advisor: ClinicalTaskAdvisor = st.session_state.advisor

st.title("ICU Task Assistant")
st.caption("Round-ready bed cards with deterministic batch rules and filtered workflow view.")
st.warning(
    "Clinical decision support only. A licensed clinician must verify every recommendation before use.",
    icon="⚠️",
)
_ensure_startup_index(knowledge_base)

st.sidebar.header("Settings")
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
    st.subheader("Batch Rounds Comparison (DOCX tables)")
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

        round_signature = "|".join(f"{item.name}:{getattr(item, 'size', 0)}" for item in selected_files)
        if st.session_state.get("round_compare_signature") != round_signature:
            st.session_state.pop("all_beds_output", None)
            st.session_state.pop("all_beds_compare_context", None)
            st.session_state["round_compare_signature"] = round_signature

        parsed_docs: list[dict[str, Any]] = []
        extraction_failed = False
        for upload in selected_files:
            try:
                extracted = extract_text(upload.name, upload.getvalue())
            except Exception as error:
                _show_extraction_error(upload.name, error)
                extraction_failed = True
                continue

            if not isinstance(extracted, dict):
                st.warning(f"`{upload.name}` is not a structured table DOCX.")
                parsed_docs.append({"name": upload.name, "rows": [], "debug_raw_rows": []})
                continue

            rows = extracted.get("table_rows", [])
            table_rows = rows if isinstance(rows, list) else []
            raw_rows = extracted.get("debug_raw_rows", [])
            debug_rows = raw_rows if isinstance(raw_rows, list) else []
            parsed_docs.append(
                {
                    "name": upload.name,
                    "rows": table_rows,
                    "debug_raw_rows": debug_rows,
                }
            )

        for doc in parsed_docs:
            st.write(f"`{doc['name']}` -> Beds parsed = {len(doc['rows'])}")

        if not extraction_failed and parsed_docs:
            if len(parsed_docs) == 2:
                file_names = [str(doc["name"]) for doc in parsed_docs]
                older_idx, newer_idx, reason = infer_round_file_order(file_names)
                older_doc = parsed_docs[older_idx]
                newer_doc = parsed_docs[newer_idx]

                st.info(
                    f"Comparing older `{older_doc['name']}` -> newer `{newer_doc['name']}` "
                    f"(inferred by {reason})."
                )
                with st.expander("Debug parsed records", expanded=False):
                    st.markdown(f"**Older:** `{older_doc['name']}`")
                    st.json(older_doc["rows"][:2])
                    st.markdown(f"**Newer:** `{newer_doc['name']}`")
                    st.json(newer_doc["rows"][:2])

                if st.button(
                    "Generate output for ALL beds",
                    type="primary",
                    use_container_width=True,
                    key="generate_all_beds_compare",
                ):
                    older_output = _safe_build_all_beds(older_doc["rows"])
                    newer_output = _safe_build_all_beds(newer_doc["rows"])
                    if older_output and newer_output:
                        compared_output = compare_rounds_outputs(older_output, newer_output)
                        st.session_state.all_beds_output = compared_output
                        st.session_state.all_beds_compare_context = (
                            f"Compared newer `{newer_doc['name']}` against older `{older_doc['name']}` "
                            f"(inferred by {reason})."
                        )
            else:
                only_doc = parsed_docs[0]
                with st.expander("Debug parsed records", expanded=False):
                    st.json(only_doc["rows"][:2])
                if st.button(
                    "Generate output for ALL beds",
                    type="primary",
                    use_container_width=True,
                    key="generate_all_beds_single_round",
                ):
                    single_output = _safe_build_all_beds(only_doc["rows"])
                    if single_output:
                        st.session_state.all_beds_output = single_output
                        st.session_state.pop("all_beds_compare_context", None)

            all_beds_output = st.session_state.get("all_beds_output", [])
            compare_context = st.session_state.get("all_beds_compare_context", "")
            if compare_context:
                st.caption(compare_context)
            if all_beds_output:
                _render_all_beds_panel(all_beds_output, key_prefix="round_compare")

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

    if patient_file is not None:
        file_signature = f"{patient_file.name}:{getattr(patient_file, 'size', 0)}"
        if st.session_state.get("all_beds_file_signature") != file_signature:
            st.session_state.pop("all_beds_output", None)
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
        elif isinstance(extracted, str):
            extracted_raw_text = extracted

    if extracted_table_rows and not round_files:
        st.info("Structured DOCX bed table detected. Single-note path is disabled for this file.")
        st.write(f"Beds parsed = {len(extracted_table_rows)}")

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

        all_beds_output = st.session_state.get("all_beds_output", [])
        if all_beds_output:
            _render_all_beds_panel(all_beds_output, key_prefix="single_note")
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
