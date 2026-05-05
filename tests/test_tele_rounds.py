from __future__ import annotations

from unittest.mock import patch

from src.tele_rounds import process_icu_report

SAMPLE_TEXT = """
ICU RMO REPORTING FORMAT
SECTION 1: PATIENT IDENTIFICATION & BASIC DETAILS
UHID Patient Name Age/Sex Bed DOA Primary Diagnosis
202601170031 RAM KUMAR 75/M 14 08/04/26 DR SB MISHRA CKD ON MHD, POST CPR STATUS ICU
SECTION 2: CLINICAL STATUS SUMMARY (CURRENT)
29.04.26 7.30 AM E1VTM1 107 161/96(106)
SECTION 9: INVESTIGATIONS & INTERVENTIONS
29.04.26 7.30 AM Y EEG REPORT PENDING TRACHEOSTOMY DONE HD TODAY
SECTION 10: RED FLAG IDENTIFICATION (CRITICAL)
29.04.26 7.30 AM Y LOW GCS Y
SECTION 11: SUMMARY FOR CONSULTANT (MOST IMPORTANT)
29.04.26 7.30 AM CRITICAL MAJOR CONCERN SHOCK WORSENED REQUIREMENT VENT SUPPORT
SECTION 12: ESCALATION & ORDERS
29.04.26 7.30 AM DR SATYAJIT REPEAT ABG AND CONTINUE NORAD YES
SECTION 13: HANDOVER ACCOUNTABILITY
29.04.26 7.30 AM RMO A
"""


def test_process_icu_report_docx_uses_section_blocks() -> None:
    payload = {"raw_text": SAMPLE_TEXT, "table_rows": []}
    with patch("src.tele_rounds.extract_text", return_value=payload):
        report = process_icu_report("CCM PT DATA UPDATE ON 22ND FEB EVENING.docx", b"fake")

    patients = report.get("patients", [])
    assert len(patients) == 1
    patient = patients[0]
    assert patient.get("bed") == "14"
    assert patient.get("red_flag") is True
    assert "SHOCK" in str(patient.get("major_concern", "")).upper()
    assert "VENT SUPPORT" in str(patient.get("rmo_recommendation", "")).upper()


def test_process_icu_report_unsupported_file_type() -> None:
    try:
        process_icu_report("rounds.txt", b"hello")
    except Exception as error:  # noqa: BLE001
        assert "Only PDF and DOCX" in str(error)
    else:
        raise AssertionError("Expected unsupported file type error")
