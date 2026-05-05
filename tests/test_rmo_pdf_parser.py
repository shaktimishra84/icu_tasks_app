from __future__ import annotations

import unittest

from src.rmo_pdf import parse_combined_rmo_text


SAMPLE_TEXT = """
ICU RMO REPORTING FORMAT
SECTION 1: PATIENT IDENTIFICATION & BASIC DETAILS
UHID Patient Name Age/Sex Bed DOA Primary Diagnosis
202601170031 RAM KUMAR 75/M 14 08/04/26 DR SB MISHRA CKD ON MHD, POST CPR STATUS ICU
SECTION 2: CLINICAL STATUS SUMMARY (CURRENT)
29.04.26 7.30 AM E1VTM1 107 161/96(106) 99 22 98F 15/10/5/NIL/5/10 138
SECTION 3: RESPIRATORY & VENTILATION STATUS
29.04.26 7.30 AM MV PRVC 30 6 98 7.37/31.5/134/19.2/0.4 NO
SECTION 4: HEMODYNAMIC STATUS
29.04.26 7.30 AM Y NORAD 0.1 2.2 FAIR NO
SECTION 6: RENAL & METABOLIC STATUS
29.04.26 7.30 AM CREAT 2.8 K 5.7 HD DONE
SECTION 7: INFECTION & SEPSIS MONITORING
29.04.26 7.30 AM FEVER Y WBC HIGH CRP HIGH
SECTION 9: INVESTIGATIONS & INTERVENTIONS
29.04.26 7.30 AM  Y  EEG REPORT PENDING  TRACHEOSTOMY DONE  HD TODAY
SECTION 10: RED FLAG IDENTIFICATION (CRITICAL)
29.04.26 7.30 AM Y LOW GCS Y
SECTION 11: SUMMARY FOR CONSULTANT (MOST IMPORTANT)
29.04.26 7.30 AM CRITICAL MAJOR CONCERN SHOCK WORSENED REQUIREMENT VENT SUPPORT
SECTION 12: ESCALATION & ORDERS
29.04.26 7.30 AM DR SATYAJIT REPEAT ABG AND CONTINUE NORAD YES
SECTION 13: HANDOVER ACCOUNTABILITY
29.04.26 7.30 AM RMO A

ICU RMO REPORTING FORMAT
SECTION 1: PATIENT IDENTIFICATION & BASIC DETAILS
UHID Patient Name Age/Sex Bed DOA Primary Diagnosis
202508070086 PARVATI MAHARANA 57/F 13 28/04/26 DR X ACUTE EXACERBATION OF COPD ICU
SECTION 2: CLINICAL STATUS SUMMARY (CURRENT)
29.04.26 7.30 AM UNDER SEDATION 107 97/61(59) 99 21 98.4 20/25/20/50/40/25 168
SECTION 3: RESPIRATORY & VENTILATION STATUS
29.04.26 7.30 AM NIV BIPAP 40 6 98 7.38/44.3/173/25.8/1.0 N
SECTION 9: INVESTIGATIONS & INTERVENTIONS
29.04.26 7.30 AM Y NIL NIV CARE PLAN
SECTION 10: RED FLAG IDENTIFICATION (CRITICAL)
29.04.26 7.30 AM N NIL N
SECTION 11: SUMMARY FOR CONSULTANT (MOST IMPORTANT)
29.04.26 7.30 AM GUARDED MAJOR CONCERN RESP DISTRESS REQUIREMENT NIV TRIAL
SECTION 12: ESCALATION & ORDERS
29.04.26 7.30 AM DR SAMAL CONTINUE NIV YES
SECTION 13: HANDOVER ACCOUNTABILITY
29.04.26 7.30 AM RMO B
"""


class RmoPdfParserTests(unittest.TestCase):
    def test_parse_combined_rmo_text_extracts_rows(self) -> None:
        payload = parse_combined_rmo_text(SAMPLE_TEXT)
        rows = payload.get("table_rows", [])
        self.assertEqual(payload.get("blocks_detected"), 2)
        self.assertEqual(len(rows), 2)

        first = rows[0]
        self.assertEqual(first.get("bed"), "14")
        self.assertEqual(first.get("patient_id"), "202601170031")
        self.assertIn("CKD", first.get("diagnosis", ""))
        self.assertIn("MV", first.get("supports", ""))
        self.assertIn("Vasopressor", first.get("supports", ""))
        self.assertIn("Dialysis", first.get("supports", ""))
        self.assertEqual(first.get("status"), "CRITICAL")
        self.assertIn("LOW GCS", first.get("new_issues", "").upper())
        self.assertEqual(first.get("red_flag_present"), "Y")
        self.assertIn("SHOCK", first.get("major_concern", "").upper())
        self.assertIn("VENT SUPPORT", first.get("rmo_recommendation", "").upper())
        self.assertIn("E1VTM1", first.get("section2_status", ""))

    def test_parse_returns_warning_when_no_blocks(self) -> None:
        payload = parse_combined_rmo_text("Plain PDF text without ICU structure")
        self.assertEqual(payload.get("blocks_detected"), 0)
        self.assertEqual(payload.get("table_rows"), [])
        warnings = payload.get("warnings", [])
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
