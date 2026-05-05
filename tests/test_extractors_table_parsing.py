from __future__ import annotations

import unittest

from src.extractors import _parse_bed_table, _select_best_table_rows


HEADER = [
    "Bed",
    "Patient ID",
    "Diagnosis",
    "Status",
    "Supports",
    "New issues",
    "Actions done",
    "Plan next 12h",
    "Pending",
    "Key labs/imaging",
]


class ExtractorBedTableParsingTests(unittest.TestCase):
    def test_duplicate_patient_id_keeps_first_bed(self) -> None:
        rows = [
            HEADER,
            ["17.", "9937160805", "Dx A", "SICK", "", "", "", "", "", ""],
            ["", "", "", "", "", "Issue 1", "", "", "", ""],
            ["20.", "9937160805", "Dx A", "SICK", "", "", "", "", "", ""],
        ]
        parsed, warnings = _parse_bed_table(rows)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["bed"], "17")
        self.assertEqual(parsed[0]["patient_id"], "9937160805")
        self.assertTrue(warnings)

    def test_selects_table_with_more_patient_rows(self) -> None:
        table_a = [
            HEADER,
            ["1", "A-1", "Dx 1", "SICK", "", "", "", "", "", ""],
        ]
        table_b = [
            HEADER,
            ["1", "B-1", "Dx 1", "SICK", "", "", "", "", "", ""],
            ["2", "B-2", "Dx 2", "CRITICAL", "", "", "", "", "", ""],
        ]
        parsed, table_index, _warnings = _select_best_table_rows([table_a, table_b])
        self.assertEqual(table_index, 1)
        self.assertEqual(len(parsed), 2)

    def test_parses_header_with_leading_serial_column(self) -> None:
        header_with_serial = [
            "Sr No",
            "ICU Bed",
            "UHID",
            "Working Diagnosis",
            "Clinical Status",
            "Organ supports",
            "New problems",
            "Interventions done",
            "Next 12 h plan",
            "Pending tests",
            "Labs/Imaging",
        ]
        rows = [
            header_with_serial,
            ["1", "4.", "6372223422", "Pneumonia", "SICK", "O2", "", "", "Repeat ABG", "HRCT pending", "K 2.26"],
            ["2", "17.", "9937160805", "Dx B", "CRITICAL", "MV", "", "", "", "EEG report", "Na 132"],
        ]
        parsed, warnings = _parse_bed_table(rows)
        self.assertFalse(warnings)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["bed"], "4")
        self.assertEqual(parsed[0]["patient_id"], "6372223422")
        self.assertEqual(parsed[1]["bed"], "17")
        self.assertEqual(parsed[1]["patient_id"], "9937160805")

    def test_bed_cell_uses_number_after_bed_label(self) -> None:
        rows = [
            HEADER,
            ["ICU1 BED 10", "202605010054", "Aspiration pneumonia", "SICK", "O2", "", "", "", "", ""],
        ]
        parsed, warnings = _parse_bed_table(rows)
        self.assertFalse(warnings)
        self.assertEqual(parsed[0]["bed"], "10")

    def test_continuation_rows_append_fields(self) -> None:
        rows = [
            HEADER,
            ["18", "9668499624", "Old CVA", "SICK", "", "", "", "", "", ""],
            ["", "", "", "", "", "Pressure sore grade 4", "Debridement day 9", "", "Sodium report", "HypoNa"],
        ]
        parsed, _warnings = _parse_bed_table(rows)
        self.assertEqual(len(parsed), 1)
        record = parsed[0]
        self.assertEqual(record["bed"], "18")
        self.assertEqual(record["patient_id"], "9668499624")
        self.assertIn("Pressure sore", record["new_issues"])
        self.assertIn("Debridement", record["actions_done"])

    def test_fallback_parses_no_header_rows_default_offset(self) -> None:
        rows = [
            ["1", "9178556677", "Seizure disorder", "SICK", "O2", "", "", "", "Psych consult pending", "K 4.2"],
            ["2", "9668499624", "Old CVA", "SICK", "", "", "", "", "EEG report", "Na 128"],
        ]
        parsed, warnings = _parse_bed_table(rows)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["bed"], "1")
        self.assertEqual(parsed[0]["patient_id"], "9178556677")
        self.assertEqual(parsed[1]["bed"], "2")
        self.assertEqual(parsed[1]["patient_id"], "9668499624")
        self.assertTrue(any("fallback" in item.lower() for item in warnings))

    def test_fallback_parses_no_header_rows_serial_offset(self) -> None:
        rows = [
            ["1", "25.", "9178556677", "Seizure disorder", "SICK", "O2", "", "", "", "Psych consult", "K 4.2"],
            ["2", "26.", "9124061892", "CAP/PTB", "SICK", "O2", "", "", "", "CT chest", "Cr 1.1"],
        ]
        parsed, warnings = _parse_bed_table(rows)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["bed"], "25")
        self.assertEqual(parsed[0]["patient_id"], "9178556677")
        self.assertEqual(parsed[1]["bed"], "26")
        self.assertEqual(parsed[1]["patient_id"], "9124061892")
        self.assertTrue(any("fallback" in item.lower() for item in warnings))

    def test_fallback_rejects_vitals_rows_as_patient_rows(self) -> None:
        rows = [
            ["05.05.26 07.30AM", "E4V2M6", "86", "156/70(85)", "96", "14", "98.6", "25/50/25/10/25/10", "102"],
            ["04.05.26 09.30PM", "E4V2M6", "91", "136/78", "97", "18", "98.8", "50/50/40/35/40/20", "116"],
        ]
        parsed, warnings = _parse_bed_table(rows)
        self.assertEqual(parsed, [])
        self.assertFalse(any("fallback" in item.lower() for item in warnings))

    def test_fallback_rejects_doctor_roster_rows_as_patient_rows(self) -> None:
        rows = [
            ["7.30AM", "DR PRITI DR TUNU", "", "", "DR SANTANU DR SONALI"],
            ["9.30PM", "DR SANTANU DR SIDHARTH", "", "", "DR PRITI DR TUNU"],
        ]
        parsed, warnings = _parse_bed_table(rows)
        self.assertEqual(parsed, [])
        self.assertFalse(any("fallback" in item.lower() for item in warnings))


if __name__ == "__main__":
    unittest.main()
