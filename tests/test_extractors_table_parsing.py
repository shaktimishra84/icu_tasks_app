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


if __name__ == "__main__":
    unittest.main()
