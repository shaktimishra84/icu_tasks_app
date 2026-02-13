from __future__ import annotations

import unittest

from src.rounds_pdf import (
    _bed_sort_key_for_pdf,
    _build_pending_tracker_rows,
    _care_check_items,
    _missing_category_items,
    _pending_items,
    _status_group_for_pdf,
)


class RoundsPdfRulesTests(unittest.TestCase):
    def test_missing_category_max_two_high_first_non_generic(self) -> None:
        row = {
            "_status_group": "CRITICAL",
            "Missing Tests": "- High: ABG trend\n- High: Lactate trend\n- High: Reassess ABCs",
        }
        tests = _missing_category_items(row, "Missing Tests", max_items=2)
        self.assertEqual(len(tests), 2)
        self.assertIn("ABG trend", " | ".join(tests))
        self.assertNotIn("Reassess ABCs", " | ".join(tests))

    def test_care_checks_max_two(self) -> None:
        row = {
            "Care checks (deterministic)": "- High: Ventilator review\n- Medium: VAP bundle check\n- Low: Sedation holiday",
        }
        checks = _care_check_items(row, max_items=2)
        self.assertEqual(len(checks), 2)
        self.assertEqual(checks[0], "Ventilator review")

    def test_pending_max_two_and_dedup(self) -> None:
        row = {
            "Pending (verbatim)": "EEG report pending; EEG report pending | CTPA report pending | UGIE pending",
        }
        pending = _pending_items(row, max_items=2)
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0], "EEG report pending")
        self.assertEqual(pending[1], "CTPA report pending")

    def test_other_maps_to_stable(self) -> None:
        row = {"_status_group": "OTHER"}
        self.assertEqual(_status_group_for_pdf(row), "STABLE")

    def test_pending_tracker_includes_pending_and_high_missing(self) -> None:
        rows = [
            {
                "Bed": "1",
                "Patient ID": "P-1",
                "_status_group": "CRITICAL",
                "Pending (verbatim)": "2D ECHO pending",
            }
        ]
        tracker_rows = _build_pending_tracker_rows(rows)
        self.assertEqual(len(tracker_rows), 1)
        priorities = {entry[3] for entry in tracker_rows}
        self.assertIn("Pending", priorities)

    def test_pending_tracker_sorted_by_bed_numeric_ascending(self) -> None:
        rows = [
            {"Bed": "10", "Patient ID": "P-10", "_status_group": "SICK", "Pending (verbatim)": "EEG pending"},
            {"Bed": "2", "Patient ID": "P-2", "_status_group": "CRITICAL", "Pending (verbatim)": "CTPA pending"},
            {"Bed": "1", "Patient ID": "P-1", "_status_group": "SERIOUS", "Pending (verbatim)": "MRI pending"},
        ]
        tracker_rows = _build_pending_tracker_rows(rows)
        ordered_beds = [entry[0] for entry in tracker_rows]
        self.assertEqual(ordered_beds, ["1", "2", "10"])

    def test_pdf_card_sort_is_bedwise_not_severitywise(self) -> None:
        rows = [
            {"Bed": "10", "Patient ID": "P-10", "_status_group": "CRITICAL"},
            {"Bed": "2", "Patient ID": "P-2", "_status_group": "SERIOUS"},
            {"Bed": "1", "Patient ID": "P-1", "_status_group": "SICK"},
        ]
        ordered = sorted(rows, key=_bed_sort_key_for_pdf)
        self.assertEqual([row["Bed"] for row in ordered], ["1", "2", "10"])


if __name__ == "__main__":
    unittest.main()
