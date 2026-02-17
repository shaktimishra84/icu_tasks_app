from __future__ import annotations

import unittest

from src.rounds_tracker import compute_snapshot_changes, group_changes


class RoundsTrackerTests(unittest.TestCase):
    def test_compute_changes_detects_status_support_and_pending_delta(self) -> None:
        previous = [
            {
                "patient_key": "pid:p1",
                "patient_id": "P-1",
                "bed": "1",
                "diagnosis": "Sepsis",
                "status": "SICK",
                "supports": "O2",
                "new_issues": "AKI",
                "actions_done": "",
                "plan_next_12h": "",
                "pending": "EEG pending; CTPA pending",
                "key_labs_imaging": "Lactate 2.0",
            }
        ]
        current = [
            {
                "patient_key": "pid:p1",
                "patient_id": "P-1",
                "bed": "1",
                "diagnosis": "Sepsis",
                "status": "CRITICAL",
                "supports": "MV; Norad",
                "new_issues": "AKI; shock",
                "actions_done": "",
                "plan_next_12h": "",
                "pending": "EEG pending; MRI pending",
                "key_labs_imaging": "Lactate 4.0",
            }
        ]

        changes = compute_snapshot_changes(current, previous)
        self.assertEqual(len(changes), 1)
        change = changes[0]
        self.assertEqual(change["trend"], "DETERIORATED")
        self.assertIn("MV", change["supports_added"])
        self.assertIn("VASO", change["supports_added"])
        self.assertIn("MRI pending", change["pending_new"])
        self.assertIn("CTPA pending", change["pending_resolved"])
        self.assertIn("EEG pending", change["pending_unresolved"])
        self.assertIn("shock", [item.lower() for item in change["issues_added"]])
        self.assertLessEqual(len(change["summary_lines"]), 4)

    def test_new_admission_when_no_previous_row(self) -> None:
        current = [
            {
                "patient_key": "pid:p7",
                "patient_id": "P-7",
                "bed": "7",
                "diagnosis": "COPD",
                "status": "SICK",
                "supports": "NIV",
                "new_issues": "",
                "actions_done": "",
                "plan_next_12h": "",
                "pending": "ABG pending",
                "key_labs_imaging": "",
            }
        ]
        changes = compute_snapshot_changes(current, [])
        self.assertEqual(changes[0]["trend"], "NEW ADMISSION")

    def test_group_changes_sections(self) -> None:
        changes = [
            {
                "trend": "DETERIORATED",
                "supports_added": ["MV"],
                "pending_new": ["MRI pending"],
                "pending_resolved": [],
            },
            {
                "trend": "IMPROVED",
                "supports_added": [],
                "pending_new": [],
                "pending_resolved": ["CT pending"],
            },
        ]
        grouped = group_changes(changes)
        self.assertEqual(len(grouped["deteriorated"]), 1)
        self.assertEqual(len(grouped["improved"]), 1)
        self.assertEqual(len(grouped["new_pending"]), 1)
        self.assertEqual(len(grouped["pending_resolved"]), 1)
        self.assertEqual(len(grouped["support_escalations"]), 1)


if __name__ == "__main__":
    unittest.main()
