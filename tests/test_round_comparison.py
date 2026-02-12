from __future__ import annotations

import unittest

from src.batch_mode import compare_rounds_outputs, infer_round_file_order


class RoundComparisonTests(unittest.TestCase):
    def test_infer_order_by_date_in_filename(self) -> None:
        older_idx, newer_idx, reason = infer_round_file_order(
            [
                "icu_round_2026-02-11_evening.docx",
                "icu_round_2026-02-12_morning.docx",
            ]
        )
        self.assertEqual((older_idx, newer_idx), (0, 1))
        self.assertIn("date/time", reason)

    def test_infer_order_by_keywords(self) -> None:
        older_idx, newer_idx, reason = infer_round_file_order(
            ["round_prev.docx", "round_latest.docx"]
        )
        self.assertEqual((older_idx, newer_idx), (0, 1))
        self.assertIn("keywords", reason)

    def test_compare_marks_deterioration(self) -> None:
        previous = [
            {
                "Bed": "5",
                "Patient ID": "P-5",
                "_status_group": "SICK",
                "_is_mv": False,
                "_is_niv": False,
                "_is_vaso": False,
                "_is_rrt": False,
                "_is_o2": True,
            }
        ]
        current = [
            {
                "Bed": "5",
                "Patient ID": "P-5",
                "_status_group": "CRITICAL",
                "_is_mv": True,
                "_is_niv": False,
                "_is_vaso": True,
                "_is_rrt": False,
                "_is_o2": True,
            }
        ]

        compared = compare_rounds_outputs(previous, current)
        row = compared[0]
        self.assertEqual(row["Round trend"], "DETERIORATED")
        self.assertEqual(row["Deterioration since last round"], "YES")
        self.assertIn("Status SICK -> CRITICAL", row["Deterioration reasons"])

    def test_compare_marks_stable(self) -> None:
        previous = [
            {
                "Bed": "7",
                "Patient ID": "P-7",
                "_status_group": "SICK",
                "_is_mv": False,
                "_is_niv": True,
                "_is_vaso": False,
                "_is_rrt": False,
                "_is_o2": True,
            }
        ]
        current = [
            {
                "Bed": "7",
                "Patient ID": "P-7",
                "_status_group": "SICK",
                "_is_mv": False,
                "_is_niv": True,
                "_is_vaso": False,
                "_is_rrt": False,
                "_is_o2": True,
            }
        ]

        compared = compare_rounds_outputs(previous, current)
        row = compared[0]
        self.assertEqual(row["Round trend"], "STABLE")
        self.assertEqual(row["Deterioration since last round"], "NO")


if __name__ == "__main__":
    unittest.main()
