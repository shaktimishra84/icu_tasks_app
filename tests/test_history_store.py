from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.history_store import ICUHistoryStore, patient_key_from_source_row


class HistoryStoreTests(unittest.TestCase):
    def test_patient_key_prefers_patient_id_then_bed(self) -> None:
        self.assertEqual(
            patient_key_from_source_row({"patient_id": "P-10", "bed": "5"}),
            "pid:p10",
        )
        self.assertEqual(
            patient_key_from_source_row({"patient_id": "", "bed": "Bed 12"}),
            "bed:12",
        )

    def test_save_and_load_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "icu.db"
            store = ICUHistoryStore(db_path)

            snapshot_one = store.save_snapshot(
                snapshot_date="2026-02-12",
                shift="Morning",
                file_hash="hash-one",
                table_rows=[
                    {
                        "bed": "1",
                        "patient_id": "P-1",
                        "diagnosis": "Sepsis",
                        "status": "SICK",
                        "supports": "O2",
                        "new_issues": "",
                        "actions_done": "",
                        "plan_next_12h": "",
                        "pending": "CTPA pending",
                        "key_labs_imaging": "Lactate 2.5",
                    },
                    {
                        "bed": "2",
                        "patient_id": "",
                        "diagnosis": "COPD",
                        "status": "SERIOUS",
                        "supports": "NIV",
                        "new_issues": "",
                        "actions_done": "",
                        "plan_next_12h": "",
                        "pending": "ABG pending",
                        "key_labs_imaging": "",
                    },
                ],
            )
            snapshot_two = store.save_snapshot(
                snapshot_date="2026-02-13",
                shift="Morning",
                file_hash="hash-two",
                table_rows=[
                    {
                        "bed": "1",
                        "patient_id": "P-1",
                        "diagnosis": "Septic shock",
                        "status": "CRITICAL",
                        "supports": "MV; Norad",
                        "new_issues": "Shock",
                        "actions_done": "",
                        "plan_next_12h": "",
                        "pending": "CTPA pending",
                        "key_labs_imaging": "Lactate 4.0",
                    },
                    {
                        "bed": "2",
                        "patient_id": "",
                        "diagnosis": "COPD",
                        "status": "SICK",
                        "supports": "NIV",
                        "new_issues": "",
                        "actions_done": "",
                        "plan_next_12h": "",
                        "pending": "ABG pending",
                        "key_labs_imaging": "",
                    },
                ],
            )

            previous = store.get_previous_snapshot(snapshot_two)
            self.assertIsNotNone(previous)
            self.assertEqual(int(previous["snapshot_id"]), snapshot_one)

            rows = store.get_snapshot_rows(snapshot_two)
            self.assertEqual(len(rows), 2)
            keys = {row["patient_key"] for row in rows}
            self.assertIn("pid:p1", keys)
            self.assertIn("bed:2", keys)

    def test_save_same_slot_overwrites_snapshot_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "icu.db"
            store = ICUHistoryStore(db_path)

            snapshot_one = store.save_snapshot(
                snapshot_date="2026-02-14",
                shift="Evening",
                file_hash="hash-a",
                table_rows=[
                    {
                        "bed": "10",
                        "patient_id": "A-10",
                        "diagnosis": "Dx A",
                        "status": "SICK",
                        "supports": "",
                        "new_issues": "",
                        "actions_done": "",
                        "plan_next_12h": "",
                        "pending": "",
                        "key_labs_imaging": "",
                    }
                ],
            )

            snapshot_two = store.save_snapshot(
                snapshot_date="2026-02-14",
                shift="Evening",
                file_hash="hash-b",
                table_rows=[
                    {
                        "bed": "11",
                        "patient_id": "B-11",
                        "diagnosis": "Dx B",
                        "status": "CRITICAL",
                        "supports": "MV",
                        "new_issues": "",
                        "actions_done": "",
                        "plan_next_12h": "",
                        "pending": "",
                        "key_labs_imaging": "",
                    }
                ],
            )

            self.assertEqual(snapshot_one, snapshot_two)
            rows = store.get_snapshot_rows(snapshot_one)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["patient_id"], "B-11")

    def test_clear_all_removes_saved_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "icu.db"
            store = ICUHistoryStore(db_path)
            store.save_snapshot(
                snapshot_date="2026-02-15",
                shift="Morning",
                file_hash="hash-x",
                table_rows=[
                    {
                        "bed": "4",
                        "patient_id": "P-4",
                        "diagnosis": "Dx",
                        "status": "SICK",
                        "supports": "",
                        "new_issues": "",
                        "actions_done": "",
                        "plan_next_12h": "",
                        "pending": "",
                        "key_labs_imaging": "",
                    }
                ],
            )
            self.assertIsNotNone(store.get_latest_snapshot())
            store.clear_all()
            self.assertIsNone(store.get_latest_snapshot())

    def test_missing_patient_outcome_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "icu.db"
            store = ICUHistoryStore(db_path)

            store.save_snapshot(
                snapshot_date="2026-02-14",
                shift="Morning",
                file_hash="hash-s1",
                table_rows=[
                    {
                        "bed": "1",
                        "patient_id": "P-1",
                        "diagnosis": "Dx 1",
                        "status": "SICK",
                        "supports": "",
                        "new_issues": "",
                        "actions_done": "",
                        "plan_next_12h": "",
                        "pending": "",
                        "key_labs_imaging": "",
                    },
                    {
                        "bed": "2",
                        "patient_id": "P-2",
                        "diagnosis": "Dx 2",
                        "status": "CRITICAL",
                        "supports": "MV",
                        "new_issues": "",
                        "actions_done": "",
                        "plan_next_12h": "",
                        "pending": "",
                        "key_labs_imaging": "",
                    },
                ],
            )
            snapshot_two = store.save_snapshot(
                snapshot_date="2026-02-15",
                shift="Morning",
                file_hash="hash-s2",
                table_rows=[
                    {
                        "bed": "1",
                        "patient_id": "P-1",
                        "diagnosis": "Dx 1",
                        "status": "SERIOUS",
                        "supports": "",
                        "new_issues": "",
                        "actions_done": "",
                        "plan_next_12h": "",
                        "pending": "",
                        "key_labs_imaging": "",
                    }
                ],
            )

            store.save_missing_outcome(
                snapshot_id=snapshot_two,
                patient_key="pid:p2",
                patient_id="P-2",
                bed="2",
                last_status="CRITICAL",
                outcome="Shifted",
                notes="Shifted to ICU 3",
            )
            outcomes = store.get_missing_outcomes(snapshot_two)
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0]["patient_key"], "pid:p2")
            self.assertEqual(outcomes[0]["outcome"], "Shifted")

            store.delete_missing_outcome(snapshot_id=snapshot_two, patient_key="pid:p2")
            self.assertEqual(store.get_missing_outcomes(snapshot_two), [])


if __name__ == "__main__":
    unittest.main()
