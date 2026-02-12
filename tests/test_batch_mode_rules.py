from __future__ import annotations

import unittest

from src.batch_mode import build_all_beds_outputs


class MissingInfoEngineTests(unittest.TestCase):
    def test_deceased_suppresses_outputs(self) -> None:
        rows = [
            {
                "bed": "1",
                "patient_id": "D-1",
                "diagnosis": "DECLARED CLINICALLY DEAD",
                "status": "CRITICAL",
                "supports": "MV",
                "new_issues": "",
                "actions_done": "",
                "plan_next_12h": "",
                "pending": "CT report pending",
                "key_labs_imaging": "",
            }
        ]
        out = build_all_beds_outputs(rows)[0]
        self.assertEqual(out["Status"], "DECEASED")
        self.assertEqual(out["Missing Tests"], "")
        self.assertEqual(out["Missing Imaging"], "")
        self.assertEqual(out["Missing Consults"], "")
        self.assertEqual(out["Care checks (deterministic)"], "")
        self.assertIn("Pending admin", out["Pending (verbatim)"])

    def test_copd_t2rf_mv_hits_respiratory_algorithm(self) -> None:
        rows = [
            {
                "bed": "2",
                "patient_id": "R-2",
                "diagnosis": "AE COPD with T2RF",
                "status": "CRITICAL",
                "supports": "MV",
                "new_issues": "",
                "actions_done": "",
                "plan_next_12h": "",
                "pending": "",
                "key_labs_imaging": "ABG pCO2 68",
            }
        ]
        out = build_all_beds_outputs(rows)[0]
        tags = out["_system_tags"]
        self.assertIn("07_respiratory", tags)
        self.assertIn("COPD_T2RF_Vent", out["Matched algorithms"])

    def test_meningitis_gets_neuro_and_infectious_tags(self) -> None:
        rows = [
            {
                "bed": "5",
                "patient_id": "N-5",
                "diagnosis": "Meningitis",
                "status": "SICK",
                "supports": "",
                "new_issues": "AMS",
                "actions_done": "",
                "plan_next_12h": "",
                "pending": "",
                "key_labs_imaging": "",
            }
        ]
        out = build_all_beds_outputs(rows)[0]
        self.assertIn("01_neuro", out["_system_tags"])
        self.assertIn("10_infectious", out["_system_tags"])
        self.assertTrue(out["Matched algorithms"])

    def test_pneumonia_septic_shock_matches_septicshock(self) -> None:
        rows = [
            {
                "bed": "6",
                "patient_id": "I-6",
                "diagnosis": "Pneumonia with septic shock",
                "status": "CRITICAL",
                "supports": "Norad infusion",
                "new_issues": "",
                "actions_done": "",
                "plan_next_12h": "",
                "pending": "",
                "key_labs_imaging": "Lactate 4.2",
            }
        ]
        out = build_all_beds_outputs(rows)[0]
        self.assertIn("10_infectious", out["_system_tags"])
        self.assertIn("SepticShock", out["Matched algorithms"])

    def test_pending_subtracts_from_missing(self) -> None:
        rows = [
            {
                "bed": "8",
                "patient_id": "X-8",
                "diagnosis": "Encephalopathy",
                "status": "SICK",
                "supports": "",
                "new_issues": "AMS",
                "actions_done": "",
                "plan_next_12h": "",
                "pending": "EEG report pending; 2D ECHO pending; Nephrology and Cardiology call",
                "key_labs_imaging": "",
            }
        ]
        out = build_all_beds_outputs(rows)[0]
        self.assertNotIn("EEG", out["Missing Tests"])
        self.assertNotIn("Nephrology consult", out["Missing Consults"])

    def test_cross_system_guardrails(self) -> None:
        rows = [
            {
                "bed": "11",
                "patient_id": "N-11",
                "diagnosis": "CVA infarct",
                "status": "CRITICAL",
                "supports": "O2",
                "new_issues": "",
                "actions_done": "",
                "plan_next_12h": "",
                "pending": "",
                "key_labs_imaging": "Creatinine 1.1",
            },
            {
                "bed": "12",
                "patient_id": "R-12",
                "diagnosis": "AKI on CKD",
                "status": "SICK",
                "supports": "RRT",
                "new_issues": "",
                "actions_done": "",
                "plan_next_12h": "",
                "pending": "",
                "key_labs_imaging": "Creatinine 3.4",
            },
        ]
        out = build_all_beds_outputs(rows)
        neuro = out[0]
        renal = out[1]
        self.assertNotIn("AKI_RRT", neuro["Matched algorithms"])
        self.assertNotIn("ARDS_VentStrategy", renal["Matched algorithms"])


if __name__ == "__main__":
    unittest.main()
