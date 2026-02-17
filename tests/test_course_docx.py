from __future__ import annotations

import unittest
from io import BytesIO

try:
    from docx import Document
except Exception:  # pragma: no cover
    Document = None

import src.course_docx as course_docx


@unittest.skipIf(Document is None, "python-docx not installed")
class CourseDocxTests(unittest.TestCase):
    def test_generate_course_docx_outputs_valid_doc(self) -> None:
        history = [
            {
                "date": "2026-02-14",
                "shift": "Morning",
                "status": "SICK",
                "diagnosis": "Sepsis",
                "supports": "O2",
                "new_issues": "AKI",
                "pending": "EEG pending; CTPA pending",
                "key_labs_imaging": "Lactate 2.5",
                "plan_next_12h": "ABG repeat; monitor urine output",
            },
            {
                "date": "2026-02-15",
                "shift": "Morning",
                "status": "CRITICAL",
                "diagnosis": "Septic shock",
                "supports": "MV; Norad",
                "new_issues": "Shock",
                "pending": "EEG pending; MRI pending",
                "key_labs_imaging": "Lactate 4.0",
                "plan_next_12h": "Vaso titration; ABG repeat",
            },
        ]
        selected_output = {
            "Missing Tests": "- High: ABG trend",
            "Missing Imaging": "- High: CXR",
            "Missing Consults": "- Medium: Nephrology consult",
            "Care checks (deterministic)": "- High: Ventilator review",
        }

        docx_bytes, filename = course_docx.generate_course_docx(
            unit_name="MICU",
            patient_label="Bed 5 | P-5",
            patient_key="pid:p5",
            chronological_rows=history,
            selected_output=selected_output,
            resource_matches=[],
        )

        self.assertTrue(filename.endswith(".docx"))
        self.assertGreater(len(docx_bytes), 1000)

        document = Document(BytesIO(docx_bytes))
        full_text = "\n".join(p.text for p in document.paragraphs)
        self.assertIn("ICU Course History and Handoff Summary", full_text)
        self.assertIn("Structured SBAR Handoff", full_text)
        self.assertIn("Pending and Missed-Item Crosscheck", full_text)


class CourseDocxSafeFallbackTests(unittest.TestCase):
    def test_generate_course_docx_safe_returns_zip_when_renderer_unavailable(self) -> None:
        history = [
            {
                "date": "2026-02-14",
                "shift": "Morning",
                "status": "SICK",
                "diagnosis": "Sepsis",
                "supports": "O2",
                "new_issues": "AKI",
                "pending": "EEG pending",
                "key_labs_imaging": "Lactate 2.5",
                "plan_next_12h": "ABG repeat",
            }
        ]
        original_document = course_docx.Document
        course_docx.Document = None
        try:
            docx_bytes, filename, warning = course_docx.generate_course_docx_safe(
                unit_name="MICU",
                patient_label="Bed 5 | P-5",
                patient_key="pid:p5",
                chronological_rows=history,
                selected_output=None,
                resource_matches=[],
            )
        finally:
            course_docx.Document = original_document

        self.assertIsNone(warning)
        self.assertTrue(filename.endswith(".docx"))
        self.assertTrue(docx_bytes.startswith(b"PK"))
        self.assertGreater(len(docx_bytes), 500)

    def test_generate_course_docx_safe_handles_empty_history(self) -> None:
        docx_bytes, filename, warning = course_docx.generate_course_docx_safe(
            unit_name="MICU",
            patient_label="Bed 3 | No ID",
            patient_key="bed:3",
            chronological_rows=[],
            selected_output=None,
            resource_matches=[],
        )
        self.assertIsNotNone(warning)
        self.assertTrue(filename.endswith(".docx"))
        self.assertTrue(docx_bytes.startswith(b"PK"))


if __name__ == "__main__":
    unittest.main()
