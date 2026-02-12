from __future__ import annotations

import os
import re
from typing import Sequence

from openai import OpenAI

from src.knowledge_base import ResourceChunk


SYSTEM_PROMPT = """
You are an ICU workflow assistant for licensed clinicians.
Return output in markdown with exactly 4 sections and no extra section.
The 4 section titles must be exactly:
1. One-paragraph de-identified summary
2. Timeline
3. Next 12h tasks
4. Missed items checklist

Rules:
- Section 1 must be a single paragraph.
- Section 2 must be date/time ordered bullet points; if exact date/time unavailable, use relative order.
- Section 3 must be prioritized actionable bullets.
- Section 4 must be checklist bullets. Each bullet must include a short reason in parentheses.
- Do not invent facts. Be explicit when data is missing.
- Keep advice as clinical decision support, not definitive diagnosis.
"""


def _truncate(text: str, limit: int = 12000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _extract_output_text(response: object) -> str:
    output_text = getattr(response, "output_text", "")
    if output_text:
        return output_text.strip()

    parts: list[str] = []
    for item in getattr(response, "output", []):
        for content_item in getattr(item, "content", []):
            text = getattr(content_item, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _fallback(patient_text: str, resources: Sequence[ResourceChunk], error_detail: str | None = None) -> str:
    text = patient_text.lower()
    timeline = [
        "- Earliest available note: presenting complaint and initial status documented.",
        "- Subsequent events: interventions, monitoring changes, and pending results should be ordered by time.",
    ]
    next_tasks = [
        "- [High] Reassess ABCs and trend vitals with neuro checks at defined intervals.",
        "- [High] Reconcile active problem list, medications, and recent labs/imaging already ordered.",
        "- [Medium] Confirm escalation triggers and consult criteria for deterioration in next 12 hours.",
    ]
    missed = []

    if "sepsis" in text and "blood culture" not in text:
        missed.append("- [ ] Blood cultures (needed before/with early antibiotic pathway in suspected sepsis).")
    if "chest pain" in text and "ecg" not in text and "ekg" not in text:
        missed.append("- [ ] ECG (screens for ischemia/arrhythmia in chest pain).")
    if "hypoxia" in text and "chest x-ray" not in text and "cxr" not in text:
        missed.append("- [ ] Chest X-ray (helps identify pulmonary causes of hypoxia).")
    if "stroke" in text and "ct head" not in text:
        missed.append("- [ ] CT head (rules out hemorrhage/acute intracranial process).")

    if not missed:
        missed.append("- [ ] No obvious gap from text alone (verify against full orders and pending studies).")

    if error_detail:
        timeline.append(f"- System note: model call fallback used ({error_detail}).")

    unique_sources: list[str] = []
    for chunk in resources:
        marker = f"{chunk.file_name} p.{chunk.page_number}"
        if marker not in unique_sources:
            unique_sources.append(marker)
    source_hint = ", ".join(unique_sources[:3]) if unique_sources else "no matched indexed chunks"

    return (
        "### 1. One-paragraph de-identified summary\n"
        f"De-identified case synopsis generated from provided note; potential identifiers should be replaced with neutral placeholders, and current status, key problems, and immediate risks should be verified against the chart before action (source context: {source_hint}).\n\n"
        "### 2. Timeline\n"
        f"{chr(10).join(timeline)}\n\n"
        "### 3. Next 12h tasks\n"
        f"{chr(10).join(next_tasks)}\n\n"
        "### 4. Missed items checklist\n"
        f"{chr(10).join(missed)}"
    )


def _enforce_four_sections(markdown_text: str) -> str:
    titles = {
        "1": "### 1. One-paragraph de-identified summary",
        "2": "### 2. Timeline",
        "3": "### 3. Next 12h tasks",
        "4": "### 4. Missed items checklist",
    }

    # Normalize minor heading variations from the model into exact required headers first.
    normalized = markdown_text.strip()
    replacements = {
        r"^#+\s*1[\).\-\s]*.*de-identified.*$": titles["1"],
        r"^#+\s*2[\).\-\s]*.*timeline.*$": titles["2"],
        r"^#+\s*3[\).\-\s]*.*12h.*$": titles["3"],
        r"^#+\s*4[\).\-\s]*.*missed.*$": titles["4"],
    }
    lines = normalized.splitlines()
    for index, line in enumerate(lines):
        for pattern, replacement in replacements.items():
            if re.match(pattern, line.strip(), flags=re.IGNORECASE):
                lines[index] = replacement
    normalized = "\n".join(lines).strip()

    # Rebuild strict 4-section output and drop any extra sections.
    sections: dict[str, list[str]] = {key: [] for key in ("1", "2", "3", "4")}
    current: str | None = None
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped == titles["1"]:
            current = "1"
            continue
        if stripped == titles["2"]:
            current = "2"
            continue
        if stripped == titles["3"]:
            current = "3"
            continue
        if stripped == titles["4"]:
            current = "4"
            continue
        if current is not None:
            sections[current].append(line)

    if not all(any(item.strip() for item in sections[key]) for key in ("1", "2", "3", "4")):
        return normalized

    rebuilt = [
        titles["1"],
        "\n".join(sections["1"]).strip(),
        "",
        titles["2"],
        "\n".join(sections["2"]).strip(),
        "",
        titles["3"],
        "\n".join(sections["3"]).strip(),
        "",
        titles["4"],
        "\n".join(sections["4"]).strip(),
    ]
    return "\n".join(rebuilt).strip()


class ClinicalTaskAdvisor:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._client = OpenAI(api_key=api_key) if api_key else None

    @property
    def llm_available(self) -> bool:
        return self._client is not None

    def _build_user_prompt(self, patient_text: str, resources: Sequence[ResourceChunk]) -> str:
        resource_blobs = []
        for index, chunk in enumerate(resources, start=1):
            resource_blobs.append(
                f"[Source {index}] {chunk.file_name} | page {chunk.page_number}\n{_truncate(chunk.text, limit=1800)}"
            )

        resources_section = "\n\n".join(resource_blobs) if resource_blobs else "No resource passages found."
        return f"""
PATIENT CASE (raw input, may contain identifiers):
{_truncate(patient_text)}

RETRIEVED GUIDELINE CHUNKS:
{resources_section}

Return markdown with exactly these headings and no additional headings:
### 1. One-paragraph de-identified summary
### 2. Timeline
### 3. Next 12h tasks
### 4. Missed items checklist
""".strip()

    def analyze(self, patient_text: str, resources: Sequence[ResourceChunk]) -> str:
        cleaned = patient_text.strip()
        if not cleaned:
            return "No patient information was provided."

        if self._client is None:
            return _fallback(cleaned, resources)

        try:
            prompt = self._build_user_prompt(cleaned, resources)
            response = self._client.responses.create(
                model=self.model,
                temperature=0.1,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT.strip()}]},
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
                ],
            )
            text = _extract_output_text(response)
            if not text:
                return _fallback(cleaned, resources, error_detail="No model text returned.")
            return _enforce_four_sections(text)
        except Exception as error:  # pragma: no cover - external API
            return _fallback(cleaned, resources, error_detail=str(error))

    def refine_bed_output(self, row_context: dict[str, str], draft_output: str) -> str:
        if self._client is None:
            return draft_output

        prompt = f"""
Refine the following ICU bed summary while preserving strict format.
Return exactly 5 lines, no extra lines.
Line 1: Bed + Patient ID
Line 2: Diagnosis (1 line)
Line 3: Status + Supports (1 line)
Line 4: Next 12h tasks: up to 3 concise items separated by ';'
Line 5: Missed items: up to 3 concise items separated by ';'
Do not add generic advice. Use only provided row details.

ROW DATA:
{row_context}

DRAFT:
{draft_output}
""".strip()

        try:
            response = self._client.responses.create(
                model=self.model,
                temperature=0.0,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": "You rewrite structured clinical text."}]},
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
                ],
            )
            text = _extract_output_text(response)
            if not text:
                return draft_output
            lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
            return "\n".join(lines[:5]) if lines else draft_output
        except Exception:
            return draft_output
