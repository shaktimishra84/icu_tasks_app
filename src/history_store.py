from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any

SHIFT_RANK = {
    "Morning": 0,
    "Evening": 1,
    "Night": 2,
}

MISSING_OUTCOME_VALUES = {"Death", "DAMA", "Discharge", "Shifted"}

BED_STATE_FIELDS = [
    "bed",
    "diagnosis",
    "status",
    "supports",
    "new_issues",
    "actions_done",
    "plan_next_12h",
    "pending",
    "key_labs_imaging",
]


def hash_payload(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _normalized_bed(value: Any) -> str:
    raw = _normalized_text(value)
    if not raw:
        return ""
    match = re.search(r"(\d+)", raw)
    return match.group(1) if match else raw


def _coerce_shift(value: Any) -> str:
    text = _normalized_text(value).title()
    return text if text in SHIFT_RANK else "Morning"


def _coerce_missing_outcome(value: Any) -> str:
    raw = _normalized_text(value)
    if not raw:
        return ""
    normalized = raw.upper()
    if normalized == "DAMA":
        return "DAMA"
    title = raw.title()
    if title in MISSING_OUTCOME_VALUES:
        return title
    raise ValueError("Outcome must be one of: Death, DAMA, Discharge, Shifted.")


def _shift_rank(value: Any) -> int:
    return SHIFT_RANK.get(_coerce_shift(value), 0)


def _shift_rank_sql(column_name: str = "shift") -> str:
    return (
        f"CASE {column_name} "
        "WHEN 'Morning' THEN 0 "
        "WHEN 'Evening' THEN 1 "
        "WHEN 'Night' THEN 2 "
        "ELSE 9 END"
    )


def patient_key_from_source_row(row: dict[str, Any]) -> str:
    patient_key = _normalized_key(row.get("patient_id", ""))
    if patient_key:
        return f"pid:{patient_key}"

    bed_key = _normalized_key(_normalized_bed(row.get("bed", "")))
    if bed_key:
        return f"bed:{bed_key}"
    return ""


def patient_key_from_output_row(row: dict[str, Any]) -> str:
    patient_key = _normalized_key(row.get("Patient ID", ""))
    if patient_key:
        return f"pid:{patient_key}"

    bed_key = _normalized_key(_normalized_bed(row.get("Bed", "")))
    if bed_key:
        return f"bed:{bed_key}"
    return ""


def _state_from_source_row(row: dict[str, Any]) -> dict[str, str] | None:
    patient_key = patient_key_from_source_row(row)
    if not patient_key:
        return None

    state: dict[str, str] = {
        "patient_key": patient_key,
        "patient_id": _normalized_text(row.get("patient_id", "")),
        "bed": _normalized_bed(row.get("bed", "")),
    }
    for field in BED_STATE_FIELDS[1:]:
        state[field] = _normalized_text(row.get(field, ""))
    return state


def _dict_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class ICUHistoryStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    shift TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, shift)
                );

                CREATE TABLE IF NOT EXISTS patients (
                    patient_key TEXT PRIMARY KEY,
                    patient_id TEXT,
                    first_seen_date TEXT NOT NULL,
                    last_seen_date TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bed_state (
                    snapshot_id INTEGER NOT NULL,
                    patient_key TEXT NOT NULL,
                    bed TEXT,
                    diagnosis TEXT,
                    status TEXT,
                    supports TEXT,
                    new_issues TEXT,
                    actions_done TEXT,
                    plan_next_12h TEXT,
                    pending TEXT,
                    key_labs_imaging TEXT,
                    PRIMARY KEY (snapshot_id, patient_key),
                    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
                    FOREIGN KEY(patient_key) REFERENCES patients(patient_key)
                );

                CREATE INDEX IF NOT EXISTS idx_snapshots_date_shift
                    ON snapshots(date, shift);
                CREATE INDEX IF NOT EXISTS idx_bed_state_snapshot
                    ON bed_state(snapshot_id);
                CREATE INDEX IF NOT EXISTS idx_bed_state_patient
                    ON bed_state(patient_key);

                CREATE TABLE IF NOT EXISTS missing_patient_outcomes (
                    snapshot_id INTEGER NOT NULL,
                    patient_key TEXT NOT NULL,
                    patient_id TEXT,
                    bed TEXT,
                    last_status TEXT,
                    outcome TEXT NOT NULL,
                    notes TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (snapshot_id, patient_key),
                    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
                    FOREIGN KEY(patient_key) REFERENCES patients(patient_key)
                );

                CREATE INDEX IF NOT EXISTS idx_missing_outcomes_snapshot
                    ON missing_patient_outcomes(snapshot_id);
                """
            )

    def save_snapshot(
        self,
        *,
        snapshot_date: str,
        shift: str,
        file_hash: str,
        table_rows: list[dict[str, Any]],
    ) -> int:
        shift_value = _coerce_shift(shift)
        date_value = _normalized_text(snapshot_date)

        state_by_key: dict[str, dict[str, str]] = {}
        for row in table_rows:
            state = _state_from_source_row(row)
            if state is None:
                continue
            key = state["patient_key"]
            if key in state_by_key:
                existing = state_by_key[key]
                for field in BED_STATE_FIELDS:
                    if not existing.get(field, "") and state.get(field, ""):
                        existing[field] = state[field]
                if not existing.get("patient_id", "") and state.get("patient_id", ""):
                    existing["patient_id"] = state["patient_id"]
                continue
            state_by_key[key] = state

        if not state_by_key:
            raise ValueError("No valid bed rows found for snapshot save.")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO snapshots(date, shift, file_hash)
                VALUES (?, ?, ?)
                ON CONFLICT(date, shift)
                DO UPDATE SET
                    file_hash = excluded.file_hash,
                    created_at = CURRENT_TIMESTAMP
                """,
                (date_value, shift_value, file_hash),
            )
            snapshot_row = conn.execute(
                "SELECT snapshot_id FROM snapshots WHERE date = ? AND shift = ?",
                (date_value, shift_value),
            ).fetchone()
            if snapshot_row is None:
                raise RuntimeError("Snapshot write failed.")
            snapshot_id = int(snapshot_row["snapshot_id"])

            conn.execute("DELETE FROM bed_state WHERE snapshot_id = ?", (snapshot_id,))

            for state in state_by_key.values():
                conn.execute(
                    """
                    INSERT INTO patients(patient_key, patient_id, first_seen_date, last_seen_date)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(patient_key)
                    DO UPDATE SET
                        patient_id = CASE
                            WHEN excluded.patient_id <> '' THEN excluded.patient_id
                            ELSE patients.patient_id
                        END,
                        last_seen_date = excluded.last_seen_date
                    """,
                    (
                        state["patient_key"],
                        state["patient_id"],
                        date_value,
                        date_value,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO bed_state(
                        snapshot_id,
                        patient_key,
                        bed,
                        diagnosis,
                        status,
                        supports,
                        new_issues,
                        actions_done,
                        plan_next_12h,
                        pending,
                        key_labs_imaging
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        state["patient_key"],
                        state.get("bed", ""),
                        state.get("diagnosis", ""),
                        state.get("status", ""),
                        state.get("supports", ""),
                        state.get("new_issues", ""),
                        state.get("actions_done", ""),
                        state.get("plan_next_12h", ""),
                        state.get("pending", ""),
                        state.get("key_labs_imaging", ""),
                    ),
                )

        return snapshot_id

    def get_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT snapshot_id, date, shift, file_hash, created_at FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        return _dict_row(row)

    def get_latest_snapshot(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT snapshot_id, date, shift, file_hash, created_at
                FROM snapshots
                ORDER BY date DESC, {_shift_rank_sql('shift')} DESC
                LIMIT 1
                """
            ).fetchone()
        return _dict_row(row)

    def get_previous_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        current = self.get_snapshot(snapshot_id)
        if current is None:
            return None

        current_date = str(current.get("date", ""))
        current_rank = _shift_rank(current.get("shift", "Morning"))

        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT snapshot_id, date, shift, file_hash, created_at
                FROM snapshots
                WHERE date < ?
                   OR (date = ? AND {_shift_rank_sql('shift')} < ?)
                ORDER BY date DESC, {_shift_rank_sql('shift')} DESC
                LIMIT 1
                """,
                (current_date, current_date, current_rank),
            ).fetchone()
        return _dict_row(row)

    def list_snapshots(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT snapshot_id, date, shift, file_hash, created_at
                FROM snapshots
                ORDER BY date DESC, {_shift_rank_sql('shift')} DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_snapshot_rows(self, snapshot_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    bs.snapshot_id,
                    bs.patient_key,
                    COALESCE(NULLIF(p.patient_id, ''), '') AS patient_id,
                    bs.bed,
                    bs.diagnosis,
                    bs.status,
                    bs.supports,
                    bs.new_issues,
                    bs.actions_done,
                    bs.plan_next_12h,
                    bs.pending,
                    bs.key_labs_imaging
                FROM bed_state AS bs
                LEFT JOIN patients AS p
                    ON p.patient_key = bs.patient_key
                WHERE bs.snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchall()

        def _sort_key(row: sqlite3.Row) -> tuple[int, int | str, str, str]:
            bed = str(row["bed"] or "").strip()
            match = re.search(r"\d+", bed)
            if match:
                return (0, int(match.group(0)), bed, str(row["patient_key"]))
            if bed:
                return (1, bed, bed, str(row["patient_key"]))
            return (2, "", "", str(row["patient_key"]))

        ordered = sorted(rows, key=_sort_key)
        return [dict(row) for row in ordered]

    def get_patient_history(self, patient_key: str, limit: int = 7) -> list[dict[str, Any]]:
        if not patient_key:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    s.snapshot_id,
                    s.date,
                    s.shift,
                    bs.patient_key,
                    COALESCE(NULLIF(p.patient_id, ''), '') AS patient_id,
                    bs.bed,
                    bs.diagnosis,
                    bs.status,
                    bs.supports,
                    bs.new_issues,
                    bs.actions_done,
                    bs.plan_next_12h,
                    bs.pending,
                    bs.key_labs_imaging
                FROM bed_state AS bs
                JOIN snapshots AS s
                    ON s.snapshot_id = bs.snapshot_id
                LEFT JOIN patients AS p
                    ON p.patient_key = bs.patient_key
                WHERE bs.patient_key = ?
                ORDER BY s.date DESC, {_shift_rank_sql('s.shift')} DESC
                LIMIT ?
                """,
                (patient_key, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_missing_outcome(
        self,
        *,
        snapshot_id: int,
        patient_key: str,
        patient_id: str,
        bed: str,
        last_status: str,
        outcome: str,
        notes: str = "",
    ) -> None:
        key = _normalized_text(patient_key)
        if not key:
            raise ValueError("Patient key is required.")

        normalized_outcome = _coerce_missing_outcome(outcome)
        if not normalized_outcome:
            self.delete_missing_outcome(snapshot_id=snapshot_id, patient_key=key)
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO missing_patient_outcomes(
                    snapshot_id,
                    patient_key,
                    patient_id,
                    bed,
                    last_status,
                    outcome,
                    notes,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(snapshot_id, patient_key)
                DO UPDATE SET
                    patient_id = excluded.patient_id,
                    bed = excluded.bed,
                    last_status = excluded.last_status,
                    outcome = excluded.outcome,
                    notes = excluded.notes,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    int(snapshot_id),
                    key,
                    _normalized_text(patient_id),
                    _normalized_bed(bed),
                    _normalized_text(last_status),
                    normalized_outcome,
                    _normalized_text(notes),
                ),
            )

    def delete_missing_outcome(self, *, snapshot_id: int, patient_key: str) -> None:
        key = _normalized_text(patient_key)
        if not key:
            return
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM missing_patient_outcomes
                WHERE snapshot_id = ? AND patient_key = ?
                """,
                (int(snapshot_id), key),
            )

    def get_missing_outcomes(self, snapshot_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    snapshot_id,
                    patient_key,
                    patient_id,
                    bed,
                    last_status,
                    outcome,
                    notes,
                    updated_at
                FROM missing_patient_outcomes
                WHERE snapshot_id = ?
                ORDER BY bed ASC, patient_key ASC
                """,
                (int(snapshot_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear_all(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM missing_patient_outcomes")
            conn.execute("DELETE FROM bed_state")
            conn.execute("DELETE FROM patients")
            conn.execute("DELETE FROM snapshots")
