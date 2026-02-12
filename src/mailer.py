from __future__ import annotations

import os
import re
import smtplib
from dataclasses import dataclass
from datetime import date, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


class MailerConfigError(Exception):
    """Raised when email configuration is missing or invalid."""


class MailerSendError(Exception):
    """Raised when SMTP send fails."""


@dataclass
class MailerConfig:
    gmail_user: str
    gmail_app_password: str
    email_from: str
    email_to: list[str]


def load_local_env_file(path: Path = ENV_FILE) -> None:
    """
    Load key=value pairs from a local .env file into process env without overriding
    values that are already present.
    """
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (key not in os.environ or not os.environ.get(key, "").strip()):
            os.environ[key] = value


def _parse_recipients(raw: str) -> list[str]:
    recipients = [part.strip() for part in raw.split(",") if part.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for recipient in recipients:
        key = recipient.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(recipient)
    return deduped


def read_mailer_config() -> MailerConfig:
    load_local_env_file()

    gmail_user = os.getenv("GMAIL_USER", "").strip()
    gmail_app_password = re.sub(r"\s+", "", os.getenv("GMAIL_APP_PASSWORD", ""))
    email_to = _parse_recipients(os.getenv("EMAIL_TO", ""))
    email_from = os.getenv("EMAIL_FROM", "").strip() or gmail_user

    missing: list[str] = []
    if not gmail_user:
        missing.append("GMAIL_USER")
    if not gmail_app_password:
        missing.append("GMAIL_APP_PASSWORD")
    if not email_to:
        missing.append("EMAIL_TO")
    if not email_from:
        missing.append("EMAIL_FROM")

    if missing:
        raise MailerConfigError(f"Missing email config: {', '.join(missing)}")

    return MailerConfig(
        gmail_user=gmail_user,
        gmail_app_password=gmail_app_password,
        email_from=email_from,
        email_to=email_to,
    )


def build_rounds_subject(shift: str, run_date: date | None = None) -> str:
    use_date = run_date or datetime.now().date()
    normalized_shift = "Morning" if shift.strip().lower() == "morning" else "Evening"
    return f"ICU Rounds Summary – {use_date.isoformat()} – {normalized_shift}"


def render_rounds_email_preview(rows: list[dict[str, Any]], scope_label: str) -> str:
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"ICU Rounds Summary ({scope_label})",
        f"Generated: {now_text}",
        f"Beds included: {len(rows)}",
        "",
    ]

    for row in rows:
        lines.extend(
            [
                f"Bed {row.get('Bed', '')} | Patient ID: {row.get('Patient ID', '')}",
                f"Diagnosis: {row.get('Diagnosis', '') or '-'}",
                f"Status/Supports: {row.get('Status', '') or '-'} | {row.get('Supports', '') or '-'}",
                f"Missing Tests: {_inline(row.get('Missing Tests', ''))}",
                f"Missing Imaging: {_inline(row.get('Missing Imaging', ''))}",
                f"Missing Consults: {_inline(row.get('Missing Consults', ''))}",
                f"Care checks: {_inline(row.get('Care checks (deterministic)', ''))}",
                f"Pending (verbatim): {_inline(row.get('Pending (verbatim)', ''))}",
                f"Key labs/imaging: {row.get('Key labs/imaging (1 line)', '') or '-'}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def send_rounds_email(subject: str, body: str) -> list[str]:
    config = read_mailer_config()

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config.email_from
    message["To"] = ", ".join(config.email_to)
    message.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(config.gmail_user, config.gmail_app_password)
            server.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        raise MailerSendError("SMTP authentication failed. Check Gmail app password settings.") from exc
    except Exception as exc:
        raise MailerSendError("Email send failed. Check SMTP connectivity and recipient addresses.") from exc

    return config.email_to


def _inline(value: Any) -> str:
    cleaned = str(value or "").strip().replace("\n", " | ")
    return cleaned or "-"
