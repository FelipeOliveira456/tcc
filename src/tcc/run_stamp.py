"""Marcadores de tempo para execuções de inferência e fine-tune."""

from __future__ import annotations

from datetime import UTC, datetime


def run_stamp(when: datetime | None = None) -> str:
    """ID compacto de execução: YYYYMMDD_HHMMSS (UTC)."""
    dt = when or datetime.now(UTC)
    return dt.strftime("%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
