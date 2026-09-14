"""Append-only autonomous benchmark decision ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def append_state(
    path: Path,
    *,
    run_id: str,
    bottleneck: str,
    hypothesis: str,
    variable: str,
    decision: str,
) -> None:
    if decision not in {"keep", "revert", "inconclusive"}:
        raise ValueError("decision must be keep, revert, or inconclusive")
    if not path.exists():
        path.write_text("# Servebench Experiment State\n\n", encoding="utf-8")
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    entry = (
        f"## {run_id} — {timestamp}\n\n"
        f"- Bottleneck: {bottleneck}\n"
        f"- Hypothesis: {hypothesis}\n"
        f"- Variable changed: {variable}\n"
        f"- Decision: {decision}\n\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)

