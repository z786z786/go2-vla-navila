"""P5 four-field action target contract for NaVILA R2R records."""
from __future__ import annotations

from actions.language_parser import parse_nav_command
from typing import Any

from actions.nav_command import NavCommand

def target_from_record(record: dict[str, Any]) -> NavCommand:
    """Validate adapter output; this is not a continuous-head training target."""
    result = parse_nav_command(str(record["action_text"]))
    if result.classification != "exact":
        raise ValueError("Training action_text must match the frozen vocabulary")
    return result.command


def targets_from_records(records: list[dict[str, Any]]) -> list[NavCommand]:
    """Build and validate the complete four-field target batch."""
    return [target_from_record(record) for record in records]
