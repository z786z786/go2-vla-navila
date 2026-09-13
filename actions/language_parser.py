"""Regex-and-substring parser for the frozen NaVILA action vocabulary.

The parser intentionally contains no model, I/O, or network dependency.  Its
native-policy branch preserves NaVILA's substring-based number matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import re

from .nav_command import NavCommand, PI_OVER_SIX, duration_to_hold_steps


ParseClassification = Literal["exact", "partial_match", "total_miss"]
FallbackPolicy = Literal["layered", "native", "brake"]
FALLBACK_POLICIES = frozenset({"layered", "native", "brake"})

_DURATION_BY_HOLD_STEPS = {25: 0.5, 50: 1.0, 75: 1.5}
_SPACE_RE = re.compile(r"\s+")


def _motion_command(vx: float, wz: float, hold_steps: int) -> NavCommand:
    return NavCommand(
        vx=vx,
        vy=0.0,
        wz=wz,
        hold_steps=duration_to_hold_steps(_DURATION_BY_HOLD_STEPS[hold_steps]),
        stop=False,
    )


_STOP_COMMAND = NavCommand(vx=0.0, vy=0.0, wz=0.0, hold_steps=0, stop=True)
_BRAKE_COMMAND = _motion_command(vx=0.0, wz=0.0, hold_steps=25)

# Normalized forms are used only to recognize the ten frozen vocabulary items.
# This permits harmless capitalization/spacing/final-period variations before
# falling through to the native-compatible raw-substring state machine.
_EXACT_VOCABULARY: dict[str, NavCommand] = {
    "the next action is move forward 25 cm": _motion_command(0.5, 0.0, 25),
    "the next action is move forward 50 cm": _motion_command(0.5, 0.0, 50),
    "the next action is move forward 75 cm": _motion_command(0.5, 0.0, 75),
    "the next action is turn left 15 degree": _motion_command(0.0, PI_OVER_SIX, 25),
    "the next action is turn left 30 degree": _motion_command(0.0, PI_OVER_SIX, 50),
    "the next action is turn left 45 degree": _motion_command(0.0, PI_OVER_SIX, 75),
    "the next action is turn right 15 degree": _motion_command(0.0, -PI_OVER_SIX, 25),
    "the next action is turn right 30 degree": _motion_command(0.0, -PI_OVER_SIX, 50),
    "the next action is turn right 45 degree": _motion_command(0.0, -PI_OVER_SIX, 75),
    "i think i should stop because i have finished the instruction": _STOP_COMMAND,
}


@dataclass(frozen=True)
class ParseCounters:
    """Tiered accounting that keeps fallback use visible to callers."""

    exact: int = 0
    partial_match: int = 0
    total_miss: int = 0

    @property
    def total(self) -> int:
        return self.exact + self.partial_match + self.total_miss

    @property
    def parse_error_rate(self) -> float:
        return (
            (self.partial_match + self.total_miss) / self.total
            if self.total
            else 0.0
        )

    def increment(self, classification: ParseClassification) -> "ParseCounters":
        if classification == "exact":
            return ParseCounters(self.exact + 1, self.partial_match, self.total_miss)
        if classification == "partial_match":
            return ParseCounters(self.exact, self.partial_match + 1, self.total_miss)
        return ParseCounters(self.exact, self.partial_match, self.total_miss + 1)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "exact": self.exact,
            "partial_match": self.partial_match,
            "total_miss": self.total_miss,
            "total": self.total,
            "parse_error_rate": self.parse_error_rate,
        }


@dataclass(frozen=True)
class ParseResult:
    """One deterministic parse plus its mandatory classification tier."""

    command: NavCommand
    classification: ParseClassification
    parse_failure: bool
    matched_branch: Literal["turn_left", "turn_right", "move", "stop", "none"]
    used_fallback: bool


def _normalize_for_exact_match(text: str) -> str:
    normalized = _SPACE_RE.sub(" ", text.strip().lower())
    # Treat whitespace immediately before the optional sentence period as
    # formatting, too (e.g. ``"25 cm ."``).
    return re.sub(r"\s*\.$", "", normalized)


def _native_branch(lowered_text: str) -> Literal[
    "turn_left", "turn_right", "move", "stop", "none"
]:
    """Mirror the native branch priority and its deliberate substring rules."""
    if "turn left" in lowered_text:
        return "turn_left"
    if "turn right" in lowered_text:
        return "turn_right"
    if "move forward" in lowered_text or "move" in lowered_text:
        return "move"
    if "stop" in lowered_text:
        return "stop"
    return "none"


def _native_branch_command(
    branch: Literal["turn_left", "turn_right", "move", "stop", "none"],
    lowered_text: str,
) -> tuple[NavCommand, bool]:
    """Return (command, used_native_branch_fallback) for a selected branch."""
    if branch == "turn_left":
        # Preserve NaVILA's exact priority and substring quirk: e.g. "145"
        # takes the 45-degree branch because "45" is contained in the text.
        if "45" in lowered_text:
            return _motion_command(0.0, PI_OVER_SIX, 75), False
        if "30" in lowered_text:
            return _motion_command(0.0, PI_OVER_SIX, 50), False
        if "15" in lowered_text:
            return _motion_command(0.0, PI_OVER_SIX, 25), False
        return _motion_command(0.0, PI_OVER_SIX, 25), True
    if branch == "turn_right":
        if "45" in lowered_text:
            return _motion_command(0.0, -PI_OVER_SIX, 75), False
        if "30" in lowered_text:
            return _motion_command(0.0, -PI_OVER_SIX, 50), False
        if "15" in lowered_text:
            return _motion_command(0.0, -PI_OVER_SIX, 25), False
        return _motion_command(0.0, -PI_OVER_SIX, 25), True
    if branch == "move":
        if "75" in lowered_text:
            return _motion_command(0.5, 0.0, 75), False
        if "50" in lowered_text:
            return _motion_command(0.5, 0.0, 50), False
        if "25" in lowered_text:
            return _motion_command(0.5, 0.0, 25), False
        return _motion_command(0.5, 0.0, 25), True
    if branch == "stop":
        return _STOP_COMMAND, False
    # The final native fallback is forward 25 cm; policy selection below may
    # replace it with the zero-velocity limiter-brake intent.
    return _motion_command(0.5, 0.0, 25), True


def _branch_for_exact_command(command: NavCommand) -> Literal[
    "turn_left", "turn_right", "move", "stop", "none"
]:
    if command.stop:
        return "stop"
    if command.vx > 0.0:
        return "move"
    return "turn_left" if command.wz > 0.0 else "turn_right"


class LanguageParser:
    """Stateful parser with cumulative, inspectable tiered error counters."""

    def __init__(self, fallback_policy: FallbackPolicy = "layered") -> None:
        if fallback_policy not in FALLBACK_POLICIES:
            raise ValueError(
                f"fallback_policy must be one of {sorted(FALLBACK_POLICIES)}, "
                f"got {fallback_policy!r}"
            )
        self.fallback_policy = fallback_policy
        self._counters = ParseCounters()

    @property
    def counters(self) -> ParseCounters:
        """A snapshot of exact / partial_match / total_miss accumulated so far."""
        return self._counters

    @property
    def parse_error_rate(self) -> float:
        return self._counters.parse_error_rate

    def reset_counters(self) -> None:
        self._counters = ParseCounters()

    def parse(self, text: str) -> ParseResult:
        """Classify and parse *text* without hiding a fallback behind success."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        exact_command = _EXACT_VOCABULARY.get(_normalize_for_exact_match(text))
        if exact_command is not None:
            result = ParseResult(
                command=exact_command,
                classification="exact",
                parse_failure=False,
                matched_branch=_branch_for_exact_command(exact_command),
                used_fallback=False,
            )
        else:
            lowered_text = text.lower()
            branch = _native_branch(lowered_text)
            native_command, used_native_fallback = _native_branch_command(
                branch, lowered_text
            )
            classification: ParseClassification = (
                "total_miss" if branch == "none" else "partial_match"
            )
            use_brake = (
                self.fallback_policy == "brake" and used_native_fallback
            ) or (
                self.fallback_policy == "layered" and branch == "none"
            )
            result = ParseResult(
                command=_BRAKE_COMMAND if use_brake else native_command,
                classification=classification,
                parse_failure=True,
                matched_branch=branch,
                used_fallback=used_native_fallback,
            )

        self._counters = self._counters.increment(result.classification)
        return result

    def parse_command(self, text: str) -> NavCommand:
        """Parse and return only the controller command while still counting tiers."""
        return self.parse(text).command


def parse_nav_command(
    text: str, fallback_policy: FallbackPolicy = "layered"
) -> ParseResult:
    """Convenience single-shot parse retaining classification in its result."""
    return LanguageParser(fallback_policy=fallback_policy).parse(text)
