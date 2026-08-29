"""Deadline detection: find date expressions and resolve them to real dates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

# Ordered most-specific first: an absolute date beats a relative phrase.
ABSOLUTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})\.?,?\s+(\d{{4}})\b", re.IGNORECASE),
    re.compile(rf"\b({MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE),
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b"),
    re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2})\b"),
)

RELATIVE_PATTERN = re.compile(
    r"\bwithin\s+(\d{1,3})\s+(working\s+days|days|weeks|months|hours)\b", re.IGNORECASE
)
RELATIVE_WORD_PATTERN = re.compile(
    r"\b(immediately|with immediate effect|forthwith|today|tomorrow|"
    r"end of this month|end of the month|next week|this week)\b",
    re.IGNORECASE,
)
MONTH_YEAR_PATTERN = re.compile(rf"\b({MONTHS})\.?\s+(\d{{4}})\b", re.IGNORECASE)

MONTH_NUMBERS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

DEADLINE_TYPE_SIGNALS: dict[str, tuple[str, ...]] = {
    "Compliance": ("compliance", "statutory", "regulation", "directive", "cmrs", "mandatory"),
    "Inspection": ("inspection", "inspect", "audit", "survey", "check"),
    "Renewal": ("renewal", "renew", "revalidate", "extension of validity"),
    "Contract Expiry": ("contract expiry", "expires", "expiry", "valid until", "validity"),
    "Response": ("reply", "response", "respond", "revert", "show cause", "clarification"),
    "Payment": ("payment", "invoice", "remit", "release payment", "due amount"),
    "Submission": ("submit", "submission", "furnish", "upload", "report to be"),
}


@dataclass
class DateMatch:
    text: str
    value: date | None
    is_relative: bool = False
    start: int = 0

    @property
    def resolved(self) -> bool:
        return self.value is not None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_absolute(match: re.Match[str], pattern_index: int) -> date | None:
    groups = match.groups()
    if pattern_index == 0:  # 15 September 2026
        day, month_name, year = groups
        month = MONTH_NUMBERS.get(month_name.lower().rstrip("."))
        return _safe_date(int(year), month, int(day)) if month else None
    if pattern_index == 1:  # September 15, 2026
        month_name, day, year = groups
        month = MONTH_NUMBERS.get(month_name.lower().rstrip("."))
        return _safe_date(int(year), month, int(day)) if month else None
    if pattern_index == 2:  # 2026-09-15
        year, month, day = groups
        return _safe_date(int(year), int(month), int(day))
    if pattern_index == 3:  # 15/09/2026 (day-first, the Indian convention)
        first, second, year = (int(g) for g in groups)
        if first > 12 and second <= 12:
            return _safe_date(year, second, first)
        if second > 12 and first <= 12:
            return _safe_date(year, first, second)
        return _safe_date(year, second, first)
    if pattern_index == 4:  # 15/09/26
        first, second, short_year = (int(g) for g in groups)
        year = 2000 + short_year if short_year < 70 else 1900 + short_year
        return _safe_date(year, second, first)
    return None


def _parse_relative(match: re.Match[str], reference: date) -> date | None:
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if "hour" in unit:
        return reference + timedelta(days=max(1, round(amount / 24)))
    if "week" in unit:
        return reference + timedelta(weeks=amount)
    if "month" in unit:
        return reference + timedelta(days=amount * 30)
    if "working" in unit:
        # Five working days per week, so ~7 calendar days per 5 working days.
        return reference + timedelta(days=round(amount * 7 / 5))
    return reference + timedelta(days=amount)


def _parse_relative_word(phrase: str, reference: date) -> date | None:
    lowered = phrase.lower()
    if lowered in {"immediately", "with immediate effect", "forthwith", "today"}:
        return reference
    if lowered == "tomorrow":
        return reference + timedelta(days=1)
    if lowered in {"this week", "next week"}:
        return reference + timedelta(days=7 if lowered == "next week" else 3)
    if lowered in {"end of this month", "end of the month"}:
        if reference.month == 12:
            return date(reference.year, 12, 31)
        return date(reference.year, reference.month + 1, 1) - timedelta(days=1)
    return None


def find_dates(text: str, reference: date | None = None) -> list[DateMatch]:
    """Every date expression in ``text``, resolved where possible."""
    reference = reference or datetime.utcnow().date()
    matches: list[DateMatch] = []
    claimed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < c_end and end > c_start for c_start, c_end in claimed)

    for index, pattern in enumerate(ABSOLUTE_PATTERNS):
        for match in pattern.finditer(text):
            if overlaps(match.start(), match.end()):
                continue
            claimed.append((match.start(), match.end()))
            matches.append(
                DateMatch(text=match.group(0), value=_parse_absolute(match, index), start=match.start())
            )

    for match in RELATIVE_PATTERN.finditer(text):
        if overlaps(match.start(), match.end()):
            continue
        claimed.append((match.start(), match.end()))
        matches.append(
            DateMatch(
                text=match.group(0),
                value=_parse_relative(match, reference),
                is_relative=True,
                start=match.start(),
            )
        )

    for match in MONTH_YEAR_PATTERN.finditer(text):
        if overlaps(match.start(), match.end()):
            continue
        month = MONTH_NUMBERS.get(match.group(1).lower().rstrip("."))
        if not month:
            continue
        claimed.append((match.start(), match.end()))
        matches.append(
            DateMatch(
                text=match.group(0),
                value=_safe_date(int(match.group(2)), month, 1),
                start=match.start(),
            )
        )

    for match in RELATIVE_WORD_PATTERN.finditer(text):
        if overlaps(match.start(), match.end()):
            continue
        claimed.append((match.start(), match.end()))
        matches.append(
            DateMatch(
                text=match.group(0),
                value=_parse_relative_word(match.group(0), reference),
                is_relative=True,
                start=match.start(),
            )
        )

    matches.sort(key=lambda item: item.start)
    return matches


def first_date(text: str, reference: date | None = None) -> DateMatch | None:
    matches = find_dates(text, reference)
    return matches[0] if matches else None


def parse_date(value: str | None, reference: date | None = None) -> date | None:
    """Resolve a single date string (also used for user-entered due dates)."""
    if not value:
        return None
    cleaned = value.strip()
    try:
        return date.fromisoformat(cleaned[:10])
    except ValueError:
        pass
    match = first_date(cleaned, reference)
    return match.value if match else None


def classify_deadline(text: str) -> str:
    lowered = (text or "").lower()
    for deadline_type, signals in DEADLINE_TYPE_SIGNALS.items():
        if any(signal in lowered for signal in signals):
            return deadline_type
    return "General"


def days_until(target: date | None, reference: date | None = None) -> int | None:
    if target is None:
        return None
    return (target - (reference or datetime.utcnow().date())).days


def urgency_bucket(target: date | None, reference: date | None = None) -> str:
    """The bucket a deadline falls into, for alerts and the compliance board."""
    delta = days_until(target, reference)
    if delta is None:
        return "undated"
    if delta < 0:
        return "overdue"
    if delta == 0:
        return "due_today"
    if delta <= 3:
        return "due_3_days"
    if delta <= 7:
        return "due_7_days"
    return "upcoming"
