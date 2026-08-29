"""Document understanding.

Two interchangeable engines produce the same ``AnalysisResult``:

* ``claude``     — Anthropic Claude with a strict JSON schema (used when
                   ``ANTHROPIC_API_KEY`` is configured).
* ``rule-based`` — a deterministic KMRL-tuned analyser used as an automatic
                   fallback so the platform stays fully functional without a key.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..taxonomy import (
    ACTION_SIGNALS,
    DEPARTMENT_KEYWORDS,
    DEPARTMENTS,
    DOCUMENT_TYPE_KEYWORDS,
    DOCUMENT_TYPES,
    HIGH_PRIORITY_SIGNALS,
    MEDIUM_PRIORITY_SIGNALS,
    STOPWORDS,
    UNASSIGNED_DEPARTMENT,
    normalise_department,
    normalise_document_type,
    normalise_priority,
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    title: str = ""
    document_type: str = "Other"
    department: str = UNASSIGNED_DEPARTMENT
    secondary_departments: list[str] = field(default_factory=list)
    priority: str = "Medium"
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    actions: list[dict[str, str]] = field(default_factory=list)
    deadlines: list[dict[str, str]] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    compliance_flags: list[str] = field(default_factory=list)
    engine: str = "rule-based"
    model: str = ""
    confidence: float | None = None
    error: str = ""


# ---------------------------------------------------------------- text helpers
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_RE = re.compile(r"[A-Za-zഀ-ൿ][A-Za-z0-9ഀ-ൿ\-]{2,}")

DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|"
        r"November|December)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bwithin\s+\d{1,3}\s+(?:days|weeks|months|hours|working days)\b", re.IGNORECASE),
    re.compile(r"\bby\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
]

RISK_SIGNALS = [
    "risk", "penalty", "non-compliance", "non compliance", "delay", "failure",
    "hazard", "breach", "litigation", "shortfall", "defect", "unsafe",
    "escalation", "shutdown", "liability", "overrun",
]

COMPLIANCE_SIGNALS = [
    "commissioner of metro rail safety", "cmrs", "ministry of housing",
    "mohua", "statutory", "regulation", "audit", "gazette", "compliance",
    "mandatory", "directive", "circular", "gst", "rti",
]


def _reflow(text: str) -> str:
    """Undo hard line wrapping so sentence splitting sees whole sentences.

    Extracted PDF text arrives wrapped at the page width; without this, every
    printed line looks like its own sentence and summaries read as fragments.
    """
    paragraphs: list[str] = []
    buffer = ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            continue
        if (
            buffer
            and not buffer.endswith((".", "!", "?", ":", ";"))
            and len(buffer) >= 45
            # An all-caps letterhead is a banner, not the start of a sentence.
            and _uppercase_ratio(buffer) <= 0.7
        ):
            buffer = f"{buffer} {line}"
        else:
            if buffer:
                paragraphs.append(buffer)
            buffer = line
    if buffer:
        paragraphs.append(buffer)
    return "\n".join(paragraphs)


def _uppercase_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for character in letters if character.isupper()) / len(letters)


ACRONYMS = {
    "KMRL", "CMRS", "MOHUA", "OHE", "HR", "IT", "CBTC", "ATP", "ATO", "SCADA",
    "NIT", "EMD", "GST", "DPR", "UNS", "IOT", "OCC", "PSD", "AC", "DC", "UPS",
    "RFP", "EOI", "PO", "TS", "JC", "INV", "ERP", "API", "SOP", "PPE",
}
TITLE_MINOR_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or",
    "the", "to", "with",
}


def _smart_case(text: str) -> str:
    """Calm down SHOUTED letterhead lines without mangling acronyms."""
    if _uppercase_ratio(text) < 0.7:
        return text
    words: list[str] = []
    for index, word in enumerate(text.split()):
        stripped = word.strip(".,:;()[]")
        if stripped.upper() in ACRONYMS or (
            stripped.isupper() and stripped.isalpha() and not set("AEIOU") & set(stripped)
        ):
            words.append(word)  # keep KMRL, CMRS, OHE, S&T …
        elif index > 0 and word.lower() in TITLE_MINOR_WORDS:
            words.append(word.lower())
        else:
            words.append(_capitalise_token(word))
    return " ".join(words)


def _capitalise_token(word: str) -> str:
    """Capitalise the first letter, leaving punctuation and ordinals alone."""
    for index, character in enumerate(word):
        if character.isalpha():
            if word[:index].strip("([{\"'“‘").isdigit():
                return word.lower()  # 112TH -> 112th, not 112Th
            return word[:index] + character.upper() + word[index + 1 :].lower()
    return word


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text) if s and s.strip()]
    return [s for s in parts if len(s) > 25]


def _tokens(text: str) -> list[str]:
    return [
        word.lower()
        for word in WORD_RE.findall(text)
        if word.lower() not in STOPWORDS and not word.isdigit()
    ]


@lru_cache(maxsize=2048)
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Whole-word matcher — so "monitoring" never counts as a hit for "nit"."""
    return re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", re.IGNORECASE
    )


def _count_phrase(haystack: str, phrase: str) -> int:
    return len(_phrase_pattern(phrase).findall(haystack))


def _has_phrase(haystack: str, phrase: str) -> bool:
    return _phrase_pattern(phrase).search(haystack) is not None


def _count_signals(haystack: str, signals: list[str]) -> int:
    return sum(_count_phrase(haystack, signal) for signal in signals)


# ------------------------------------------------------------- rule-based engine
def analyse_with_rules(text: str, filename: str) -> AnalysisResult:
    result = AnalysisResult(engine="rule-based", model="kmrl-heuristics-v1")
    raw_body = (text or "").strip()
    body = _reflow(raw_body)
    haystack = f"{filename}\n{body}".lower()

    if not body:
        result.title = _title_from_filename(filename)
        result.summary = (
            "No machine-readable text could be extracted from this file, so the "
            "analysis is based on the filename only."
        )
        result.confidence = 0.15
        result.document_type = _guess_document_type(haystack, filename)
        return result

    # --- department -------------------------------------------------------
    scores: dict[str, int] = {}
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        score = sum(
            _count_phrase(haystack, keyword) * (2 if len(keyword) > 12 else 1)
            for keyword in keywords
        )
        if score:
            scores[dept] = score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if ranked:
        result.department = ranked[0][0]
        result.secondary_departments = [
            dept for dept, score in ranked[1:4] if score >= max(2, ranked[0][1] * 0.35)
        ]
    else:
        result.department = UNASSIGNED_DEPARTMENT

    # --- document type ----------------------------------------------------
    result.document_type = _guess_document_type(haystack, filename)

    # --- priority ---------------------------------------------------------
    high_hits = _count_signals(haystack, HIGH_PRIORITY_SIGNALS)
    medium_hits = _count_signals(haystack, MEDIUM_PRIORITY_SIGNALS)
    dated = any(pattern.search(body) for pattern in DATE_PATTERNS)
    if (
        high_hits >= 3
        or (high_hits >= 1 and result.department == "Safety")
        or (high_hits >= 2 and dated)
    ):
        result.priority = "High"
    elif high_hits >= 1 or medium_hits >= 2 or dated:
        result.priority = "Medium"
    else:
        result.priority = "Low"

    # --- keywords ---------------------------------------------------------
    counts = Counter(_tokens(body))
    result.keywords = [word for word, _ in counts.most_common(12)]

    # --- summary & key points --------------------------------------------
    sentences = _sentences(body)
    # Letterheads and all-caps banners carry no information for a summary.
    candidates = [s for s in sentences if _uppercase_ratio(s) <= 0.6] or sentences
    top_terms = {word for word, _ in counts.most_common(40)}
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(candidates[:400]):
        lowered = sentence.lower()
        overlap = sum(1 for word in set(_tokens(sentence)) if word in top_terms)
        signal_bonus = 2 if any(_has_phrase(lowered, sig) for sig in ACTION_SIGNALS) else 0
        position_bonus = 1.5 if index < 5 else 0
        length_penalty = 0.5 if len(sentence) > 320 else 0
        scored.append((overlap + signal_bonus + position_bonus - length_penalty, index, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = sorted(scored[:5], key=lambda item: item[1])
    summary_sentences = [item[2] for item in chosen]
    result.summary = _trim(" ".join(summary_sentences), 900) or body[:400]
    result.key_points = [_trim(s, 220) for s in summary_sentences[:4]]

    # --- actions ----------------------------------------------------------
    for sentence in sentences:
        lowered = sentence.lower()
        if any(_has_phrase(lowered, signal) for signal in ACTION_SIGNALS):
            due = _first_date(sentence)
            result.actions.append(
                {
                    "action": _trim(sentence, 260),
                    "owner": _owner_for(sentence) or (result.department if result.department != UNASSIGNED_DEPARTMENT else ""),
                    "due_date": due or "",
                    "priority": (
                        "High"
                        if any(_has_phrase(lowered, sig) for sig in HIGH_PRIORITY_SIGNALS)
                        else result.priority
                    ),
                }
            )
        if len(result.actions) >= 8:
            break

    # --- deadlines --------------------------------------------------------
    seen_dates: set[str] = set()
    for sentence in sentences:
        date = _first_date(sentence)
        if date and date.lower() not in seen_dates:
            seen_dates.add(date.lower())
            result.deadlines.append({"date": date, "description": _trim(sentence, 200)})
        if len(result.deadlines) >= 6:
            break

    # --- risks & compliance ----------------------------------------------
    for sentence in sentences:
        lowered = sentence.lower()
        if any(_has_phrase(lowered, signal) for signal in RISK_SIGNALS):
            result.risks.append(_trim(sentence, 200))
        if len(result.risks) >= 5:
            break

    result.compliance_flags = sorted(
        {
            signal.title() if signal.islower() else signal
            for signal in COMPLIANCE_SIGNALS
            if _has_phrase(haystack, signal)
        }
    )[:6]

    # --- title & confidence ----------------------------------------------
    result.title = _title_from_text(raw_body) or _title_from_filename(filename)
    coverage = min(1.0, len(body) / 4000)
    signal_strength = min(1.0, (ranked[0][1] if ranked else 0) / 12)
    result.confidence = round(0.35 + 0.35 * coverage + 0.25 * signal_strength, 2)
    return result


def _guess_document_type(haystack: str, filename: str) -> str:
    best_type, best_score = "Other", 0
    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
        score = sum(
            _count_phrase(haystack, keyword) * (2 if len(keyword) > 14 else 1)
            for keyword in keywords
        )
        # An explicit self-description ("safety circular", "invoice") outranks
        # incidental vocabulary from a neighbouring category.
        if _has_phrase(haystack, doc_type.lower()):
            score += 3
        if score > best_score:
            best_type, best_score = doc_type, score
    if best_score == 0:
        lowered = filename.lower()
        if lowered.endswith((".png", ".jpg", ".jpeg", ".tiff")):
            return "Other"
    return best_type


def _title_from_text(text: str) -> str:
    lines = [line.strip(" \t-—_*#") for line in text.splitlines()[:30]]

    # A "Subject:" line is the closest thing KMRL documents have to a title.
    for candidate in lines:
        lowered = candidate.lower()
        for prefix in ("subject:", "sub:", "subject -"):
            if lowered.startswith(prefix):
                subject = candidate[len(prefix):].strip(" -–—")
                if len(subject) >= 8:
                    return _smart_case(subject)

    for candidate in lines[:8]:
        if 8 <= len(candidate) <= 140 and not candidate.lower().startswith("page "):
            return _smart_case(candidate)
    return ""


def _title_from_filename(filename: str) -> str:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename)
    stem = re.sub(r"[_\-]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip().title()


def _first_date(sentence: str) -> str | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(sentence)
        if match:
            return match.group(0)
    return None


def _owner_for(sentence: str) -> str:
    lowered = sentence.lower()
    for dept in DEPARTMENTS:
        if dept.lower() in lowered:
            return dept
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        if any(_has_phrase(lowered, keyword) for keyword in keywords[:5]):
            return dept
    return ""


def _trim(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ------------------------------------------------------------------ Claude engine
ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "document_type": {"type": "string", "enum": DOCUMENT_TYPES},
        "department": {"type": "string", "enum": DEPARTMENTS + [UNASSIGNED_DEPARTMENT]},
        "secondary_departments": {
            "type": "array",
            "items": {"type": "string", "enum": DEPARTMENTS},
        },
        "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "owner": {"type": "string"},
                    "due_date": {"type": "string"},
                    "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                },
                "required": ["action", "owner", "due_date", "priority"],
                "additionalProperties": False,
            },
        },
        "deadlines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["date", "description"],
                "additionalProperties": False,
            },
        },
        "risks": {"type": "array", "items": {"type": "string"}},
        "compliance_flags": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": [
        "title",
        "document_type",
        "department",
        "secondary_departments",
        "priority",
        "summary",
        "key_points",
        "keywords",
        "actions",
        "deadlines",
        "risks",
        "compliance_flags",
        "confidence",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the document analyst for Kochi Metro Rail Limited (KMRL).

You classify and summarise operational documents — engineering drawings, maintenance
job cards, incident reports, invoices, purchase orders, regulatory directives,
safety circulars, environmental studies, HR policies, legal opinions and board
minutes — which arrive in English, Malayalam or a bilingual mix.

Rules:
- Use ONLY the supplied document text. Never invent facts, dates, amounts,
  people, departments or obligations that are not in the text.
- If a field cannot be determined from the text, return an empty string or an
  empty list rather than guessing.
- Copy dates exactly as they appear in the document. Do not normalise or infer them.
- `department` is the single team that owns the follow-up; `secondary_departments`
  are other teams that must be aware of it.
- `priority` is High when there is a statutory, safety or time-critical
  obligation; Medium when action is required without immediate urgency; Low when
  the document is informational.
- `summary` is 3-5 sentences written for a busy KMRL manager: what the document
  is, what it asks for, and who is affected.
- `confidence` is your own 0-1 confidence in this classification.
"""


def analyse_with_claude(text: str, filename: str, page_count: int | None) -> AnalysisResult:
    """Analyse a document with Claude. Raises on any API/parsing failure."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    excerpt = text[: settings.kmrl_analysis_max_chars]
    truncated = len(text) > len(excerpt)

    user_content = (
        f"Filename: {filename}\n"
        f"Page count: {page_count if page_count is not None else 'unknown'}\n"
        f"{'NOTE: the text below is truncated to the first part of the document.' if truncated else ''}\n\n"
        "--- BEGIN DOCUMENT TEXT ---\n"
        f"{excerpt}\n"
        "--- END DOCUMENT TEXT ---\n\n"
        "Analyse this document for KMRL and return the structured classification."
    )

    response = client.messages.create(
        model=settings.kmrl_analysis_model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_config={"format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}},
    )

    payload = next((block.text for block in response.content if block.type == "text"), "")
    data = json.loads(payload)

    return AnalysisResult(
        title=_trim(str(data.get("title") or ""), 300),
        document_type=normalise_document_type(data.get("document_type")),
        department=normalise_department(data.get("department")),
        secondary_departments=[
            normalise_department(dept)
            for dept in (data.get("secondary_departments") or [])
            if normalise_department(dept) != UNASSIGNED_DEPARTMENT
        ][:4],
        priority=normalise_priority(data.get("priority")),
        summary=str(data.get("summary") or "").strip(),
        key_points=[str(point).strip() for point in (data.get("key_points") or [])][:8],
        keywords=[str(word).strip().lower() for word in (data.get("keywords") or [])][:15],
        actions=[
            {
                "action": str(item.get("action", "")).strip(),
                "owner": str(item.get("owner", "")).strip(),
                "due_date": str(item.get("due_date", "")).strip(),
                "priority": normalise_priority(item.get("priority")),
            }
            for item in (data.get("actions") or [])
            if str(item.get("action", "")).strip()
        ][:12],
        deadlines=[
            {
                "date": str(item.get("date", "")).strip(),
                "description": str(item.get("description", "")).strip(),
            }
            for item in (data.get("deadlines") or [])
            if str(item.get("date", "")).strip()
        ][:12],
        risks=[str(risk).strip() for risk in (data.get("risks") or []) if str(risk).strip()][:8],
        compliance_flags=[
            str(flag).strip() for flag in (data.get("compliance_flags") or []) if str(flag).strip()
        ][:8],
        engine="claude",
        model=settings.kmrl_analysis_model,
        confidence=_as_confidence(data.get("confidence")),
    )


def _as_confidence(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(min(max(number, 0.0), 1.0), 2)


def analyse_document(text: str, filename: str, page_count: int | None = None) -> AnalysisResult:
    """Analyse with Claude when configured, otherwise with the rule engine."""
    if settings.ai_enabled and (text or "").strip():
        try:
            return analyse_with_claude(text, filename, page_count)
        except Exception as exc:  # noqa: BLE001 - fall back, never fail the upload
            logger.warning("Claude analysis failed for %s: %s", filename, exc)
            fallback = analyse_with_rules(text, filename)
            fallback.error = f"AI analysis unavailable ({type(exc).__name__}); used rule-based analysis."
            return fallback
    return analyse_with_rules(text, filename)
