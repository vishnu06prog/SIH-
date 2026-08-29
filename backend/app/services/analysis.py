"""Document understanding: classification, summarisation and extraction.

Two interchangeable engines produce the same ``DocumentAnalysis``:

``claude``      an LLM with a strict JSON schema (used when an API key is set).
``rule-based``  a deterministic KMRL-tuned analyser used as an automatic
                fallback, so the platform is fully functional with no key.

Both run on the *normalised English* text, and both attribute every extracted
statement back to the page it came from, so the UI can show the original —
Malayalam included — beside the AI's conclusion.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any

from ..config import settings
from ..taxonomy import (
    ACTION_SIGNALS,
    COMPLIANCE_SIGNALS,
    CRITICAL_PRIORITY_SIGNALS,
    DEPARTMENT_KEYWORDS,
    DEPARTMENTS,
    DOCUMENT_TYPE_KEYWORDS,
    DOCUMENT_TYPES,
    ENTITY_TYPES,
    HIGH_PRIORITY_SIGNALS,
    MEDIUM_PRIORITY_SIGNALS,
    PRIORITIES,
    RISK_CATEGORIES,
    RISK_SIGNALS,
    STOPWORDS,
    UNASSIGNED_DEPARTMENT,
    normalise_department,
    normalise_document_type,
    normalise_priority,
)
from .dates import classify_deadline, find_dates

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ payloads
@dataclass
class ExtractedAction:
    action: str = ""
    responsible_department: str = ""
    responsible_person: str = ""
    due_date_text: str = ""
    due_date: date | None = None
    priority: str = "Medium"
    source_page: int | None = None
    source_excerpt: str = ""
    source_excerpt_en: str = ""
    source_language: str = "English"


@dataclass
class ExtractedDeadline:
    title: str = ""
    description: str = ""
    due_date_text: str = ""
    due_date: date | None = None
    deadline_type: str = "General"
    department: str = ""
    priority: str = "Medium"
    is_compliance: bool = False
    source_page: int | None = None
    source_excerpt: str = ""
    source_language: str = "English"


@dataclass
class ExtractedRisk:
    description: str = ""
    severity: str = "Medium"
    category: str = "Operational"
    mitigation: str = ""
    source_page: int | None = None
    source_excerpt: str = ""
    source_language: str = "English"


@dataclass
class ExtractedEntity:
    entity_type: str = "Other"
    value: str = ""
    normalised_value: str = ""
    page_number: int | None = None
    source_language: str = "English"
    confidence: float | None = None


@dataclass
class ExtractedStakeholder:
    name: str = ""
    designation: str = ""
    organisation: str = ""
    department: str = ""
    kind: str = "Internal"
    reason: str = ""


@dataclass
class RoutedDepartment:
    department: str = ""
    confidence: float = 0.0
    reason: str = ""
    urgency: str = "Medium"
    is_primary: bool = False


@dataclass
class DocumentAnalysis:
    title: str = ""
    document_type: str = "Other"
    type_confidence: float | None = None
    type_reasoning: str = ""
    department: str = UNASSIGNED_DEPARTMENT
    secondary_departments: list[str] = field(default_factory=list)
    priority: str = "Medium"
    executive_summary: str = ""
    detailed_summary: str = ""
    key_points: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    ai_explanation: str = ""
    keywords: list[str] = field(default_factory=list)
    actions: list[ExtractedAction] = field(default_factory=list)
    deadlines: list[ExtractedDeadline] = field(default_factory=list)
    risks: list[ExtractedRisk] = field(default_factory=list)
    entities: list[ExtractedEntity] = field(default_factory=list)
    stakeholders: list[ExtractedStakeholder] = field(default_factory=list)
    routing: list[RoutedDepartment] = field(default_factory=list)
    compliance_flags: list[str] = field(default_factory=list)
    engine: str = "rule-based"
    model: str = ""
    confidence: float | None = None
    error: str = ""


@dataclass
class PageContext:
    """Page text used to attribute a statement back to its page."""

    page_number: int
    text: str
    original_text: str = ""
    language: str = "English"


# ---------------------------------------------------------------- text tools
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_RE = re.compile(r"[A-Za-zഀ-ൿ][A-Za-z0-9ഀ-ൿ\-]{2,}")

MONEY_RE = re.compile(
    r"(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d{1,2})?(?:\s?(?:lakh|lakhs|crore|crores|million))?",
    re.IGNORECASE,
)
REFERENCE_RE = re.compile(
    r"\b(?:[A-Z]{2,6}[/-][A-Z0-9]{1,8}(?:[/-][A-Z0-9]{1,8}){1,4}"
    r"|(?:Ref|No|PO|NIT|Invoice|Drawing|File|Circular)\.?\s?(?:No\.?)?\s?[:#]?\s?[A-Z0-9][A-Z0-9/-]{3,})\b"
)
PERSON_RE = re.compile(
    r"\b(?:Shri|Smt|Dr|Mr|Ms|Er)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)
DESIGNATION_RE = re.compile(
    r"\b(?:Chief|Deputy|Assistant|Senior|Joint|Additional)?\s?"
    r"(?:Managing Director|Director|General Manager|Manager|Engineer|Officer|"
    r"Commissioner|Secretary|Head|Superintendent|Controller|Supervisor|Inspector)\b",
    re.IGNORECASE,
)
ORG_RE = re.compile(
    r"\b(?:Kochi Metro Rail Limited|KMRL|CMRS|Commissioner of Metro Rail Safety|"
    r"Ministry of Housing and Urban Affairs|MoHUA|Kerala State Pollution Control Board|"
    r"Indian Railways|DMRC|RDSO|Alstom|Siemens|Bombardier|BEML)\b",
    re.IGNORECASE,
)
LOCATION_RE = re.compile(
    r"\b(?:Aluva|Pulinchodu|Companypady|Ambattukavu|Muttom|Kalamassery|"
    r"Cusat|Pathadipalam|Edapally|Changampuzha|Palarivattom|JLN Stadium|Kaloor|"
    r"Town Hall|MG Road|Maharaja's College|Ernakulam South|Kadavanthra|Elamkulam|"
    r"Vyttila|Thaikoodam|Petta|SN Junction|Tripunithura|Kochi|Ernakulam|Kerala)\b"
)
ASSET_RE = re.compile(
    r"\b(?:trainset|train set|TS-?\d+|escalator|lift|platform screen door|PSD|"
    r"traction motor|bogie|brake pad|OHE|substation|axle counter|point machine|"
    r"UPS|transformer|HVAC)\b",
    re.IGNORECASE,
)
REGULATION_RE = re.compile(
    r"\b(?:Metro Railways \(Operation and Maintenance\) Act[, ]?\d{0,4}|"
    r"Metro Railways Act|Factories Act|Environment \(Protection\) Act|"
    r"CGST Act|Contract Labour Act|Section \d+[A-Z]?|Rule \d+|Clause \d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)


def _uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _reflow(text: str) -> str:
    """Undo hard line wrapping so sentence splitting sees whole sentences."""
    paragraphs: list[str] = []
    buffer = ""
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line:
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            continue
        if (
            buffer
            and not buffer.endswith((".", "!", "?", ":", ";", "|"))
            and len(buffer) >= 45
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


ACRONYMS = {
    "KMRL", "CMRS", "MOHUA", "OHE", "HR", "IT", "CBTC", "ATP", "ATO", "SCADA",
    "NIT", "EMD", "GST", "DPR", "IOT", "OCC", "PSD", "AC", "DC", "UPS", "RFP",
    "EOI", "PO", "TS", "JC", "INV", "ERP", "API", "SOP", "PPE", "EIA", "KSPCB",
}
TITLE_MINOR_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or",
    "the", "to", "with",
}


def _capitalise_token(word: str) -> str:
    for index, character in enumerate(word):
        if character.isalpha():
            if word[:index].strip("([{\"'“‘").isdigit():
                return word.lower()  # 112TH -> 112th
            return word[:index] + character.upper() + word[index + 1 :].lower()
    return word


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
            words.append(word)
        elif index > 0 and word.lower() in TITLE_MINOR_WORDS:
            words.append(word.lower())
        else:
            words.append(_capitalise_token(word))
    return " ".join(words)


def _sentences(text: str, minimum: int = 25) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text or "") if s and s.strip()]
    return [s for s in parts if len(s) > minimum]


def _tokens(text: str) -> list[str]:
    return [
        word.lower()
        for word in WORD_RE.findall(text or "")
        if word.lower() not in STOPWORDS and not word.isdigit()
    ]


@lru_cache(maxsize=4096)
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Whole-word matcher — so "monitoring" never counts as a hit for "nit"."""
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", re.IGNORECASE)


def _count_phrase(haystack: str, phrase: str) -> int:
    return len(_phrase_pattern(phrase).findall(haystack))


def _has_phrase(haystack: str, phrase: str) -> bool:
    return _phrase_pattern(phrase).search(haystack) is not None


def _count_signals(haystack: str, signals: list[str]) -> int:
    return sum(_count_phrase(haystack, signal) for signal in signals)


def _trim(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# --------------------------------------------------------- page attribution
class PageIndex:
    """Maps an extracted sentence back to the page it came from."""

    def __init__(self, pages: list[PageContext]):
        self.pages = pages
        self._normalised = [
            (page, re.sub(r"\s+", " ", (page.text or "")).lower()) for page in pages
        ]

    def page_for(self, excerpt: str) -> PageContext | None:
        needle = re.sub(r"\s+", " ", (excerpt or "")).lower().strip()
        if not needle or not self._normalised:
            return None
        probe = needle[:120]
        for page, haystack in self._normalised:
            if probe and probe in haystack:
                return page
        # Fall back to the page sharing the most distinctive words.
        words = {word for word in _tokens(needle) if len(word) > 4}
        if not words:
            return None
        best, best_score = None, 0
        for page, haystack in self._normalised:
            score = sum(1 for word in words if word in haystack)
            if score > best_score:
                best, best_score = page, score
        return best if best_score >= max(2, len(words) // 3) else None

    def page_number_for(self, excerpt: str) -> int | None:
        page = self.page_for(excerpt)
        return page.page_number if page else None

    def original_for(self, excerpt: str) -> tuple[str, str]:
        """(original-language excerpt, its language) for a translated sentence."""
        page = self.page_for(excerpt)
        if page is None or not page.original_text or page.original_text == page.text:
            return "", "English"
        return _trim(page.original_text, 400), page.language or "English"


# ----------------------------------------------------------- rule-based engine
def _guess_document_type(haystack: str, filename: str) -> tuple[str, float, str]:
    scores: dict[str, int] = {}
    hits: dict[str, list[str]] = {}
    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
        score = 0
        matched: list[str] = []
        for keyword in keywords:
            count = _count_phrase(haystack, keyword)
            if count:
                matched.append(keyword)
                score += count * (2 if len(keyword) > 14 else 1)
        # An explicit self-description ("safety circular") outranks incidental
        # vocabulary borrowed from a neighbouring category.
        if _has_phrase(haystack, doc_type.lower()):
            score += 4
            matched.append(doc_type.lower())
        if score:
            scores[doc_type] = score
            hits[doc_type] = matched[:4]

    if not scores:
        return "Other", 0.25, "No distinctive document-type vocabulary was found."

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    margin = (best_score - runner_up) / best_score if best_score else 0
    confidence = round(min(0.95, 0.45 + 0.3 * min(1.0, best_score / 10) + 0.25 * margin), 2)
    reasoning = (
        f"Classified as {best} because the text contains "
        + ", ".join(f"'{term}'" for term in hits.get(best, [])[:3])
        + (
            f"; the next best match was {ranked[1][0]}."
            if len(ranked) > 1
            else "; no competing category scored."
        )
    )
    return best, confidence, reasoning


def _score_departments(haystack: str) -> list[tuple[str, int, list[str]]]:
    ranked: list[tuple[str, int, list[str]]] = []
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        score = 0
        matched: list[str] = []
        for keyword in keywords:
            count = _count_phrase(haystack, keyword)
            if count:
                matched.append(keyword)
                score += count * (2 if len(keyword) > 12 else 1)
        if score:
            ranked.append((dept, score, matched[:4]))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _priority_for(haystack: str, department: str, dated: bool) -> str:
    critical = _count_signals(haystack, CRITICAL_PRIORITY_SIGNALS)
    high = _count_signals(haystack, HIGH_PRIORITY_SIGNALS)
    medium = _count_signals(haystack, MEDIUM_PRIORITY_SIGNALS)

    if critical >= 2 or (critical >= 1 and department == "Safety") or (critical >= 1 and high >= 3):
        return "Critical"
    if critical >= 1 or high >= 3 or (high >= 1 and department == "Safety" and dated):
        return "High"
    if high >= 1 or medium >= 2 or dated:
        return "Medium"
    return "Low"


def _owner_for(sentence: str, default: str = "") -> str:
    lowered = sentence.lower()
    for dept in DEPARTMENTS:
        if _has_phrase(lowered, dept.lower()):
            return dept
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        if any(_has_phrase(lowered, keyword) for keyword in keywords[:6]):
            return dept
    return default


def _risk_severity(sentence: str) -> str:
    lowered = sentence.lower()
    if any(_has_phrase(lowered, signal) for signal in CRITICAL_PRIORITY_SIGNALS):
        return "Critical"
    if any(
        _has_phrase(lowered, signal)
        for signal in ("penalty", "non-compliance", "litigation", "unsafe", "shutdown", "hazard")
    ):
        return "High"
    if any(_has_phrase(lowered, signal) for signal in ("delay", "defect", "shortfall", "overrun")):
        return "Medium"
    return "Low"


RISK_CATEGORY_SIGNALS: dict[str, tuple[str, ...]] = {
    "Safety": ("safety", "hazard", "accident", "injury", "unsafe", "fire", "evacuation"),
    "Regulatory": ("cmrs", "statutory", "regulation", "non-compliance", "directive", "gazette"),
    "Financial": ("penalty", "cost", "budget", "payment", "invoice", "overrun", "liquidated"),
    "Legal": ("litigation", "arbitration", "breach of contract", "court", "liability"),
    "Environmental": ("pollution", "emission", "effluent", "environmental", "waste"),
    "Technical": ("defect", "failure", "malfunction", "fault", "wear", "breakdown"),
    "Reputational": ("reputation", "public", "media", "passenger complaint"),
}


def _risk_category(sentence: str) -> str:
    lowered = sentence.lower()
    for category, signals in RISK_CATEGORY_SIGNALS.items():
        if any(signal in lowered for signal in signals):
            return category
    return "Operational"


def _extract_entities(text: str, index: PageIndex) -> list[ExtractedEntity]:
    found: list[ExtractedEntity] = []
    seen: set[tuple[str, str]] = set()

    def add(entity_type: str, value: str, confidence: float) -> None:
        cleaned = _trim(value, 200)
        key = (entity_type, cleaned.lower())
        if not cleaned or key in seen:
            return
        seen.add(key)
        found.append(
            ExtractedEntity(
                entity_type=entity_type,
                value=cleaned,
                normalised_value=cleaned.lower(),
                page_number=index.page_number_for(cleaned),
                confidence=confidence,
            )
        )

    for match in ORG_RE.finditer(text):
        add("Organisation", match.group(0), 0.9)
    for match in PERSON_RE.finditer(text):
        add("Person", match.group(0), 0.75)
    for match in LOCATION_RE.finditer(text):
        add("Location", match.group(0), 0.8)
    for match in MONEY_RE.finditer(text):
        add("Money", match.group(0), 0.85)
    for match in REFERENCE_RE.finditer(text):
        add("Reference", match.group(0), 0.7)
    for match in ASSET_RE.finditer(text):
        add("Asset", match.group(0), 0.75)
    for match in REGULATION_RE.finditer(text):
        add("Regulation", match.group(0), 0.8)
    for match in find_dates(text)[:12]:
        add("Date", match.text, 0.85)

    return found[:60]


def _extract_stakeholders(
    text: str, entities: list[ExtractedEntity], department: str
) -> list[ExtractedStakeholder]:
    stakeholders: list[ExtractedStakeholder] = []
    seen: set[str] = set()

    for entity in entities:
        if entity.entity_type == "Person" and entity.value.lower() not in seen:
            seen.add(entity.value.lower())
            window = _window_around(text, entity.value)
            designation = DESIGNATION_RE.search(window)
            stakeholders.append(
                ExtractedStakeholder(
                    name=entity.value,
                    designation=_trim(designation.group(0), 120) if designation else "",
                    department=department if department != UNASSIGNED_DEPARTMENT else "",
                    kind="Internal",
                    reason="Named in the document.",
                )
            )

    for entity in entities:
        if entity.entity_type == "Organisation" and entity.value.lower() not in seen:
            seen.add(entity.value.lower())
            external = "kmrl" not in entity.value.lower()
            stakeholders.append(
                ExtractedStakeholder(
                    name=entity.value,
                    organisation=entity.value,
                    kind="External" if external else "Internal",
                    reason=(
                        "External body referenced in the document."
                        if external
                        else "The issuing/receiving organisation."
                    ),
                )
            )

    return stakeholders[:15]


def _window_around(text: str, needle: str, radius: int = 120) -> str:
    position = text.find(needle)
    if position < 0:
        return ""
    return text[max(0, position - radius) : position + len(needle) + radius]


def analyse_with_rules(
    text: str,
    filename: str,
    pages: list[PageContext] | None = None,
    reference_date: date | None = None,
) -> DocumentAnalysis:
    """The deterministic KMRL analyser. Never raises; always returns a result."""
    result = DocumentAnalysis(engine="rule-based", model="kmrl-heuristics-v2")
    raw_body = (text or "").strip()
    body = _reflow(raw_body)
    index = PageIndex(pages or [PageContext(page_number=1, text=raw_body)])
    haystack = f"{filename}\n{body}".lower()

    if not body:
        result.title = _title_from_filename(filename)
        result.executive_summary = (
            "No machine-readable text could be extracted from this file, so the "
            "analysis is based on the filename alone. Open the original document "
            "or re-run OCR to get a real summary."
        )
        result.detailed_summary = result.executive_summary
        result.confidence = 0.15
        result.document_type, result.type_confidence, result.type_reasoning = (
            "Other",
            0.1,
            "No readable text was available to classify.",
        )
        return result

    # --- departments -------------------------------------------------------
    ranked = _score_departments(haystack)
    if ranked:
        result.department = ranked[0][0]
        top_score = ranked[0][1]
        result.secondary_departments = [
            dept for dept, score, _ in ranked[1:4] if score >= max(2, top_score * 0.35)
        ]
    else:
        result.department = UNASSIGNED_DEPARTMENT
        top_score = 0

    # --- classification ----------------------------------------------------
    result.document_type, result.type_confidence, result.type_reasoning = _guess_document_type(
        haystack, filename
    )

    # --- priority ----------------------------------------------------------
    date_matches = find_dates(body, reference_date)
    result.priority = _priority_for(haystack, result.department, bool(date_matches))

    # --- keywords ----------------------------------------------------------
    counts = Counter(_tokens(body))
    result.keywords = [word for word, _ in counts.most_common(15)]

    # --- summaries ---------------------------------------------------------
    sentences = _sentences(body)
    candidates = [s for s in sentences if _uppercase_ratio(s) <= 0.6] or sentences
    top_terms = {word for word, _ in counts.most_common(40)}
    scored: list[tuple[float, int, str]] = []
    for position, sentence in enumerate(candidates[:400]):
        lowered = sentence.lower()
        overlap = sum(1 for word in set(_tokens(sentence)) if word in top_terms)
        signal_bonus = 2 if any(_has_phrase(lowered, sig) for sig in ACTION_SIGNALS) else 0
        position_bonus = 1.5 if position < 5 else 0
        length_penalty = 0.5 if len(sentence) > 320 else 0
        scored.append(
            (overlap + signal_bonus + position_bonus - length_penalty, position, sentence)
        )
    scored.sort(key=lambda item: item[0], reverse=True)

    detailed = [item[2] for item in sorted(scored[:7], key=lambda item: item[1])]
    executive = [item[2] for item in sorted(scored[:3], key=lambda item: item[1])]
    result.detailed_summary = _trim(" ".join(detailed), 1600) or body[:600]
    result.executive_summary = _trim(" ".join(executive), 600) or result.detailed_summary[:400]
    result.key_points = [_trim(sentence, 220) for sentence in detailed[:5]]

    # --- actions -----------------------------------------------------------
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(_has_phrase(lowered, signal) for signal in ACTION_SIGNALS):
            continue
        matches = find_dates(sentence, reference_date)
        due = matches[0] if matches else None
        original, language = index.original_for(sentence)
        result.actions.append(
            ExtractedAction(
                action=_trim(sentence, 400),
                responsible_department=_owner_for(
                    sentence,
                    result.department if result.department != UNASSIGNED_DEPARTMENT else "",
                ),
                due_date_text=due.text if due else "",
                due_date=due.value if due else None,
                priority=(
                    "Critical"
                    if any(_has_phrase(lowered, sig) for sig in CRITICAL_PRIORITY_SIGNALS)
                    else "High"
                    if any(_has_phrase(lowered, sig) for sig in HIGH_PRIORITY_SIGNALS)
                    else result.priority
                ),
                source_page=index.page_number_for(sentence),
                source_excerpt=original or _trim(sentence, 400),
                source_excerpt_en=_trim(sentence, 400),
                source_language=language,
            )
        )
        if len(result.actions) >= 12:
            break

    # --- deadlines ---------------------------------------------------------
    result.deadlines = _deadlines_from(sentences, index, result, reference_date)

    # --- risks -------------------------------------------------------------
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(_has_phrase(lowered, signal) for signal in RISK_SIGNALS):
            continue
        original, language = index.original_for(sentence)
        result.risks.append(
            ExtractedRisk(
                description=_trim(sentence, 400),
                severity=_risk_severity(sentence),
                category=_risk_category(sentence),
                source_page=index.page_number_for(sentence),
                source_excerpt=original or _trim(sentence, 300),
                source_language=language,
            )
        )
        if len(result.risks) >= 8:
            break

    # --- entities & stakeholders ------------------------------------------
    result.entities = _extract_entities(body, index)
    result.stakeholders = _extract_stakeholders(body, result.entities, result.department)

    # --- compliance & next steps ------------------------------------------
    result.compliance_flags = sorted(
        {
            signal.upper() if len(signal) <= 5 else signal.title()
            for signal in COMPLIANCE_SIGNALS
            if _has_phrase(haystack, signal)
        }
    )[:8]
    result.next_steps = _next_steps(result)

    # --- routing -----------------------------------------------------------
    result.routing = _routing_from_scores(ranked, result)

    # --- title, explanation, confidence -----------------------------------
    result.title = _title_from_text(raw_body) or _title_from_filename(filename)
    result.ai_explanation = (
        f"{result.type_reasoning} "
        f"Routed to {result.department} because the text repeatedly uses that team's "
        f"vocabulary ({', '.join(ranked[0][2][:3]) if ranked else 'no strong signal'}). "
        f"Priority is {result.priority} based on "
        f"{_count_signals(haystack, CRITICAL_PRIORITY_SIGNALS)} critical and "
        f"{_count_signals(haystack, HIGH_PRIORITY_SIGNALS)} high-urgency phrases"
        f"{' and a dated obligation' if date_matches else ''}. "
        f"Extracted {len(result.actions)} action(s), {len(result.deadlines)} deadline(s) "
        f"and {len(result.risks)} risk(s), each linked to its source page."
    )
    coverage = min(1.0, len(body) / 4000)
    signal_strength = min(1.0, top_score / 12)
    result.confidence = round(0.35 + 0.35 * coverage + 0.25 * signal_strength, 2)
    return result


def _deadlines_from(
    sentences: list[str],
    index: PageIndex,
    result: DocumentAnalysis,
    reference_date: date | None,
) -> list[ExtractedDeadline]:
    deadlines: list[ExtractedDeadline] = []
    seen: set[str] = set()
    for sentence in sentences:
        for match in find_dates(sentence, reference_date):
            key = f"{match.text.lower()}|{sentence[:40].lower()}"
            if key in seen:
                continue
            seen.add(key)
            original, language = index.original_for(sentence)
            deadline_type = classify_deadline(sentence)
            is_compliance = deadline_type in {"Compliance", "Inspection", "Renewal"} or any(
                _has_phrase(sentence.lower(), signal) for signal in COMPLIANCE_SIGNALS
            )
            deadlines.append(
                ExtractedDeadline(
                    title=_trim(sentence, 200),
                    description=_trim(sentence, 400),
                    due_date_text=match.text,
                    due_date=match.value,
                    deadline_type=deadline_type,
                    department=result.department
                    if result.department != UNASSIGNED_DEPARTMENT
                    else "",
                    priority=result.priority,
                    is_compliance=is_compliance,
                    source_page=index.page_number_for(sentence),
                    source_excerpt=original or _trim(sentence, 300),
                    source_language=language,
                )
            )
            break  # one deadline per sentence keeps the list readable
        if len(deadlines) >= 10:
            break
    return deadlines


def _next_steps(result: DocumentAnalysis) -> list[str]:
    """Concrete follow-ups implied by what was extracted."""
    steps: list[str] = []
    if result.actions:
        owners = sorted(
            {action.responsible_department for action in result.actions if action.responsible_department}
        )
        target = ", ".join(owners) if owners else result.department
        steps.append(f"Assign the {len(result.actions)} extracted action(s) to {target}.")
    dated = [d for d in result.deadlines if d.due_date]
    if dated:
        earliest = min(dated, key=lambda item: item.due_date)  # type: ignore[arg-type]
        steps.append(
            f"Track the earliest deadline ({earliest.due_date_text}) on the compliance board."
        )
    severe = [r for r in result.risks if r.severity in {"Critical", "High"}]
    if severe:
        steps.append(
            f"Escalate {len(severe)} high-severity risk(s) to the {result.department} head for mitigation."
        )
    if result.priority in {"Critical", "High"}:
        steps.append("Notify the routed department heads and acknowledge receipt.")
    if result.compliance_flags:
        steps.append(
            "Record the compliance reference and file the response with the regulator."
        )
    steps.append("Verify the AI classification and routing, then mark the document reviewed.")
    return steps[:6]


def _routing_from_scores(
    ranked: list[tuple[str, int, list[str]]], result: DocumentAnalysis
) -> list[RoutedDepartment]:
    if not ranked:
        return []
    top = ranked[0][1]
    routed: list[RoutedDepartment] = []
    for position, (dept, score, matched) in enumerate(ranked[:5]):
        if position > 0 and score < max(2, top * 0.3):
            break
        confidence = round(min(0.97, 0.4 + 0.55 * (score / top)), 2)
        routed.append(
            RoutedDepartment(
                department=dept,
                confidence=confidence,
                reason=(
                    f"The document uses {dept} vocabulary: "
                    + ", ".join(f"'{term}'" for term in matched[:3])
                    + "."
                ),
                urgency=result.priority,
                is_primary=position == 0,
            )
        )
    return routed


def _title_from_text(text: str) -> str:
    lines = [line.strip(" \t-—_*#") for line in (text or "").splitlines()[:30]]
    for candidate in lines:
        lowered = candidate.lower()
        for prefix in ("subject:", "sub:", "subject -", "re:"):
            if lowered.startswith(prefix):
                subject = candidate[len(prefix) :].strip(" -–—")
                if len(subject) >= 8:
                    return _smart_case(_trim(subject, 300))
    for candidate in lines[:8]:
        if 8 <= len(candidate) <= 160 and not candidate.lower().startswith("page "):
            return _smart_case(_trim(candidate, 300))
    return ""


def _title_from_filename(filename: str) -> str:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename or "document")
    stem = re.sub(r"[_\-]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip().title()


# ------------------------------------------------------------------ LLM engine
ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "document_type": {"type": "string", "enum": DOCUMENT_TYPES},
        "type_confidence": {"type": "number"},
        "type_reasoning": {"type": "string"},
        "department": {"type": "string", "enum": DEPARTMENTS + [UNASSIGNED_DEPARTMENT]},
        "secondary_departments": {"type": "array", "items": {"type": "string", "enum": DEPARTMENTS}},
        "priority": {"type": "string", "enum": PRIORITIES},
        "executive_summary": {"type": "string"},
        "detailed_summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "ai_explanation": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "responsible_department": {"type": "string"},
                    "responsible_person": {"type": "string"},
                    "due_date": {"type": "string"},
                    "priority": {"type": "string", "enum": PRIORITIES},
                    "source_page": {"type": "integer"},
                    "source_excerpt": {"type": "string"},
                },
                "required": [
                    "action", "responsible_department", "responsible_person",
                    "due_date", "priority", "source_page", "source_excerpt",
                ],
                "additionalProperties": False,
            },
        },
        "deadlines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "deadline_type": {"type": "string"},
                    "is_compliance": {"type": "boolean"},
                    "source_page": {"type": "integer"},
                    "source_excerpt": {"type": "string"},
                },
                "required": [
                    "title", "date", "deadline_type", "is_compliance",
                    "source_page", "source_excerpt",
                ],
                "additionalProperties": False,
            },
        },
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "severity": {"type": "string", "enum": ["Critical", "High", "Medium", "Low"]},
                    "category": {"type": "string", "enum": RISK_CATEGORIES},
                    "mitigation": {"type": "string"},
                    "source_page": {"type": "integer"},
                },
                "required": ["description", "severity", "category", "mitigation", "source_page"],
                "additionalProperties": False,
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string", "enum": ENTITY_TYPES},
                    "value": {"type": "string"},
                    "page_number": {"type": "integer"},
                },
                "required": ["entity_type", "value", "page_number"],
                "additionalProperties": False,
            },
        },
        "stakeholders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "designation": {"type": "string"},
                    "organisation": {"type": "string"},
                    "kind": {"type": "string", "enum": ["Internal", "External"]},
                    "reason": {"type": "string"},
                },
                "required": ["name", "designation", "organisation", "kind", "reason"],
                "additionalProperties": False,
            },
        },
        "routing": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "department": {"type": "string", "enum": DEPARTMENTS},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                    "urgency": {"type": "string", "enum": PRIORITIES},
                },
                "required": ["department", "confidence", "reason", "urgency"],
                "additionalProperties": False,
            },
        },
        "compliance_flags": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": [
        "title", "document_type", "type_confidence", "type_reasoning", "department",
        "secondary_departments", "priority", "executive_summary", "detailed_summary",
        "key_points", "next_steps", "ai_explanation", "keywords", "actions",
        "deadlines", "risks", "entities", "stakeholders", "routing",
        "compliance_flags", "confidence",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the document analyst for Kochi Metro Rail Limited (KMRL).

You classify, summarise and extract obligations from operational documents —
engineering drawings, maintenance job cards, incident reports, invoices, purchase
orders, regulatory directives, safety circulars, environmental studies, HR
policies, legal opinions and board minutes. Documents arrive in English,
Malayalam, or a bilingual mix; Malayalam is given to you already translated into
English, with page numbers preserved.

Rules:
- Use ONLY the supplied document text. Never invent facts, dates, amounts,
  people, departments or obligations that are not in the text.
- If a field cannot be determined, return an empty string or an empty list.
  An empty answer is always better than a guess.
- Copy dates exactly as they appear. Do not normalise or infer them.
- Every action, deadline and risk MUST carry `source_page` (the page number the
  text came from) and, where the schema asks for it, the verbatim
  `source_excerpt` it was derived from. This traceability is mandatory.
- `department` is the single team that owns the follow-up. `secondary_departments`
  are teams that must be aware. `routing` explains, per department, why it should
  receive the document and how urgent it is for them.
- `priority`: Critical for safety-critical or statutory obligations with an
  immediate deadline; High for time-bound mandatory action; Medium when action is
  required without urgency; Low for informational documents.
- `executive_summary` is 2-3 sentences for a busy KMRL executive.
  `detailed_summary` is 5-8 sentences covering what the document is, what it
  requires, who is affected and by when.
- `ai_explanation` states, in two or three sentences, WHY you classified, prioritised
  and routed the document as you did, naming the evidence you used.
- `next_steps` are concrete follow-ups KMRL staff should take.
"""


def analyse_with_llm(
    text: str,
    filename: str,
    pages: list[PageContext] | None = None,
    page_count: int | None = None,
    reference_date: date | None = None,
) -> DocumentAnalysis:
    """Analyse with Claude. Raises on any API or parsing failure."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    excerpt = _paginated_excerpt(text, pages, settings.kmrl_analysis_max_chars)
    truncated = len(text) > settings.kmrl_analysis_max_chars

    user_content = (
        f"Filename: {filename}\n"
        f"Page count: {page_count if page_count is not None else 'unknown'}\n"
        f"Today's date: {(reference_date or date.today()).isoformat()}\n"
        + ("NOTE: the text below is truncated to the first part of the document.\n" if truncated else "")
        + "\n--- BEGIN DOCUMENT TEXT ---\n"
        f"{excerpt}\n"
        "--- END DOCUMENT TEXT ---\n\n"
        "Analyse this document for KMRL and return the structured result."
    )

    response = client.messages.create(
        model=settings.kmrl_analysis_model,
        max_tokens=12000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_config={"format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}},
    )
    payload = next(
        (block.text for block in response.content if getattr(block, "type", "") == "text"), ""
    )
    data = json.loads(payload)
    return _analysis_from_payload(data, pages, reference_date)


def _paginated_excerpt(
    text: str, pages: list[PageContext] | None, limit: int
) -> str:
    """Feed the model page-tagged text so it can cite real page numbers."""
    if not pages:
        return text[:limit]
    parts: list[str] = []
    used = 0
    for page in pages:
        block = f"\n[PAGE {page.page_number}]\n{page.text}"
        if used + len(block) > limit:
            parts.append(block[: max(0, limit - used)])
            break
        parts.append(block)
        used += len(block)
    return "".join(parts).strip() or text[:limit]


def _analysis_from_payload(
    data: dict[str, Any],
    pages: list[PageContext] | None,
    reference_date: date | None,
) -> DocumentAnalysis:
    index = PageIndex(pages or [])

    def page_of(value: Any, excerpt: str = "") -> int | None:
        try:
            number = int(value)
            if number > 0:
                return number
        except (TypeError, ValueError):
            pass
        return index.page_number_for(excerpt) if excerpt else None

    actions: list[ExtractedAction] = []
    for item in data.get("actions") or []:
        description = str(item.get("action", "")).strip()
        if not description:
            continue
        excerpt = str(item.get("source_excerpt", "")).strip() or description
        due_text = str(item.get("due_date", "")).strip()
        matches = find_dates(due_text, reference_date) if due_text else []
        original, language = index.original_for(excerpt)
        actions.append(
            ExtractedAction(
                action=_trim(description, 600),
                responsible_department=normalise_department(item.get("responsible_department"))
                if normalise_department(item.get("responsible_department")) != UNASSIGNED_DEPARTMENT
                else "",
                responsible_person=_trim(str(item.get("responsible_person", "")), 200),
                due_date_text=due_text,
                due_date=matches[0].value if matches else None,
                priority=normalise_priority(item.get("priority")),
                source_page=page_of(item.get("source_page"), excerpt),
                source_excerpt=original or _trim(excerpt, 600),
                source_excerpt_en=_trim(excerpt, 600),
                source_language=language,
            )
        )

    deadlines: list[ExtractedDeadline] = []
    for item in data.get("deadlines") or []:
        title = str(item.get("title", "")).strip()
        due_text = str(item.get("date", "")).strip()
        if not title and not due_text:
            continue
        excerpt = str(item.get("source_excerpt", "")).strip() or title
        matches = find_dates(due_text, reference_date) if due_text else []
        original, language = index.original_for(excerpt)
        deadlines.append(
            ExtractedDeadline(
                title=_trim(title or due_text, 300),
                description=_trim(excerpt, 600),
                due_date_text=due_text,
                due_date=matches[0].value if matches else None,
                deadline_type=str(item.get("deadline_type", "General")).strip() or "General",
                priority=normalise_priority(data.get("priority")),
                is_compliance=bool(item.get("is_compliance")),
                source_page=page_of(item.get("source_page"), excerpt),
                source_excerpt=original or _trim(excerpt, 400),
                source_language=language,
            )
        )

    risks: list[ExtractedRisk] = []
    for item in data.get("risks") or []:
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        original, language = index.original_for(description)
        risks.append(
            ExtractedRisk(
                description=_trim(description, 600),
                severity=normalise_priority(item.get("severity")),
                category=str(item.get("category", "Operational")).strip() or "Operational",
                mitigation=_trim(str(item.get("mitigation", "")), 400),
                source_page=page_of(item.get("source_page"), description),
                source_excerpt=original or _trim(description, 400),
                source_language=language,
            )
        )

    entities = [
        ExtractedEntity(
            entity_type=str(item.get("entity_type", "Other")).strip() or "Other",
            value=_trim(str(item.get("value", "")), 300),
            normalised_value=_trim(str(item.get("value", "")), 300).lower(),
            page_number=page_of(item.get("page_number"), str(item.get("value", ""))),
        )
        for item in (data.get("entities") or [])
        if str(item.get("value", "")).strip()
    ][:80]

    stakeholders = [
        ExtractedStakeholder(
            name=_trim(str(item.get("name", "")), 200),
            designation=_trim(str(item.get("designation", "")), 200),
            organisation=_trim(str(item.get("organisation", "")), 200),
            kind="External" if str(item.get("kind", "")).lower() == "external" else "Internal",
            reason=_trim(str(item.get("reason", "")), 400),
        )
        for item in (data.get("stakeholders") or [])
        if str(item.get("name", "")).strip()
    ][:20]

    routing: list[RoutedDepartment] = []
    for position, item in enumerate(data.get("routing") or []):
        department = normalise_department(item.get("department"))
        if department == UNASSIGNED_DEPARTMENT:
            continue
        routing.append(
            RoutedDepartment(
                department=department,
                confidence=_as_confidence(item.get("confidence")) or 0.5,
                reason=_trim(str(item.get("reason", "")), 500),
                urgency=normalise_priority(item.get("urgency")),
                is_primary=position == 0,
            )
        )

    return DocumentAnalysis(
        title=_trim(str(data.get("title") or ""), 300),
        document_type=normalise_document_type(data.get("document_type")),
        type_confidence=_as_confidence(data.get("type_confidence")),
        type_reasoning=_trim(str(data.get("type_reasoning") or ""), 800),
        department=normalise_department(data.get("department")),
        secondary_departments=[
            dept
            for dept in (normalise_department(d) for d in (data.get("secondary_departments") or []))
            if dept != UNASSIGNED_DEPARTMENT
        ][:5],
        priority=normalise_priority(data.get("priority")),
        executive_summary=_trim(str(data.get("executive_summary") or ""), 1200),
        detailed_summary=_trim(str(data.get("detailed_summary") or ""), 4000),
        key_points=[_trim(str(p), 300) for p in (data.get("key_points") or []) if str(p).strip()][:10],
        next_steps=[_trim(str(p), 300) for p in (data.get("next_steps") or []) if str(p).strip()][:8],
        ai_explanation=_trim(str(data.get("ai_explanation") or ""), 1500),
        keywords=[str(k).strip().lower() for k in (data.get("keywords") or []) if str(k).strip()][:20],
        actions=actions[:20],
        deadlines=deadlines[:20],
        risks=risks[:15],
        entities=entities,
        stakeholders=stakeholders,
        routing=routing[:6],
        compliance_flags=[
            _trim(str(f), 120) for f in (data.get("compliance_flags") or []) if str(f).strip()
        ][:10],
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


# -------------------------------------------------------------------- facade
def analyse_document(
    text: str,
    filename: str,
    pages: list[PageContext] | None = None,
    page_count: int | None = None,
    reference_date: date | None = None,
) -> DocumentAnalysis:
    """Analyse with the LLM when configured, otherwise with the rule engine."""
    if settings.llm_enabled and (text or "").strip():
        try:
            result = analyse_with_llm(text, filename, pages, page_count, reference_date)
            if not result.routing:
                # Keep routing populated even if the model omitted it.
                fallback = analyse_with_rules(text, filename, pages, reference_date)
                result.routing = fallback.routing
            return result
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the upload
            logger.warning("LLM analysis failed for %s: %s", filename, exc)
            fallback = analyse_with_rules(text, filename, pages, reference_date)
            fallback.error = (
                f"AI analysis unavailable ({type(exc).__name__}); used the rule-based analyser."
            )
            return fallback
    return analyse_with_rules(text, filename, pages, reference_date)
