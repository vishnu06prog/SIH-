"""Malayalam → English normalisation.

The original text is never replaced. Everything here produces a *parallel*
English rendering that the AI layer, the keyword index and the embeddings run
on, so an English query can retrieve a Malayalam document while the Malayalam
evidence stays on screen next to the answer.

Three interchangeable engines, tried in order:

``anthropic``  Claude, with a KMRL-aware prompt that preserves names, numbers,
               dates, references and technical/regulatory terminology.
``argos``      ``argostranslate`` when the ml→en package is installed — an
               offline neural model, no API key needed.
``glossary``   The built-in KMRL domain glossary plus transliteration. Always
               available, no downloads, deterministic. Marked as low
               confidence so the UI can say the translation is machine-assisted.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from ..config import settings
from ..glossary import (
    CHILLUS,
    CONSONANTS,
    INDEPENDENT_VOWELS,
    MALAYALAM_DIGITS,
    MALAYALAM_PHRASES,
    MALAYALAM_WORDS,
    VIRAMA,
    VOWEL_SIGNS,
    ZWJ,
    ZWNJ,
)
from .extraction import BILINGUAL, ENGLISH, MALAYALAM, has_malayalam

logger = logging.getLogger(__name__)

ENGINE_ANTHROPIC = "anthropic"
ENGINE_ARGOS = "argos"
ENGINE_GLOSSARY = "glossary"
ENGINE_NONE = "none"

STATUS_NOT_REQUIRED = "not_required"
STATUS_TRANSLATED = "translated"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

# Claude is asked for at most this much text per call.
MAX_TRANSLATE_CHARS = 24_000


@dataclass
class TranslationResult:
    text: str = ""
    engine: str = ENGINE_NONE
    model: str = ""
    confidence: float | None = None
    status: str = STATUS_NOT_REQUIRED
    error: str = ""

    @property
    def translated(self) -> bool:
        return self.status in {STATUS_TRANSLATED, STATUS_PARTIAL}


# ---------------------------------------------------------------- tokenising
MALAYALAM_TOKEN = re.compile(r"[ഀ-ൿ‌‍]+")


def needs_translation(text: str, language: str | None = None) -> bool:
    """True when there is Malayalam script worth normalising to English."""
    if not (text or "").strip():
        return False
    if language in {MALAYALAM, BILINGUAL}:
        return True
    return has_malayalam(text)


# ------------------------------------------------------------ transliteration
def transliterate(word: str) -> str:
    """Romanise a Malayalam word so proper nouns stay recognisable."""
    out: list[str] = []
    index = 0
    length = len(word)
    while index < length:
        char = word[index]
        if char in (ZWNJ, ZWJ):
            index += 1
            continue
        if char in CHILLUS:
            out.append(CHILLUS[char])
            index += 1
            continue
        if char in INDEPENDENT_VOWELS:
            out.append(INDEPENDENT_VOWELS[char])
            index += 1
            continue
        if char in CONSONANTS:
            out.append(CONSONANTS[char])
            nxt = word[index + 1] if index + 1 < length else ""
            if nxt == VIRAMA:
                # Virama kills the inherent 'a'; a following consonant clusters.
                index += 2
                continue
            if nxt in VOWEL_SIGNS:
                out.append(VOWEL_SIGNS[nxt])
                index += 2
                continue
            out.append("a")
            index += 1
            continue
        if char in VOWEL_SIGNS:
            out.append(VOWEL_SIGNS[char])
            index += 1
            continue
        if char == VIRAMA:
            index += 1
            continue
        out.append(char)
        index += 1
    result = "".join(out)
    return re.sub(r"a{2,}", "aa", result)


# ---------------------------------------------------------- glossary engine
@lru_cache(maxsize=1)
def _phrase_pattern() -> re.Pattern[str] | None:
    """One alternation of every glossary phrase, longest first."""
    phrases = sorted(MALAYALAM_PHRASES, key=len, reverse=True)
    if not phrases:
        return None
    return re.compile("|".join(re.escape(phrase) for phrase in phrases))


def translate_with_glossary(text: str) -> TranslationResult:
    """Offline, deterministic Malayalam → English using the KMRL glossary.

    Known domain phrases and words are replaced with their English equivalent;
    anything unknown is transliterated so names and identifiers survive. The
    result is not fluent prose, but it carries the domain meaning — which is
    what classification, routing, keyword search and embeddings need.
    """
    if not (text or "").strip():
        return TranslationResult(status=STATUS_NOT_REQUIRED)

    working = text.translate(MALAYALAM_DIGITS)

    pattern = _phrase_pattern()
    if pattern is not None:
        working = pattern.sub(lambda m: MALAYALAM_PHRASES[m.group(0)], working)

    known = 0
    total = 0

    def replace_token(match: re.Match[str]) -> str:
        nonlocal known, total
        token = match.group(0)
        stripped = token.strip(ZWNJ + ZWJ)
        if not stripped:
            return token
        total += 1
        direct = MALAYALAM_WORDS.get(stripped)
        if direct:
            known += 1
            return direct
        # Malayalam is agglutinative: try trimming common case suffixes.
        for suffix_length in (3, 2, 1):
            if len(stripped) > suffix_length + 1:
                stem = stripped[:-suffix_length]
                hit = MALAYALAM_WORDS.get(stem)
                if hit:
                    known += 1
                    return hit
        return transliterate(stripped)

    working = MALAYALAM_TOKEN.sub(replace_token, working)
    working = re.sub(r"[ \t]{2,}", " ", working)

    coverage = (known / total) if total else 1.0
    return TranslationResult(
        text=working.strip(),
        engine=ENGINE_GLOSSARY,
        model="kmrl-ml-en-glossary-v1",
        # Glossary coverage is an honest ceiling on quality, not a claim of
        # fluency — the UI shows it so a human knows when to check the original.
        confidence=round(0.35 + 0.45 * coverage, 2),
        status=STATUS_PARTIAL,
    )


# -------------------------------------------------------------- argos engine
@lru_cache(maxsize=1)
def argos_available() -> bool:
    try:
        import argostranslate.translate as argos
    except ImportError:
        return False
    try:
        languages = argos.get_installed_languages()
    except Exception:  # pragma: no cover - depends on local packages
        return False
    codes = {language.code for language in languages}
    return "ml" in codes and "en" in codes


def translate_with_argos(text: str) -> TranslationResult:
    import argostranslate.translate as argos

    languages = {language.code: language for language in argos.get_installed_languages()}
    translation = languages["ml"].get_translation(languages["en"])
    rendered = translation.translate(text)
    return TranslationResult(
        text=(rendered or "").strip(),
        engine=ENGINE_ARGOS,
        model="argos-ml-en",
        confidence=0.82,
        status=STATUS_TRANSLATED,
    )


# ----------------------------------------------------------- anthropic engine
TRANSLATION_SYSTEM_PROMPT = """You translate Kochi Metro Rail Limited (KMRL) documents from Malayalam into English.

Rules:
- Translate ONLY what is in the source. Never add, explain or summarise.
- Keep every number, date, amount, percentage, reference number, drawing number,
  invoice number and file number EXACTLY as written.
- Keep proper nouns (people, stations, vendors, places) in their normal English
  spelling; transliterate them if there is no established spelling. Never
  translate a name into its literal meaning.
- Use the standard English metro-rail term for technical, safety and regulatory
  vocabulary (for example: emergency brake, rolling stock, platform screen door,
  Commissioner of Metro Rail Safety, non-compliance, environmental clearance).
- Preserve the structure: line breaks, headings, list items and table rows stay
  where they are.
- Text that is already English must be passed through unchanged.
- Output the translation only — no preamble, no notes, no quotation marks."""


def translate_with_anthropic(text: str) -> TranslationResult:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    excerpt = text[:MAX_TRANSLATE_CHARS]
    response = client.messages.create(
        model=settings.kmrl_analysis_model,
        max_tokens=8000,
        system=TRANSLATION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Translate the following KMRL document text into English.\n\n"
                    "--- BEGIN SOURCE ---\n"
                    f"{excerpt}\n"
                    "--- END SOURCE ---"
                ),
            }
        ],
    )
    rendered = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    if not rendered:
        raise ValueError("The translation model returned no text.")
    return TranslationResult(
        text=rendered,
        engine=ENGINE_ANTHROPIC,
        model=settings.kmrl_analysis_model,
        confidence=0.94,
        status=STATUS_TRANSLATED if len(text) <= MAX_TRANSLATE_CHARS else STATUS_PARTIAL,
    )


# ------------------------------------------------------------------ dispatch
def translate_to_english(text: str, language: str | None = None) -> TranslationResult:
    """Normalise ``text`` to English, choosing the best engine available."""
    if not needs_translation(text, language):
        return TranslationResult(
            text=text or "", engine=ENGINE_NONE, status=STATUS_NOT_REQUIRED, confidence=1.0
        )

    if settings.llm_enabled:
        try:
            return translate_with_anthropic(text)
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the upload
            logger.warning("Claude translation failed (%s); falling back.", exc)

    if argos_available():
        try:
            return translate_with_argos(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Argos translation failed (%s); using the glossary.", exc)

    return translate_with_glossary(text)


def translation_capabilities() -> dict[str, object]:
    return {
        "engine": (
            ENGINE_ANTHROPIC
            if settings.llm_enabled
            else (ENGINE_ARGOS if argos_available() else ENGINE_GLOSSARY)
        ),
        "anthropic": settings.llm_enabled,
        "argos": argos_available(),
        "glossary_terms": len(MALAYALAM_WORDS) + len(MALAYALAM_PHRASES),
        "direction": "Malayalam → English",
    }


def language_of(text: str) -> str:
    from .extraction import detect_language

    return detect_language(text)


__all__ = [
    "ENGINE_ANTHROPIC",
    "ENGINE_ARGOS",
    "ENGINE_GLOSSARY",
    "ENGINE_NONE",
    "STATUS_FAILED",
    "STATUS_NOT_REQUIRED",
    "STATUS_PARTIAL",
    "STATUS_TRANSLATED",
    "TranslationResult",
    "needs_translation",
    "translate_to_english",
    "translate_with_glossary",
    "transliterate",
    "translation_capabilities",
    "ENGLISH",
    "MALAYALAM",
]
