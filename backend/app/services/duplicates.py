"""Duplicate and version detection.

Three signals, cheapest first:

``checksum``   byte-identical upload — an exact duplicate, no doubt about it.
``similarity`` cosine distance between document embeddings — near-duplicates and
               revisions of the same circular.
``shingles``   Jaccard overlap of word 5-grams, which catches "same document,
               different wording" that embeddings smooth over.

Above ``kmrl_duplicate_threshold`` the upload is flagged as a duplicate; between
that and ``kmrl_version_threshold`` it is proposed as a NEW VERSION of the
earlier document, with a diff of what changed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Action, Deadline, Document
from ..taxonomy import STATUS_COMPLETED
from .embeddings import cosine, document_vector

logger = logging.getLogger(__name__)

SHINGLE_SIZE = 5
WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class SimilarDocument:
    document_id: str
    title: str
    filename: str
    similarity: float
    reason: str
    uploaded_at: str = ""
    is_exact: bool = False


@dataclass
class DuplicateReport:
    exact_duplicate_id: str | None = None
    best_match_id: str | None = None
    best_similarity: float = 0.0
    is_duplicate: bool = False
    is_new_version: bool = False
    candidates: list[SimilarDocument] = field(default_factory=list)


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    words = WORD_RE.findall((text or "").lower())
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def find_similar(
    db: Session, document: Document, limit: int = 5
) -> DuplicateReport:
    """Compare a freshly processed document against the existing library."""
    report = DuplicateReport()

    exact = db.execute(
        select(Document)
        .where(
            Document.checksum == document.checksum,
            Document.id != document.id,
            Document.checksum != "",
        )
        .order_by(Document.uploaded_at.asc())
        .limit(1)
    ).scalars().first()

    if exact is not None:
        report.exact_duplicate_id = exact.id
        report.best_match_id = exact.id
        report.best_similarity = 1.0
        report.is_duplicate = True
        report.candidates.append(
            SimilarDocument(
                document_id=exact.id,
                title=exact.title or exact.filename,
                filename=exact.filename,
                similarity=1.0,
                reason="Byte-identical file (matching SHA-256 checksum).",
                uploaded_at=exact.uploaded_at.isoformat() if exact.uploaded_at else "",
                is_exact=True,
            )
        )
        return report

    own_vector = document_vector(db, document.id)
    own_shingles = shingles(document.normalised_text or document.extracted_text)
    if own_vector is None and not own_shingles:
        return report

    others = db.execute(
        select(Document)
        .where(
            Document.id != document.id,
            Document.status == STATUS_COMPLETED,
        )
        .order_by(Document.uploaded_at.desc())
        .limit(400)
    ).scalars().all()

    scored: list[SimilarDocument] = []
    for other in others:
        vector_score = 0.0
        if own_vector is not None:
            other_vector = document_vector(db, other.id)
            if other_vector is not None:
                vector_score = cosine(own_vector, other_vector)

        text_score = jaccard(
            own_shingles, shingles(other.normalised_text or other.extracted_text)
        )
        # Embeddings find "about the same thing"; shingles find "literally the
        # same wording". Taking the max keeps both kinds of duplicate visible.
        score = max(vector_score, text_score)
        if score < 0.45:
            continue
        scored.append(
            SimilarDocument(
                document_id=other.id,
                title=other.title or other.filename,
                filename=other.filename,
                similarity=round(score, 3),
                reason=(
                    f"{round(text_score * 100)}% of the wording overlaps"
                    if text_score >= vector_score
                    else f"{round(vector_score * 100)}% semantic similarity"
                ),
                uploaded_at=other.uploaded_at.isoformat() if other.uploaded_at else "",
            )
        )

    scored.sort(key=lambda item: item.similarity, reverse=True)
    report.candidates = scored[:limit]
    if scored:
        best = scored[0]
        report.best_match_id = best.document_id
        report.best_similarity = best.similarity
        report.is_duplicate = best.similarity >= settings.kmrl_duplicate_threshold
        report.is_new_version = (
            not report.is_duplicate and best.similarity >= settings.kmrl_version_threshold
        )
    return report


# ------------------------------------------------------------------- diffing
@dataclass
class VersionDiff:
    added_sections: list[str] = field(default_factory=list)
    removed_sections: list[str] = field(default_factory=list)
    changed_deadlines: list[dict[str, str]] = field(default_factory=list)
    new_deadlines: list[dict[str, str]] = field(default_factory=list)
    removed_requirements: list[str] = field(default_factory=list)
    changed_responsibilities: list[dict[str, str]] = field(default_factory=list)
    similarity: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "added_sections": self.added_sections,
            "removed_sections": self.removed_sections,
            "new_deadlines": self.new_deadlines,
            "changed_deadlines": self.changed_deadlines,
            "removed_requirements": self.removed_requirements,
            "changed_responsibilities": self.changed_responsibilities,
            "similarity": self.similarity,
        }


def _paragraphs(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", block).strip()
        for block in re.split(r"\n\s*\n", text or "")
        if len(block.strip()) > 40
    ]


def diff_versions(db: Session, current: Document, previous: Document) -> VersionDiff:
    """What changed between two versions of the same document."""
    diff = VersionDiff()

    current_blocks = _paragraphs(current.normalised_text or current.extracted_text)
    previous_blocks = _paragraphs(previous.normalised_text or previous.extracted_text)
    current_keys = {block.lower(): block for block in current_blocks}
    previous_keys = {block.lower(): block for block in previous_blocks}

    diff.added_sections = [
        text[:400] for key, text in current_keys.items() if key not in previous_keys
    ][:12]
    diff.removed_sections = [
        text[:400] for key, text in previous_keys.items() if key not in current_keys
    ][:12]

    current_deadlines = db.execute(
        select(Deadline).where(Deadline.document_id == current.id)
    ).scalars().all()
    previous_deadlines = db.execute(
        select(Deadline).where(Deadline.document_id == previous.id)
    ).scalars().all()

    previous_by_title = {
        (row.title or "").lower()[:60]: row for row in previous_deadlines
    }
    for row in current_deadlines:
        key = (row.title or "").lower()[:60]
        earlier = previous_by_title.get(key)
        if earlier is None:
            diff.new_deadlines.append(
                {
                    "title": row.title,
                    "due_date": row.due_date.isoformat() if row.due_date else row.due_date_text,
                    "type": row.deadline_type,
                }
            )
        elif (earlier.due_date or earlier.due_date_text) != (row.due_date or row.due_date_text):
            diff.changed_deadlines.append(
                {
                    "title": row.title,
                    "was": earlier.due_date.isoformat() if earlier.due_date else earlier.due_date_text,
                    "now": row.due_date.isoformat() if row.due_date else row.due_date_text,
                }
            )

    current_actions = db.execute(
        select(Action).where(Action.document_id == current.id)
    ).scalars().all()
    previous_actions = db.execute(
        select(Action).where(Action.document_id == previous.id)
    ).scalars().all()

    current_action_keys = {(a.description or "").lower()[:80]: a for a in current_actions}
    previous_action_keys = {(a.description or "").lower()[:80]: a for a in previous_actions}

    diff.removed_requirements = [
        action.description[:300]
        for key, action in previous_action_keys.items()
        if key not in current_action_keys
    ][:10]

    for key, action in current_action_keys.items():
        earlier = previous_action_keys.get(key)
        if earlier is None:
            continue
        was = earlier.responsible_department.name if earlier.responsible_department else ""
        now = action.responsible_department.name if action.responsible_department else ""
        if was != now:
            diff.changed_responsibilities.append(
                {"action": action.description[:200], "was": was or "—", "now": now or "—"}
            )

    return diff
