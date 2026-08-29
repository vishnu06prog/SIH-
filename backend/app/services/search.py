"""Keyword, semantic and hybrid search over the document library.

Keyword ranking runs entirely in SQL — the API never loads the library into
memory and never ships raw document text to the browser. Semantic search adds
embedding similarity over chunks, and hybrid search blends the two so a query
finds both the document that says the words and the one that means them.

Malayalam documents are indexed through their English translation, so an
English query retrieves them; the original passage is returned alongside.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Float, and_, case, func, literal, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import ColumnElement, Select

from ..config import settings
from ..models import Department, Document, DocumentChunk
from ..taxonomy import (
    ORG_WIDE_ROLES,
    PRIORITY_RANK,
    STATUS_LABELS,
    normalise_status,
)
from .embeddings import search_similar_chunks

# Field weights — a hit in the title matters more than one deep in the body.
WEIGHT_FILENAME = 50
WEIGHT_TITLE = 44
WEIGHT_DOC_TYPE = 40
WEIGHT_KEYWORDS = 30
WEIGHT_SUMMARY = 22
WEIGHT_CONTENT = 10

RECENT_WINDOW_DAYS = 30
SNIPPET_RADIUS = 130
SNIPPET_LENGTH = 320

_TERM_RE = re.compile(r'"[^"]+"|\S+')

SORT_OPTIONS = {
    "relevance": "Best match",
    "newest": "Newest first",
    "oldest": "Oldest first",
    "name": "Title (A-Z)",
    "priority": "Priority (Critical first)",
}

DATE_RANGES = {
    "any": "Any time",
    "today": "Today",
    "week": "Last 7 days",
    "month": "Last 30 days",
    "quarter": "Last 90 days",
    "year": "Last 12 months",
}

MODE_KEYWORD = "keyword"
MODE_SEMANTIC = "semantic"
MODE_HYBRID = "hybrid"
SEARCH_MODES = (MODE_KEYWORD, MODE_SEMANTIC, MODE_HYBRID)


@dataclass
class SearchParams:
    q: str = ""
    mode: str = MODE_KEYWORD
    departments: list[str] = field(default_factory=list)  # department ids
    priorities: list[str] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    document_types: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    date_range: str = "any"
    date_from: datetime | None = None
    date_to: datetime | None = None
    sort: str = "relevance"
    page: int = 1
    page_size: int = 20
    # Restricts results to documents the caller may read (RBAC).
    visible_department_ids: list[str] | None = None
    visible_uploader_id: str | None = None

    def normalised_sort(self) -> str:
        if self.sort not in SORT_OPTIONS:
            return "relevance" if self.q.strip() else "newest"
        if self.sort == "relevance" and not self.q.strip():
            return "newest"
        return self.sort

    def normalised_mode(self) -> str:
        mode = (self.mode or MODE_KEYWORD).lower()
        return mode if mode in SEARCH_MODES else MODE_KEYWORD

    def as_applied_filters(self) -> dict[str, Any]:
        applied: dict[str, Any] = {}
        for key, values in (
            ("department", self.departments),
            ("priority", self.priorities),
            ("file_type", self.file_types),
            ("status", self.statuses),
            ("document_type", self.document_types),
            ("language", self.languages),
            ("source", self.sources),
            ("tag", self.tags),
        ):
            if values:
                applied[key] = values
        if self.date_range and self.date_range != "any":
            applied["date_range"] = self.date_range
        if self.date_from:
            applied["date_from"] = self.date_from.isoformat()
        if self.date_to:
            applied["date_to"] = self.date_to.isoformat()
        return applied

    @property
    def has_filters(self) -> bool:
        return bool(self.as_applied_filters())


def params_for_user(params: SearchParams, user) -> SearchParams:
    """Narrow a search to what ``user``'s role allows them to see."""
    if user is None:
        params.visible_department_ids = []
        return params
    if user.role_name in ORG_WIDE_ROLES:
        params.visible_department_ids = None
        params.visible_uploader_id = None
        return params
    params.visible_department_ids = [user.department_id] if user.department_id else []
    params.visible_uploader_id = user.id
    return params


def parse_terms(query: str) -> list[str]:
    """Split a query into terms, honouring "quoted phrases"."""
    if not query:
        return []
    terms: list[str] = []
    for raw in _TERM_RE.findall(query.strip().lower()):
        term = raw.strip('"').strip()
        if len(term) < 2:
            continue
        terms.append(term)
    if not terms:
        cleaned = query.strip().lower()
        if cleaned:
            terms.append(cleaned)
    return terms[:10]


def _like(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _column_match(column: ColumnElement, term: str) -> ColumnElement:
    return func.lower(column).like(_like(term), escape="\\")


def build_score_expression(terms: list[str]) -> ColumnElement:
    """Weighted relevance score, computed entirely inside the database."""
    if not terms:
        return literal(0.0)

    weighted = (
        (Document.filename, WEIGHT_FILENAME),
        (Document.title, WEIGHT_TITLE),
        (Document.document_type, WEIGHT_DOC_TYPE),
        (Document.keywords, WEIGHT_KEYWORDS),
        (Document.search_blob, WEIGHT_CONTENT),
    )
    parts: list[ColumnElement] = []
    for term in terms:
        for column, weight in weighted:
            parts.append(case((_column_match(column, term), literal(weight)), else_=literal(0)))

    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total.cast(Float)


def build_match_clause(terms: list[str]) -> ColumnElement | None:
    """Every term must appear somewhere in the document (AND across terms)."""
    if not terms:
        return None
    return and_(
        *[
            or_(
                _column_match(Document.search_blob, term),
                _column_match(Document.filename, term),
            )
            for term in terms
        ]
    )


def _date_bounds(params: SearchParams) -> tuple[datetime | None, datetime | None]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    start, end = params.date_from, params.date_to
    windows = {"today": 0, "week": 7, "month": 30, "quarter": 90, "year": 365}
    preset = (params.date_range or "any").lower()
    if preset in windows:
        floor = (
            now.replace(hour=0, minute=0, second=0, microsecond=0)
            if preset == "today"
            else now - timedelta(days=windows[preset])
        )
        start = max(start, floor) if start else floor
    return start, end


def apply_filters(stmt: Select, params: SearchParams) -> Select:
    if params.departments:
        stmt = stmt.where(Document.department_id.in_(params.departments))
    if params.priorities:
        stmt = stmt.where(Document.priority.in_(params.priorities))
    if params.file_types:
        stmt = stmt.where(Document.file_type.in_([ft.upper() for ft in params.file_types]))
    if params.document_types:
        stmt = stmt.where(Document.document_type.in_(params.document_types))
    if params.languages:
        stmt = stmt.where(Document.language.in_(params.languages))
    if params.sources:
        stmt = stmt.where(Document.source.in_(params.sources))
    if params.tags:
        stmt = stmt.where(
            or_(*[_column_match(Document.tags, tag.lower()) for tag in params.tags])
        )
    if params.statuses:
        resolved = [s for s in (normalise_status(s) for s in params.statuses) if s]
        stmt = stmt.where(Document.status.in_(resolved)) if resolved else stmt.where(literal(False))

    start, end = _date_bounds(params)
    if start:
        stmt = stmt.where(Document.uploaded_at >= start)
    if end:
        stmt = stmt.where(Document.uploaded_at <= end)

    stmt = apply_visibility(stmt, params)
    return stmt


def apply_visibility(stmt: Select, params: SearchParams) -> Select:
    """Restrict to documents the caller is allowed to read."""
    if params.visible_department_ids is None:
        return stmt  # admin / executive: everything

    from ..models import RoutingResult

    clauses: list[ColumnElement] = [Document.confidentiality == "Public"]
    if params.visible_uploader_id:
        clauses.append(Document.uploaded_by_id == params.visible_uploader_id)
    if params.visible_department_ids:
        clauses.append(Document.department_id.in_(params.visible_department_ids))
        clauses.append(Document.declared_department_id.in_(params.visible_department_ids))
        clauses.append(
            Document.id.in_(
                select(RoutingResult.document_id).where(
                    RoutingResult.department_id.in_(params.visible_department_ids)
                )
            )
        )
    return stmt.where(or_(*clauses))


def _order_by(sort: str, score: ColumnElement) -> list[ColumnElement]:
    if sort == "newest":
        return [Document.uploaded_at.desc(), Document.id.desc()]
    if sort == "oldest":
        return [Document.uploaded_at.asc(), Document.id.asc()]
    if sort == "name":
        return [func.lower(Document.title).asc(), Document.id.asc()]
    if sort == "priority":
        rank = case(
            *[(Document.priority == name, rank) for name, rank in PRIORITY_RANK.items()],
            else_=9,
        )
        return [rank.asc(), Document.uploaded_at.desc()]
    return [score.desc(), Document.uploaded_at.desc(), Document.id.desc()]


def _snippet_expressions(terms: list[str], dialect: str) -> tuple[ColumnElement, ColumnElement]:
    """A ~320-character window around the first hit, computed inside SQL."""
    preview = func.substr(Document.normalised_text, 1, SNIPPET_LENGTH)
    if not terms or dialect != "sqlite":
        return preview, literal(1)

    lowered = func.lower(Document.normalised_text)
    text_expression: ColumnElement = preview
    start_expression: ColumnElement = literal(1)
    for term in reversed(terms):
        position = func.instr(lowered, term.lower())
        start = func.max(position - SNIPPET_RADIUS, 1)
        text_expression = case(
            (position > 0, func.substr(Document.normalised_text, start, SNIPPET_LENGTH)),
            else_=text_expression,
        )
        start_expression = case((position > 0, start), else_=start_expression)
    return text_expression, start_expression


SELECTED_COLUMNS = (
    Document.id,
    Document.title,
    Document.filename,
    Document.file_type,
    Document.file_ext,
    Document.size_bytes,
    Document.uploaded_at,
    Document.processed_at,
    Document.status,
    Document.status_detail,
    Document.progress,
    Document.page_count,
    Document.word_count,
    Document.language,
    Document.languages_detected,
    Document.translation_status,
    Document.ocr_used,
    Document.ocr_quality,
    Document.needs_ocr_review,
    Document.document_type,
    Document.department_id,
    Document.priority,
    Document.confidentiality,
    Document.source,
    Document.tags,
    Document.keywords,
    Document.is_reviewed,
    Document.analysis_engine,
    Document.analysis_confidence,
    Document.duplicate_of_id,
    Document.version_no,
)


def _decorate(rows: list[dict[str, Any]], db: Session) -> list[dict[str, Any]]:
    """Attach department names, summaries and derived counts to result rows."""
    from ..models import Action, Deadline, DocumentSummary

    department_names = {
        row.id: row.name for row in db.execute(select(Department)).scalars()
    }
    ids = [row["id"] for row in rows]
    if not ids:
        return rows

    summaries = {
        row.document_id: row
        for row in db.execute(
            select(DocumentSummary).where(
                DocumentSummary.document_id.in_(ids), DocumentSummary.is_current.is_(True)
            )
        ).scalars()
    }
    action_counts = dict(
        db.execute(
            select(Action.document_id, func.count(Action.id))
            .where(Action.document_id.in_(ids))
            .group_by(Action.document_id)
        ).all()
    )
    deadline_counts = dict(
        db.execute(
            select(Deadline.document_id, func.count(Deadline.id))
            .where(Deadline.document_id.in_(ids))
            .group_by(Deadline.document_id)
        ).all()
    )

    for row in rows:
        summary = summaries.get(row["id"])
        row["summary"] = summary.executive_summary if summary else ""
        row["department"] = department_names.get(row.pop("department_id", None), "Unassigned")
        row["keywords"] = _json_list(row.get("keywords"))
        row["tags"] = _json_list(row.get("tags"))
        row["languages_detected"] = _json_list(row.get("languages_detected"))
        row["action_count"] = int(action_counts.get(row["id"], 0))
        row["deadline_count"] = int(deadline_counts.get(row["id"], 0))
        row["status_label"] = STATUS_LABELS.get(row["status"], row["status"])
        row["uploaded_at"] = row["uploaded_at"].isoformat() if row.get("uploaded_at") else None
        row["processed_at"] = row["processed_at"].isoformat() if row.get("processed_at") else None
    return rows


def search_documents(db: Session, params: SearchParams) -> tuple[list[dict[str, Any]], int]:
    """Keyword search: ``(rows, total_matches)``."""
    terms = parse_terms(params.q)
    score = build_score_expression(terms)
    match_clause = build_match_clause(terms)
    dialect = db.bind.dialect.name if db.bind is not None else "sqlite"

    count_stmt = apply_filters(select(func.count(Document.id)), params)
    if match_clause is not None:
        count_stmt = count_stmt.where(match_clause)
    total = int(db.execute(count_stmt).scalar_one())

    snippet_text, snippet_start = _snippet_expressions(terms, dialect)
    stmt = select(
        *SELECTED_COLUMNS,
        score.label("score"),
        snippet_text.label("snippet"),
        snippet_start.label("snippet_start"),
    )
    stmt = apply_filters(stmt, params)
    if match_clause is not None:
        stmt = stmt.where(match_clause)
    stmt = stmt.order_by(*_order_by(params.normalised_sort(), score))
    stmt = stmt.limit(params.page_size).offset((params.page - 1) * params.page_size)

    rows = [dict(row._mapping) for row in db.execute(stmt)]
    for row in rows:
        clipped = int(row.pop("snippet_start", 1) or 1) > 1
        row["snippet"] = _format_snippet(row.get("snippet"), clipped)
        row["matched_text"] = row["snippet"]
        row["score"] = round(float(row.get("score") or 0), 1)
        row["match_mode"] = MODE_KEYWORD
        row["page_number"] = None
    return _decorate(rows, db), total


def semantic_search(
    db: Session, params: SearchParams, top_k: int | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Embedding search over chunks, collapsed to one row per document."""
    if not params.q.strip():
        return search_documents(db, params)

    allowed_ids = _allowed_document_ids(db, params)
    if allowed_ids is not None and not allowed_ids:
        return [], 0

    top_k = top_k or max(params.page_size * 4, settings.kmrl_semantic_top_k * 4)
    hits = search_similar_chunks(db, params.q, top_k=top_k, document_ids=allowed_ids)

    best: dict[str, Any] = {}
    for hit in hits:
        current = best.get(hit.document_id)
        if current is None or hit.score > current["score"]:
            best[hit.document_id] = {
                "score": hit.score,
                "page_number": hit.page_number,
                "matched_text": _trim_snippet(hit.text),
                "matched_original": _trim_snippet(hit.original_text) if hit.original_text else "",
                "language": hit.language,
                "section": hit.section,
            }

    if not best:
        return [], 0

    ordered_ids = sorted(best, key=lambda doc_id: best[doc_id]["score"], reverse=True)
    total = len(ordered_ids)
    start = (params.page - 1) * params.page_size
    page_ids = ordered_ids[start : start + params.page_size]
    if not page_ids:
        return [], total

    stmt = apply_filters(select(*SELECTED_COLUMNS), params).where(Document.id.in_(page_ids))
    rows = {row._mapping["id"]: dict(row._mapping) for row in db.execute(stmt)}

    results: list[dict[str, Any]] = []
    for document_id in page_ids:
        row = rows.get(document_id)
        if row is None:
            continue
        detail = best[document_id]
        row["score"] = round(detail["score"] * 100, 1)
        row["snippet"] = detail["matched_text"]
        row["matched_text"] = detail["matched_text"]
        row["matched_original"] = detail["matched_original"]
        row["page_number"] = detail["page_number"]
        row["section"] = detail["section"]
        row["match_mode"] = MODE_SEMANTIC
        results.append(row)
    return _decorate(results, db), total


def hybrid_search(db: Session, params: SearchParams) -> tuple[list[dict[str, Any]], int]:
    """Blend keyword and semantic scores so both kinds of match surface."""
    keyword_params = SearchParams(**{**params.__dict__, "page": 1, "page_size": 200})
    keyword_rows, _ = search_documents(db, keyword_params)
    semantic_rows, _ = semantic_search(db, keyword_params, top_k=400)

    alpha = min(max(settings.kmrl_hybrid_alpha, 0.0), 1.0)
    max_keyword = max((row["score"] for row in keyword_rows), default=0.0) or 1.0

    merged: dict[str, dict[str, Any]] = {}
    for row in keyword_rows:
        row = dict(row)
        row["keyword_score"] = row["score"]
        row["semantic_score"] = 0.0
        merged[row["id"]] = row
    for row in semantic_rows:
        existing = merged.get(row["id"])
        if existing is None:
            row = dict(row)
            row["keyword_score"] = 0.0
            row["semantic_score"] = row["score"]
            merged[row["id"]] = row
        else:
            existing["semantic_score"] = row["score"]
            if not existing.get("matched_text") or existing["keyword_score"] == 0:
                existing["matched_text"] = row["matched_text"]
                existing["snippet"] = row["matched_text"]
                existing["page_number"] = row.get("page_number")
                existing["matched_original"] = row.get("matched_original", "")

    for row in merged.values():
        normalised_keyword = (row["keyword_score"] / max_keyword) * 100
        row["score"] = round(
            alpha * row["semantic_score"] + (1 - alpha) * normalised_keyword, 1
        )
        row["match_mode"] = MODE_HYBRID

    ordered = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    total = len(ordered)
    start = (params.page - 1) * params.page_size
    return ordered[start : start + params.page_size], total


def run_search(db: Session, params: SearchParams) -> tuple[list[dict[str, Any]], int]:
    mode = params.normalised_mode()
    if mode == MODE_SEMANTIC:
        return semantic_search(db, params)
    if mode == MODE_HYBRID:
        return hybrid_search(db, params)
    return search_documents(db, params)


def _allowed_document_ids(db: Session, params: SearchParams) -> list[str] | None:
    """Document ids matching the filters — ``None`` means "no restriction"."""
    if params.visible_department_ids is None and not params.has_filters:
        return None
    stmt = apply_filters(select(Document.id), params)
    return [row for row in db.execute(stmt).scalars()]


def _trim_snippet(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= SNIPPET_LENGTH:
        return cleaned
    return cleaned[:SNIPPET_LENGTH].rsplit(" ", 1)[0] + "…"


def _format_snippet(raw: str | None, clipped_start: bool) -> str | None:
    if not raw:
        return None
    text = re.sub(r"\s+", " ", raw).strip()
    if not text:
        return None
    if clipped_start and " " in text[:40]:
        text = "…" + text.split(" ", 1)[1]
    if len(text) >= SNIPPET_LENGTH:
        text = text.rsplit(" ", 1)[0] + "…"
    return text


def _json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def facet_counts(db: Session, params: SearchParams) -> dict[str, list[dict[str, Any]]]:
    """Counts per filter value. A facet ignores its own filter, so the user can
    always see and switch to sibling values instead of watching the list
    collapse to the single option they just picked."""
    terms = parse_terms(params.q)
    match_clause = build_match_clause(terms)
    department_names = {row.id: row.name for row in db.execute(select(Department)).scalars()}

    def counts_for(column, exclude: str) -> list[dict[str, Any]]:
        scoped = SearchParams(
            q=params.q,
            departments=[] if exclude == "department" else params.departments,
            priorities=[] if exclude == "priority" else params.priorities,
            file_types=[] if exclude == "file_type" else params.file_types,
            statuses=[] if exclude == "status" else params.statuses,
            document_types=[] if exclude == "document_type" else params.document_types,
            languages=[] if exclude == "language" else params.languages,
            sources=[] if exclude == "source" else params.sources,
            date_range=params.date_range,
            date_from=params.date_from,
            date_to=params.date_to,
            visible_department_ids=params.visible_department_ids,
            visible_uploader_id=params.visible_uploader_id,
        )
        stmt = apply_filters(select(column, func.count(Document.id)), scoped)
        if match_clause is not None:
            stmt = stmt.where(match_clause)
        stmt = stmt.group_by(column).order_by(func.count(Document.id).desc())
        return [
            {"value": value, "label": value, "count": int(count)}
            for value, count in db.execute(stmt)
            if value
        ]

    departments = counts_for(Document.department_id, "department")
    for item in departments:
        item["label"] = department_names.get(item["value"], "Unassigned")

    statuses = counts_for(Document.status, "status")
    for item in statuses:
        item["label"] = STATUS_LABELS.get(item["value"], item["value"])

    return {
        "departments": departments,
        "priorities": counts_for(Document.priority, "priority"),
        "file_types": counts_for(Document.file_type, "file_type"),
        "statuses": statuses,
        "document_types": counts_for(Document.document_type, "document_type"),
        "languages": counts_for(Document.language, "language"),
        "sources": counts_for(Document.source, "source"),
    }


def build_search_blob(document: Document, extra: list[str] | None = None) -> str:
    """Lower-cased haystack backing the keyword match clause.

    Both the original text AND its English translation go in, which is what
    lets an English query match a Malayalam document — and a Malayalam query
    match the Malayalam original.
    """
    parts = [
        document.filename,
        document.title,
        document.document_type,
        document.priority,
        document.language,
        document.source,
        " ".join(str(k) for k in document.json_list("keywords")),
        " ".join(str(t) for t in document.json_list("tags")),
        document.normalised_text or "",
        document.extracted_text or "",
    ]
    if extra:
        parts.extend(extra)
    return " \n".join(part for part in parts if part).lower()


def chunk_page_numbers(db: Session, document_id: str) -> dict[str, int | None]:
    return {
        chunk_id: page
        for chunk_id, page in db.execute(
            select(DocumentChunk.id, DocumentChunk.page_number).where(
                DocumentChunk.document_id == document_id
            )
        ).all()
    }


def date_bounds_for(range_key: str) -> tuple[date | None, date | None]:
    params = SearchParams(date_range=range_key)
    start, end = _date_bounds(params)
    return (start.date() if start else None, end.date() if end else None)
