"""Keyword search, filtering, ranking and pagination for the document library.

Everything here runs in SQL: the API never loads the whole library into memory
and never ships raw document text to the browser.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Float, and_, case, func, literal, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import ColumnElement, Select

from ..models import Document
from ..taxonomy import STATUS_LABELS, normalise_status

# Field weights — the ranking order required by the library spec.
WEIGHT_FILENAME = 50
WEIGHT_TITLE = 34
WEIGHT_DOC_TYPE = 40
WEIGHT_DEPARTMENT = 30
WEIGHT_KEYWORDS = 26
WEIGHT_SUMMARY = 20
WEIGHT_CONTENT = 10

RECENT_WINDOW_DAYS = 30
SNIPPET_RADIUS = 130
SNIPPET_LENGTH = 320

_TERM_RE = re.compile(r'"[^"]+"|\S+')

SORT_OPTIONS = {
    "relevance": "Best match",
    "newest": "Newest first",
    "oldest": "Oldest first",
    "name": "Filename (A-Z)",
    "priority": "Priority (High first)",
}

DATE_RANGES = {
    "any": "Any time",
    "today": "Today",
    "week": "Last 7 days",
    "recent": f"Last {RECENT_WINDOW_DAYS} days",
    "older": f"Older than {RECENT_WINDOW_DAYS} days",
}


@dataclass
class SearchParams:
    q: str = ""
    departments: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    document_types: list[str] = field(default_factory=list)
    date_range: str = "any"
    date_from: datetime | None = None
    date_to: datetime | None = None
    sort: str = "relevance"
    page: int = 1
    page_size: int = 20

    def normalised_sort(self) -> str:
        if self.sort not in SORT_OPTIONS:
            return "relevance" if self.q.strip() else "newest"
        if self.sort == "relevance" and not self.q.strip():
            return "newest"
        return self.sort

    def as_applied_filters(self) -> dict[str, Any]:
        applied: dict[str, Any] = {}
        if self.departments:
            applied["department"] = self.departments
        if self.priorities:
            applied["priority"] = self.priorities
        if self.file_types:
            applied["file_type"] = self.file_types
        if self.statuses:
            applied["status"] = self.statuses
        if self.document_types:
            applied["document_type"] = self.document_types
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


def parse_terms(query: str) -> list[str]:
    """Split a query into search terms, honouring "quoted phrases"."""
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
    return terms[:8]


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
        (Document.department, WEIGHT_DEPARTMENT),
        (Document.keywords, WEIGHT_KEYWORDS),
        (Document.summary, WEIGHT_SUMMARY),
        (Document.extracted_text, WEIGHT_CONTENT),
    )

    parts: list[ColumnElement] = []
    for term in terms:
        for column, weight in weighted:
            parts.append(
                case((_column_match(column, term), literal(weight)), else_=literal(0))
            )

    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total.cast(Float)


def build_match_clause(terms: list[str]) -> ColumnElement | None:
    """Every term must appear somewhere in the document (AND across terms)."""
    if not terms:
        return None
    clauses = [
        or_(
            _column_match(Document.search_blob, term),
            _column_match(Document.filename, term),
        )
        for term in terms
    ]
    return and_(*clauses)


def _date_bounds(params: SearchParams) -> tuple[datetime | None, datetime | None]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    start, end = params.date_from, params.date_to
    preset = (params.date_range or "any").lower()
    if preset == "today":
        floor = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = max(start, floor) if start else floor
    elif preset == "week":
        floor = now - timedelta(days=7)
        start = max(start, floor) if start else floor
    elif preset == "recent":
        floor = now - timedelta(days=RECENT_WINDOW_DAYS)
        start = max(start, floor) if start else floor
    elif preset == "older":
        boundary = now - timedelta(days=RECENT_WINDOW_DAYS)
        end = min(end, boundary) if end else boundary
    return start, end


def apply_filters(stmt: Select, params: SearchParams) -> Select:
    if params.departments:
        stmt = stmt.where(Document.department.in_(params.departments))
    if params.priorities:
        stmt = stmt.where(Document.priority.in_(params.priorities))
    if params.file_types:
        stmt = stmt.where(Document.file_type.in_([ft.upper() for ft in params.file_types]))
    if params.document_types:
        stmt = stmt.where(Document.document_type.in_(params.document_types))
    if params.statuses:
        resolved = [s for s in (normalise_status(s) for s in params.statuses) if s]
        if resolved:
            stmt = stmt.where(Document.status.in_(resolved))
        else:
            stmt = stmt.where(literal(False))
    start, end = _date_bounds(params)
    if start:
        stmt = stmt.where(Document.uploaded_at >= start)
    if end:
        stmt = stmt.where(Document.uploaded_at <= end)
    return stmt


def _order_by(sort: str, score: ColumnElement) -> list[ColumnElement]:
    if sort == "newest":
        return [Document.uploaded_at.desc(), Document.id.desc()]
    if sort == "oldest":
        return [Document.uploaded_at.asc(), Document.id.asc()]
    if sort == "name":
        return [func.lower(Document.filename).asc(), Document.id.asc()]
    if sort == "priority":
        rank = case(
            (Document.priority == "High", 0),
            (Document.priority == "Medium", 1),
            else_=2,
        )
        return [rank.asc(), Document.uploaded_at.desc()]
    return [score.desc(), Document.uploaded_at.desc(), Document.id.desc()]


def _snippet_expressions(terms: list[str], dialect: str) -> tuple[ColumnElement, ColumnElement]:
    """A ~320 character window around the first hit, plus where it started.

    Both are computed inside SQL so the API never pulls whole documents back to
    Python just to build an excerpt.
    """
    preview = func.substr(Document.extracted_text, 1, SNIPPET_LENGTH)
    if not terms or dialect != "sqlite":
        return preview, literal(1)

    lowered = func.lower(Document.extracted_text)
    text_expression: ColumnElement = preview
    start_expression: ColumnElement = literal(1)
    for term in reversed(terms):
        position = func.instr(lowered, term.lower())
        start = func.max(position - SNIPPET_RADIUS, 1)
        text_expression = case(
            (position > 0, func.substr(Document.extracted_text, start, SNIPPET_LENGTH)),
            else_=text_expression,
        )
        start_expression = case((position > 0, start), else_=start_expression)
    return text_expression, start_expression


SELECTED_COLUMNS = (
    Document.id,
    Document.filename,
    Document.title,
    Document.file_type,
    Document.file_ext,
    Document.size_bytes,
    Document.uploaded_at,
    Document.processed_at,
    Document.status,
    Document.status_detail,
    Document.page_count,
    Document.word_count,
    Document.language,
    Document.ocr_used,
    Document.ocr_quality,
    Document.document_type,
    Document.department,
    Document.priority,
    Document.summary,
    Document.keywords,
    Document.actions,
    Document.deadlines,
    Document.analysis_engine,
)


def search_documents(db: Session, params: SearchParams) -> tuple[list[dict[str, Any]], int]:
    """Return ``(rows, total_matches)`` for the given search parameters."""
    terms = parse_terms(params.q)
    score = build_score_expression(terms)
    match_clause = build_match_clause(terms)
    dialect = db.bind.dialect.name if db.bind is not None else "sqlite"

    count_stmt = select(func.count(Document.id))
    count_stmt = apply_filters(count_stmt, params)
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

    sort = params.normalised_sort()
    stmt = stmt.order_by(*_order_by(sort, score))
    stmt = stmt.limit(params.page_size).offset((params.page - 1) * params.page_size)

    rows = [dict(row._mapping) for row in db.execute(stmt)]
    for row in rows:
        row["snippet"] = _format_snippet(row.get("snippet"), int(row.pop("snippet_start", 1) or 1) > 1)
        row["keywords"] = _json_list(row.get("keywords"))
        row["action_count"] = len(_json_list(row.get("actions")))
        row["deadline_count"] = len(_json_list(row.get("deadlines")))
        row.pop("actions", None)
        row.pop("deadlines", None)
        row["status_label"] = STATUS_LABELS.get(row["status"], row["status"])
        row["score"] = round(float(row.get("score") or 0), 1)
    return rows, total


def _format_snippet(raw: str | None, clipped_start: bool) -> str | None:
    """Tidy a raw SQL window into a readable excerpt with whole words."""
    if not raw:
        return None
    text = re.sub(r"\s+", " ", raw).strip()
    if not text:
        return None
    if clipped_start and " " in text[:40]:
        # The window may start mid-word — drop the fragment.
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
    """Counts per filter value for the current query.

    A facet ignores its own filter so the user can always see (and switch to)
    the sibling values instead of watching the list collapse to one option.
    """
    terms = parse_terms(params.q)
    match_clause = build_match_clause(terms)

    def counts_for(column, exclude: str) -> list[dict[str, Any]]:
        scoped = SearchParams(
            q=params.q,
            departments=[] if exclude == "department" else params.departments,
            priorities=[] if exclude == "priority" else params.priorities,
            file_types=[] if exclude == "file_type" else params.file_types,
            statuses=[] if exclude == "status" else params.statuses,
            document_types=[] if exclude == "document_type" else params.document_types,
            date_range=params.date_range,
            date_from=params.date_from,
            date_to=params.date_to,
        )
        stmt = select(column, func.count(Document.id))
        stmt = apply_filters(stmt, scoped)
        if match_clause is not None:
            stmt = stmt.where(match_clause)
        stmt = stmt.group_by(column).order_by(func.count(Document.id).desc())
        return [
            {"value": value, "label": value, "count": int(count)}
            for value, count in db.execute(stmt)
            if value
        ]

    statuses = counts_for(Document.status, "status")
    for item in statuses:
        item["label"] = STATUS_LABELS.get(item["value"], item["value"])

    return {
        "departments": counts_for(Document.department, "department"),
        "priorities": counts_for(Document.priority, "priority"),
        "file_types": counts_for(Document.file_type, "file_type"),
        "statuses": statuses,
        "document_types": counts_for(Document.document_type, "document_type"),
    }


def build_search_blob(document: Document) -> str:
    """Lower-cased haystack used for the AND-across-terms match clause."""
    keywords = " ".join(str(k) for k in document.json_field("keywords"))
    key_points = " ".join(str(k) for k in document.json_field("key_points"))
    actions = " ".join(
        str(item.get("action", ""))
        for item in document.json_field("actions")
        if isinstance(item, dict)
    )
    parts = [
        document.filename,
        document.title,
        document.document_type,
        document.department,
        " ".join(str(d) for d in document.json_field("secondary_departments")),
        document.priority,
        document.summary,
        keywords,
        key_points,
        actions,
        document.language,
        document.extracted_text or "",
    ]
    return " \n".join(part for part in parts if part).lower()
