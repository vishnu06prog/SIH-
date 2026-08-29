"""Document upload, library, search and intelligence endpoints."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Document
from ..schemas import (
    DocumentDetail,
    DocumentSearchResponse,
    DocumentSummary,
    FilterOptions,
    LibraryStats,
    SearchFacets,
    UploadResponse,
)
from ..services import extraction as extraction_service
from ..services import processing as processing_service
from ..services import search as search_service
from ..taxonomy import (
    DEPARTMENTS,
    DOCUMENT_TYPES,
    FILE_TYPES,
    PRIORITIES,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_LABELS,
    STATUSES,
    UNASSIGNED_DEPARTMENT,
)

router = APIRouter(prefix="/documents", tags=["documents"])

TEXT_PREVIEW_CHARS = 4000


# ----------------------------------------------------------------- helpers
def _split_multi(values: list[str] | None) -> list[str]:
    """Accept ``?department=A&department=B`` and ``?department=A,B`` alike."""
    result: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
    return result


def _document_summary(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "filename": document.filename,
        "title": document.title,
        "file_type": document.file_type,
        "file_ext": document.file_ext,
        "size_bytes": document.size_bytes,
        "uploaded_at": document.uploaded_at,
        "processed_at": document.processed_at,
        "status": document.status,
        "status_label": document.status_label,
        "status_detail": document.status_detail,
        "page_count": document.page_count,
        "word_count": document.word_count,
        "language": document.language,
        "ocr_used": document.ocr_used,
        "ocr_quality": document.ocr_quality,
        "document_type": document.document_type,
        "department": document.department,
        "priority": document.priority,
        "summary": document.summary,
        "keywords": document.json_field("keywords"),
        "action_count": len(document.json_field("actions")),
        "deadline_count": len(document.json_field("deadlines")),
        "analysis_engine": document.analysis_engine,
    }


def _build_params(
    q: str,
    department: list[str],
    priority: list[str],
    file_type: list[str],
    status: list[str],
    document_type: list[str],
    date_range: str,
    date_from: datetime | None,
    date_to: datetime | None,
    sort: str,
    page: int,
    page_size: int,
) -> search_service.SearchParams:
    return search_service.SearchParams(
        q=(q or "").strip(),
        departments=_split_multi(department),
        priorities=_split_multi(priority),
        file_types=_split_multi(file_type),
        statuses=_split_multi(status),
        document_types=_split_multi(document_type),
        date_range=(date_range or "any").strip().lower(),
        date_from=date_from,
        date_to=date_to,
        sort=(sort or "relevance").strip().lower(),
        page=max(1, page),
        page_size=max(1, min(page_size, settings.kmrl_max_page_size)),
    )


# ----------------------------------------------------------------- metadata
@router.get("/filters", response_model=FilterOptions)
def filter_options() -> FilterOptions:
    """Static filter vocabulary so the UI never hard-codes KMRL taxonomy."""
    return FilterOptions(
        departments=DEPARTMENTS + [UNASSIGNED_DEPARTMENT],
        priorities=PRIORITIES,
        file_types=FILE_TYPES,
        document_types=DOCUMENT_TYPES,
        statuses=[{"value": s, "label": STATUS_LABELS[s]} for s in STATUSES],
        sort_options=[
            {"value": key, "label": label}
            for key, label in search_service.SORT_OPTIONS.items()
        ],
        date_ranges=[
            {"value": key, "label": label}
            for key, label in search_service.DATE_RANGES.items()
        ],
    )


@router.get("/stats", response_model=LibraryStats)
def library_stats(db: Session = Depends(get_db)) -> LibraryStats:
    total = db.execute(select(func.count(Document.id))).scalar_one()
    complete = db.execute(
        select(func.count(Document.id)).where(Document.status == STATUS_COMPLETE)
    ).scalar_one()
    failed = db.execute(
        select(func.count(Document.id)).where(Document.status == STATUS_FAILED)
    ).scalar_one()
    high = db.execute(
        select(func.count(Document.id)).where(
            Document.priority == "High", Document.status == STATUS_COMPLETE
        )
    ).scalar_one()
    departments = db.execute(
        select(func.count(func.distinct(Document.department))).where(
            Document.department.notin_(["", UNASSIGNED_DEPARTMENT])
        )
    ).scalar_one()
    pages = db.execute(select(func.coalesce(func.sum(Document.page_count), 0))).scalar_one()

    open_actions = 0
    for (raw,) in db.execute(
        select(Document.actions).where(Document.status == STATUS_COMPLETE)
    ):
        try:
            open_actions += len(json.loads(raw or "[]"))
        except (TypeError, ValueError):
            continue

    return LibraryStats(
        total_documents=int(total),
        analysis_complete=int(complete),
        in_progress=int(total) - int(complete) - int(failed),
        failed=int(failed),
        high_priority=int(high),
        departments_covered=int(departments),
        total_pages=int(pages or 0),
        open_actions=open_actions,
        ai_engine="claude" if settings.ai_enabled else "rule-based",
    )


@router.get("/facets", response_model=SearchFacets)
def search_facets(
    q: str = Query("", description="Keyword query"),
    department: list[str] = Query(default=[]),
    priority: list[str] = Query(default=[]),
    file_type: list[str] = Query(default=[]),
    status: list[str] = Query(default=[]),
    document_type: list[str] = Query(default=[]),
    date_range: str = Query("any"),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    db: Session = Depends(get_db),
) -> SearchFacets:
    params = _build_params(
        q, department, priority, file_type, status, document_type,
        date_range, date_from, date_to, "relevance", 1, 20,
    )
    return SearchFacets(**search_service.facet_counts(db, params))


# ------------------------------------------------------------------- search
@router.get("/search", response_model=DocumentSearchResponse)
@router.get("", response_model=DocumentSearchResponse)
def search_documents(
    q: str = Query("", description="Free-text query across filename, AI analysis and content"),
    department: list[str] = Query(default=[]),
    priority: list[str] = Query(default=[]),
    file_type: list[str] = Query(default=[]),
    status: list[str] = Query(default=[]),
    document_type: list[str] = Query(default=[]),
    date_range: str = Query("any"),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    sort: str = Query("relevance"),
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.kmrl_default_page_size, ge=1, le=settings.kmrl_max_page_size),
    db: Session = Depends(get_db),
) -> DocumentSearchResponse:
    started = time.perf_counter()
    params = _build_params(
        q, department, priority, file_type, status, document_type,
        date_range, date_from, date_to, sort, page, page_size,
    )
    rows, total = search_service.search_documents(db, params)

    total_pages = (total + params.page_size - 1) // params.page_size if total else 0
    message: str | None = None
    suggestion: str | None = None
    if total == 0:
        message = "No documents found"
        if params.q and params.has_filters:
            suggestion = "Try a different keyword or remove a filter."
        elif params.q:
            suggestion = "Try a different keyword, or search for a department such as “Safety”."
        elif params.has_filters:
            suggestion = "Try removing a filter to widen the results."
        else:
            suggestion = "Upload a document to start building the KMRL library."

    return DocumentSearchResponse(
        items=[DocumentSummary(**row) for row in rows],
        total=total,
        page=params.page,
        page_size=params.page_size,
        total_pages=total_pages,
        has_next=params.page < total_pages,
        has_prev=params.page > 1 and total > 0,
        query=params.q,
        applied_filters=params.as_applied_filters(),
        sort=params.normalised_sort(),
        took_ms=int((time.perf_counter() - started) * 1000),
        message=message,
        suggestion=suggestion,
    )


# ------------------------------------------------------------------- upload
@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    uploaded_by: str = Form("KMRL User"),
    db: Session = Depends(get_db),
) -> UploadResponse:
    filename = file.filename or "document"
    if not extraction_service.is_supported(filename):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type “{Path(filename).suffix or 'unknown'}”. "
                "Upload a PDF, DOCX, TXT or image file."
            ),
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {settings.kmrl_max_upload_mb} MB limit.",
        )

    document = processing_service.create_document(db, raw, filename, uploaded_by)
    background_tasks.add_task(processing_service.process_document, document.id)

    return UploadResponse(
        document=DocumentSummary(**_document_summary(document)),
        message=f"“{document.filename}” uploaded. Extraction and AI analysis are running.",
    )


# ------------------------------------------------------------ single document
def _get_document_or_404(db: Session, document_id: int) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} was not found.")
    return document


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: int, db: Session = Depends(get_db)) -> DocumentDetail:
    document = _get_document_or_404(db, document_id)
    payload = _document_summary(document)
    payload.update(
        {
            "secondary_departments": document.json_field("secondary_departments"),
            "key_points": document.json_field("key_points"),
            "actions": document.json_field("actions"),
            "deadlines": document.json_field("deadlines"),
            "risks": document.json_field("risks"),
            "compliance_flags": document.json_field("compliance_flags"),
            "analysis_model": document.analysis_model,
            "analysis_confidence": document.analysis_confidence,
            "analysis_error": document.analysis_error,
            "error_message": document.error_message,
            "extraction_method": document.extraction_method,
            "char_count": document.char_count,
            "uploaded_by": document.uploaded_by,
            "text_preview": (document.extracted_text or "")[:TEXT_PREVIEW_CHARS],
        }
    )
    return DocumentDetail(**payload)


@router.get("/{document_id}/text")
def get_document_text(
    document_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(20_000, ge=1, le=200_000),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    document = _get_document_or_404(db, document_id)
    text = document.extracted_text or ""
    chunk = text[offset : offset + limit]
    return {
        "document_id": document.id,
        "filename": document.filename,
        "offset": offset,
        "limit": limit,
        "total_chars": len(text),
        "has_more": offset + limit < len(text),
        "text": chunk,
    }


@router.get("/{document_id}/file")
def download_document(document_id: int, db: Session = Depends(get_db)) -> FileResponse:
    document = _get_document_or_404(db, document_id)
    path = settings.upload_dir / document.stored_filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="The stored file is no longer available.")
    return FileResponse(path, filename=document.filename)


@router.post("/{document_id}/reprocess", response_model=UploadResponse)
def reprocess_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> UploadResponse:
    document = _get_document_or_404(db, document_id)
    document.status = "processing"
    document.status_detail = "Re-queued for extraction and analysis"
    document.error_message = ""
    document.analysis_error = ""
    db.commit()
    db.refresh(document)
    background_tasks.add_task(processing_service.process_document, document.id)
    return UploadResponse(
        document=DocumentSummary(**_document_summary(document)),
        message=f"“{document.filename}” is being re-processed.",
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db)) -> None:
    document = _get_document_or_404(db, document_id)
    path = settings.upload_dir / document.stored_filename
    db.delete(document)
    db.commit()
    if path.exists():
        try:
            path.unlink()
        except OSError:  # pragma: no cover - best effort cleanup
            pass
