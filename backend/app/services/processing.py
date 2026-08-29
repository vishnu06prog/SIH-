"""Upload storage and the extraction → analysis pipeline."""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal
from ..models import Document, DocumentPage
from ..taxonomy import (
    STATUS_ANALYZING,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_OCR,
    STATUS_PROCESSING,
)
from . import analysis as analysis_service
from . import extraction as extraction_service
from .search import build_search_blob

logger = logging.getLogger(__name__)

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\- ]+")

# Below this mean tesseract confidence the extracted text is unreliable.
LOW_OCR_QUALITY_THRESHOLD = 70.0


def safe_filename(filename: str) -> str:
    name = Path(filename or "document").name
    cleaned = SAFE_NAME_RE.sub("_", name).strip() or "document"
    return cleaned[:200]


def store_upload(raw: bytes, filename: str) -> tuple[Path, str]:
    """Persist the uploaded bytes under a collision-free name."""
    settings.ensure_dirs()
    original = safe_filename(filename)
    stored_name = f"{uuid.uuid4().hex}_{original}"
    destination = settings.upload_dir / stored_name
    destination.write_bytes(raw)
    return destination, stored_name


def checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def create_document(db: Session, raw: bytes, filename: str, uploaded_by: str) -> Document:
    path, stored_name = store_upload(raw, filename)
    original = safe_filename(filename)
    document = Document(
        filename=original,
        stored_filename=stored_name,
        file_ext=Path(original).suffix.lower().lstrip("."),
        file_type=extraction_service.classify_file_type(original),
        size_bytes=len(raw),
        checksum=checksum(raw),
        uploaded_by=uploaded_by or "KMRL User",
        uploaded_at=datetime.now(timezone.utc).replace(tzinfo=None),
        status=STATUS_PROCESSING,
        status_detail="Queued for text extraction",
        title=analysis_service._title_from_filename(original),
        search_blob=original.lower(),
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    logger.info("Stored upload %s as %s", original, path.name)
    return document


def _set_status(db: Session, document: Document, status: str, detail: str = "") -> None:
    document.status = status
    document.status_detail = detail
    db.add(document)
    db.commit()


def process_document(document_id: int) -> None:
    """Run the full pipeline for one document. Safe to call in a worker thread."""
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        if document is None:
            logger.warning("process_document: document %s vanished", document_id)
            return

        path = settings.upload_dir / document.stored_filename
        if not path.exists():
            document.status = STATUS_FAILED
            document.status_detail = "Stored file is missing"
            document.error_message = "The uploaded file could not be found on disk."
            db.commit()
            return

        # --- 1. extraction ------------------------------------------------
        _set_status(db, document, STATUS_PROCESSING, "Extracting text")
        try:
            needs_ocr = document.file_type == "IMAGE" or document.file_type == "PDF"
            if needs_ocr and extraction_service.ocr_available():
                _set_status(db, document, STATUS_OCR, "Reading document (OCR enabled)")
            result = extraction_service.extract_document(path, document.filename)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Extraction failed for document %s", document_id)
            document.status = STATUS_FAILED
            document.status_detail = "Text extraction failed"
            document.error_message = str(exc)[:1000]
            db.commit()
            return

        document.extracted_text = result.text
        document.page_count = result.page_count
        document.word_count = result.word_count
        document.char_count = result.char_count
        document.extraction_method = result.method
        document.ocr_used = result.ocr_used
        document.ocr_quality = result.ocr_quality
        document.language = result.language

        db.query(DocumentPage).filter(DocumentPage.document_id == document.id).delete()
        for page in result.pages:
            db.add(
                DocumentPage(
                    document_id=document.id,
                    page_number=page.page_number,
                    text=page.text,
                    char_count=page.char_count,
                    ocr_used=page.ocr_used,
                    ocr_confidence=page.ocr_confidence,
                )
            )
        db.commit()

        # --- 2. analysis --------------------------------------------------
        _set_status(db, document, STATUS_ANALYZING, "Running AI analysis")
        analysis = analysis_service.analyse_document(
            result.text, document.filename, result.page_count
        )

        document.title = analysis.title or document.title
        document.document_type = analysis.document_type
        document.department = analysis.department
        document.set_json_field("secondary_departments", analysis.secondary_departments)
        document.priority = analysis.priority
        document.summary = analysis.summary
        document.set_json_field("key_points", analysis.key_points)
        document.set_json_field("keywords", analysis.keywords)
        document.set_json_field("actions", analysis.actions)
        document.set_json_field("deadlines", analysis.deadlines)
        document.set_json_field("risks", analysis.risks)
        document.set_json_field("compliance_flags", analysis.compliance_flags)
        document.analysis_engine = analysis.engine
        document.analysis_model = analysis.model
        document.analysis_confidence = analysis.confidence
        document.analysis_error = analysis.error

        warnings = list(result.warnings)
        if is_low_quality_ocr(document):
            warnings.append(
                "OCR confidence is low; verify details against the original document."
            )
        document.error_message = " ".join(warnings)[:1000]

        document.search_blob = build_search_blob(document)
        document.status = STATUS_COMPLETE
        document.status_detail = (
            "Analysed with Claude" if analysis.engine == "claude" else "Analysed with rule engine"
        )
        document.processed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        logger.info("Document %s processed (%s)", document_id, analysis.engine)
    except Exception:  # noqa: BLE001  pragma: no cover - last-resort guard
        logger.exception("Unexpected failure processing document %s", document_id)
        db.rollback()
        document = db.get(Document, document_id)
        if document is not None:
            document.status = STATUS_FAILED
            document.status_detail = "Processing failed"
            db.commit()
    finally:
        db.close()


def is_low_quality_ocr(document: Document) -> bool:
    return bool(
        document.ocr_used
        and document.ocr_quality is not None
        and document.ocr_quality < LOW_OCR_QUALITY_THRESHOLD
    )
