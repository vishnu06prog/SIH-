"""SQLAlchemy models for the KMRL document intelligence platform."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base
from .taxonomy import STATUS_LABELS, STATUS_QUEUED, UNASSIGNED_DEPARTMENT


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    """One uploaded source document plus everything derived from it."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)

    # --- file facts --------------------------------------------------------
    filename: Mapped[str] = mapped_column(String(512), index=True)
    stored_filename: Mapped[str] = mapped_column(String(512))
    file_ext: Mapped[str] = mapped_column(String(16), default="")
    file_type: Mapped[str] = mapped_column(String(16), default="PDF", index=True)
    mime_type: Mapped[str] = mapped_column(String(128), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    checksum: Mapped[str] = mapped_column(String(64), default="", index=True)
    uploaded_by: Mapped[str] = mapped_column(String(128), default="KMRL User")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # --- processing state --------------------------------------------------
    status: Mapped[str] = mapped_column(String(32), default=STATUS_QUEUED, index=True)
    status_detail: Mapped[str] = mapped_column(String(512), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")

    # --- extraction --------------------------------------------------------
    extracted_text: Mapped[str] = mapped_column(Text, default="")
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    extraction_method: Mapped[str] = mapped_column(String(64), default="")
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    language: Mapped[str] = mapped_column(String(32), default="Unknown")

    # --- AI analysis -------------------------------------------------------
    title: Mapped[str] = mapped_column(String(512), default="")
    document_type: Mapped[str] = mapped_column(String(64), default="", index=True)
    department: Mapped[str] = mapped_column(
        String(64), default=UNASSIGNED_DEPARTMENT, index=True
    )
    secondary_departments: Mapped[str] = mapped_column(Text, default="[]")
    priority: Mapped[str] = mapped_column(String(16), default="Medium", index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    key_points: Mapped[str] = mapped_column(Text, default="[]")
    keywords: Mapped[str] = mapped_column(Text, default="[]")
    actions: Mapped[str] = mapped_column(Text, default="[]")
    deadlines: Mapped[str] = mapped_column(Text, default="[]")
    risks: Mapped[str] = mapped_column(Text, default="[]")
    compliance_flags: Mapped[str] = mapped_column(Text, default="[]")
    analysis_engine: Mapped[str] = mapped_column(String(32), default="")
    analysis_model: Mapped[str] = mapped_column(String(64), default="")
    analysis_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    analysis_error: Mapped[str] = mapped_column(Text, default="")

    # Lower-cased haystack of every searchable AI field. Keeps keyword search
    # to a single indexed column instead of scanning JSON blobs at query time.
    search_blob: Mapped[str] = mapped_column(Text, default="")

    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentPage.page_number",
    )

    __table_args__ = (
        Index("ix_documents_dept_priority", "department", "priority"),
        Index("ix_documents_status_uploaded", "status", "uploaded_at"),
    )

    # --- helpers -----------------------------------------------------------
    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

    def json_field(self, name: str) -> list[Any]:
        raw = getattr(self, name) or "[]"
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return []
        return value if isinstance(value, list) else []

    def set_json_field(self, name: str, value: Any) -> None:
        setattr(self, name, json.dumps(value or [], ensure_ascii=False))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Document id={self.id} filename={self.filename!r} status={self.status}>"


class DocumentPage(Base):
    """Per-page text, kept for page-level citations and future retrieval."""

    __tablename__ = "document_pages"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, default="")
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    document: Mapped[Document] = relationship(back_populates="pages")

    __table_args__ = (Index("ix_pages_doc_page", "document_id", "page_number"),)
