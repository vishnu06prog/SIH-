"""SQLAlchemy models for the KMRL DMS.

Primary keys are UUID strings so identifiers are stable across environments and
safe to expose in URLs. The schema is normalised: everything the AI derives from
a document (classification, summary, entities, actions, deadlines, risks,
stakeholders, routing) lives in its own table with a foreign key back to the
document, which is what makes traceability and human correction possible.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base
from .taxonomy import (
    ACTION_PENDING,
    STATUS_LABELS,
    STATUS_UPLOADED,
    UNASSIGNED_DEPARTMENT,
)


def utcnow() -> datetime:
    """Naive UTC — SQLite has no tz-aware storage, so we normalise on write."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_id() -> str:
    return str(uuid.uuid4())


UUID_LEN = 36


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )


class JSONMixin:
    """Portable JSON-in-TEXT helpers (SQLite and PostgreSQL behave the same)."""

    def json_field(self, name: str) -> Any:
        raw = getattr(self, name) or "[]"
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return []

    def json_list(self, name: str) -> list[Any]:
        value = self.json_field(name)
        return value if isinstance(value, list) else []

    def set_json_field(self, name: str, value: Any) -> None:
        setattr(self, name, json.dumps(value if value is not None else [], ensure_ascii=False))


# --------------------------------------------------------------------- people
class Role(Base):
    __tablename__ = "roles"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    description: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    users: Mapped[list["User"]] = relationship(back_populates="role")


class Department(Base, TimestampMixin):
    __tablename__ = "departments"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    code: Mapped[str] = mapped_column(String(8), default="")
    description: Mapped[str] = mapped_column(String(512), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    users: Mapped[list["User"]] = relationship(
        back_populates="department", foreign_keys="User.department_id"
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128), default="")
    hashed_password: Mapped[str] = mapped_column(String(256))
    designation: Mapped[str] = mapped_column(String(128), default="")
    phone: Mapped[str] = mapped_column(String(32), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    role_id: Mapped[str] = mapped_column(String(UUID_LEN), ForeignKey("roles.id"), index=True)
    department_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("departments.id"), nullable=True, index=True
    )

    role: Mapped[Role] = relationship(back_populates="users", lazy="joined")
    department: Mapped[Department | None] = relationship(
        back_populates="users", foreign_keys=[department_id], lazy="joined"
    )

    @property
    def role_name(self) -> str:
        return self.role.name if self.role else ""

    @property
    def department_name(self) -> str:
        return self.department.name if self.department else UNASSIGNED_DEPARTMENT


# ------------------------------------------------------------------ documents
class Document(Base, JSONMixin):
    """One uploaded source document plus its file-level facts and state."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)

    # --- file facts --------------------------------------------------------
    title: Mapped[str] = mapped_column(String(512), default="", index=True)
    filename: Mapped[str] = mapped_column(String(512), index=True)
    stored_filename: Mapped[str] = mapped_column(String(512))
    file_ext: Mapped[str] = mapped_column(String(16), default="")
    file_type: Mapped[str] = mapped_column(String(16), default="PDF", index=True)
    mime_type: Mapped[str] = mapped_column(String(128), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    checksum: Mapped[str] = mapped_column(String(64), default="", index=True)
    source: Mapped[str] = mapped_column(String(64), default="Manual Upload", index=True)
    confidentiality: Mapped[str] = mapped_column(String(32), default="Internal", index=True)
    tags: Mapped[str] = mapped_column(Text, default="[]")

    uploaded_by_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id"), nullable=True, index=True
    )
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    # --- processing state --------------------------------------------------
    status: Mapped[str] = mapped_column(String(24), default=STATUS_UPLOADED, index=True)
    status_detail: Mapped[str] = mapped_column(String(512), default="")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str] = mapped_column(Text, default="")

    # --- extraction --------------------------------------------------------
    # ``extracted_text`` is ALWAYS the original-language text, exactly as it
    # came out of the PDF/DOCX/OCR. It is never overwritten by translation.
    extracted_text: Mapped[str] = mapped_column(Text, default="")
    # ``normalised_text`` is the English representation the AI layer, the
    # keyword index and the embeddings all run on. For English documents it is
    # identical to ``extracted_text``.
    normalised_text: Mapped[str] = mapped_column(Text, default="")
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    extraction_method: Mapped[str] = mapped_column(String(64), default="")
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_page_count: Mapped[int] = mapped_column(Integer, default=0)
    ocr_languages: Mapped[str] = mapped_column(String(64), default="")
    needs_ocr_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    ocr_verified_by_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id"), nullable=True
    )
    ocr_verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    text_edited: Mapped[bool] = mapped_column(Boolean, default=False)
    language: Mapped[str] = mapped_column(String(32), default="Unknown", index=True)
    languages_detected: Mapped[str] = mapped_column(Text, default="[]")

    # --- translation -------------------------------------------------------
    translation_status: Mapped[str] = mapped_column(String(24), default="not_required", index=True)
    translation_engine: Mapped[str] = mapped_column(String(32), default="")
    translation_model: Mapped[str] = mapped_column(String(64), default="")
    translation_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    translated_page_count: Mapped[int] = mapped_column(Integer, default=0)

    # --- structure ---------------------------------------------------------
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    has_tables: Mapped[bool] = mapped_column(Boolean, default=False)
    image_count: Mapped[int] = mapped_column(Integer, default=0)

    # --- classification / priority ----------------------------------------
    document_type: Mapped[str] = mapped_column(String(64), default="Other", index=True)
    priority: Mapped[str] = mapped_column(String(16), default="Medium", index=True)
    department_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("departments.id"), nullable=True, index=True
    )
    declared_department_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("departments.id"), nullable=True
    )

    # --- review ------------------------------------------------------------
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reviewed_by_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id"), nullable=True
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    review_note: Mapped[str] = mapped_column(Text, default="")

    # --- versions & duplicates --------------------------------------------
    version_no: Mapped[int] = mapped_column(Integer, default=1)
    version_group_id: Mapped[str] = mapped_column(String(UUID_LEN), default=new_id, index=True)
    previous_document_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id"), nullable=True
    )
    duplicate_of_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id"), nullable=True, index=True
    )
    duplicate_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- analysis provenance ----------------------------------------------
    analysis_engine: Mapped[str] = mapped_column(String(32), default="")
    analysis_model: Mapped[str] = mapped_column(String(64), default="")
    analysis_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    analysis_error: Mapped[str] = mapped_column(Text, default="")

    # Lower-cased haystack of every searchable field, so keyword search hits a
    # single indexed column instead of scanning JSON blobs at query time.
    search_blob: Mapped[str] = mapped_column(Text, default="")
    keywords: Mapped[str] = mapped_column(Text, default="[]")

    # --- relationships -----------------------------------------------------
    department: Mapped[Department | None] = relationship(
        foreign_keys=[department_id], lazy="joined"
    )
    declared_department: Mapped[Department | None] = relationship(
        foreign_keys=[declared_department_id]
    )
    uploaded_by: Mapped[User | None] = relationship(
        foreign_keys=[uploaded_by_id], lazy="joined"
    )
    reviewed_by: Mapped[User | None] = relationship(foreign_keys=[reviewed_by_id])

    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentPage.page_number",
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )
    classifications: Mapped[list["DocumentClassification"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentClassification.created_at.desc()",
    )
    summaries: Mapped[list["DocumentSummary"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentSummary.created_at.desc()",
    )
    entities: Mapped[list["Entity"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    actions: Mapped[list["Action"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    deadlines: Mapped[list["Deadline"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    risks: Mapped[list["Risk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    stakeholders: Mapped[list["Stakeholder"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    routing_results: Mapped[list["RoutingResult"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="RoutingResult.confidence.desc()",
    )
    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version_no.desc()",
        foreign_keys="DocumentVersion.document_id",
    )

    __table_args__ = (
        Index("ix_documents_dept_priority", "department_id", "priority"),
        Index("ix_documents_status_uploaded", "status", "uploaded_at"),
        Index("ix_documents_type_uploaded", "document_type", "uploaded_at"),
    )

    # --- helpers -----------------------------------------------------------
    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

    @property
    def department_name(self) -> str:
        return self.department.name if self.department else UNASSIGNED_DEPARTMENT

    @property
    def uploader_name(self) -> str:
        return self.uploaded_by.full_name if self.uploaded_by else "System"

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Document {self.id} {self.filename!r} {self.status}>"


class DocumentPage(Base, JSONMixin):
    """Per-page text, kept so every AI statement can cite a page."""

    __tablename__ = "document_pages"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    # Original-language text for this page — the evidence a citation points at.
    text: Mapped[str] = mapped_column(Text, default="")
    # English rendering, empty when the page is already English.
    translated_text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(32), default="Unknown", index=True)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    extraction_method: Mapped[str] = mapped_column(String(32), default="")
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    translation_engine: Mapped[str] = mapped_column(String(32), default="")
    tables: Mapped[str] = mapped_column(Text, default="[]")
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    is_edited: Mapped[bool] = mapped_column(Boolean, default=False)

    document: Mapped[Document] = relationship(back_populates="pages")

    __table_args__ = (Index("ix_pages_doc_page", "document_id", "page_number"),)


class DocumentVersion(Base):
    """A recorded lineage step: this document supersedes an earlier one."""

    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    version_group_id: Mapped[str] = mapped_column(String(UUID_LEN), index=True)
    version_no: Mapped[int] = mapped_column(Integer, default=1)
    previous_document_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id"), nullable=True
    )
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    change_summary: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped[Document] = relationship(
        back_populates="versions", foreign_keys=[document_id]
    )


class DocumentChunk(Base):
    """A retrieval unit: ~1 200 characters of text with its page number."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str] = mapped_column(String(256), default="")
    # ``text`` is what gets embedded and keyword-matched (English/normalised).
    text: Mapped[str] = mapped_column(Text, default="")
    # ``original_text`` is the untouched source-language passage, so a citation
    # can always show the Malayalam the answer actually rests on.
    original_text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(32), default="English")
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped[Document] = relationship(back_populates="chunks")
    embedding: Mapped["DocumentEmbedding | None"] = relationship(
        back_populates="chunk", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (Index("ix_chunks_doc_index", "document_id", "chunk_index"),)


class DocumentEmbedding(Base):
    """A chunk's vector, stored as raw float32 bytes for portability.

    SQLite and PostgreSQL both store the same bytes; when pgvector is available
    the vector column is populated in addition, so ANN search can use it.
    """

    __tablename__ = "document_embeddings"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    chunk_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("document_chunks.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    model: Mapped[str] = mapped_column(String(128), default="")
    dim: Mapped[int] = mapped_column(Integer, default=0)
    vector: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    chunk: Mapped[DocumentChunk] = relationship(back_populates="embedding")


class DocumentClassification(Base):
    """Every classification the document has ever had, AI or human."""

    __tablename__ = "document_classifications"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    category: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning: Mapped[str] = mapped_column(Text, default="")
    engine: Mapped[str] = mapped_column(String(32), default="")
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_human_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    corrected_by_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped[Document] = relationship(back_populates="classifications")
    corrected_by: Mapped[User | None] = relationship()


class DocumentSummary(Base, JSONMixin):
    """Executive + detailed summary with key points and next steps."""

    __tablename__ = "document_summaries"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    executive_summary: Mapped[str] = mapped_column(Text, default="")
    detailed_summary: Mapped[str] = mapped_column(Text, default="")
    key_points: Mapped[str] = mapped_column(Text, default="[]")
    next_steps: Mapped[str] = mapped_column(Text, default="[]")
    ai_explanation: Mapped[str] = mapped_column(Text, default="")
    engine: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    generated_by_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped[Document] = relationship(back_populates="summaries")


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    entity_type: Mapped[str] = mapped_column(String(32), index=True)
    value: Mapped[str] = mapped_column(String(512))
    normalised_value: Mapped[str] = mapped_column(String(512), default="")
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_language: Mapped[str] = mapped_column(String(32), default="English")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped[Document] = relationship(back_populates="entities")


class Action(Base):
    __tablename__ = "actions"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    description: Mapped[str] = mapped_column(Text)
    responsible_department_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("departments.id"), nullable=True, index=True
    )
    responsible_person: Mapped[str] = mapped_column(String(256), default="")
    assigned_to_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id"), nullable=True, index=True
    )
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    due_date_text: Mapped[str] = mapped_column(String(128), default="")
    priority: Mapped[str] = mapped_column(String(16), default="Medium", index=True)
    status: Mapped[str] = mapped_column(String(24), default=ACTION_PENDING, index=True)
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Original-language sentence the action was lifted from, plus its English
    # rendering — both kept so the extraction stays explainable.
    source_excerpt: Mapped[str] = mapped_column(Text, default="")
    source_excerpt_en: Mapped[str] = mapped_column(Text, default="")
    source_language: Mapped[str] = mapped_column(String(32), default="English")
    notes: Mapped[str] = mapped_column(Text, default="")
    created_by_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id"), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    document: Mapped[Document] = relationship(back_populates="actions")
    responsible_department: Mapped[Department | None] = relationship(lazy="joined")
    assigned_to: Mapped[User | None] = relationship(
        foreign_keys=[assigned_to_id], lazy="joined"
    )

    __table_args__ = (Index("ix_actions_status_due", "status", "due_date"),)


class Deadline(Base):
    __tablename__ = "deadlines"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    action_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("actions.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(512))
    description: Mapped[str] = mapped_column(Text, default="")
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    due_date_text: Mapped[str] = mapped_column(String(128), default="")
    deadline_type: Mapped[str] = mapped_column(String(32), default="General", index=True)
    department_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("departments.id"), nullable=True, index=True
    )
    priority: Mapped[str] = mapped_column(String(16), default="Medium", index=True)
    is_compliance: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_met: Mapped[bool] = mapped_column(Boolean, default=False)
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_excerpt: Mapped[str] = mapped_column(Text, default="")
    source_language: Mapped[str] = mapped_column(String(32), default="English")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped[Document] = relationship(back_populates="deadlines")
    department: Mapped[Department | None] = relationship(lazy="joined")


class Risk(Base):
    __tablename__ = "risks"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    description: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(16), default="Medium", index=True)
    category: Mapped[str] = mapped_column(String(32), default="Operational", index=True)
    mitigation: Mapped[str] = mapped_column(Text, default="")
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_excerpt: Mapped[str] = mapped_column(Text, default="")
    source_language: Mapped[str] = mapped_column(String(32), default="English")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped[Document] = relationship(back_populates="risks")


class Stakeholder(Base):
    __tablename__ = "stakeholders"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(256))
    designation: Mapped[str] = mapped_column(String(256), default="")
    organisation: Mapped[str] = mapped_column(String(256), default="")
    department_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("departments.id"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(16), default="Internal")
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped[Document] = relationship(back_populates="stakeholders")
    department: Mapped[Department | None] = relationship(lazy="joined")


class RoutingResult(Base):
    """Which department should see this document, why, and how urgently."""

    __tablename__ = "routing_results"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    department_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("departments.id"), index=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")
    urgency: Mapped[str] = mapped_column(String(16), default="Medium", index=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    is_override: Mapped[bool] = mapped_column(Boolean, default=False)
    overridden_by_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id"), nullable=True
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    engine: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped[Document] = relationship(back_populates="routing_results")
    department: Mapped[Department] = relationship(lazy="joined")

    __table_args__ = (
        UniqueConstraint("document_id", "department_id", name="uq_routing_doc_dept"),
    )


class RoutingRule(Base, TimestampMixin):
    """An admin-authored override that forces a routing decision."""

    __tablename__ = "routing_rules"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(128))
    keywords: Mapped[str] = mapped_column(Text, default="[]")
    document_type: Mapped[str] = mapped_column(String(64), default="")
    department_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("departments.id"), index=True
    )
    priority: Mapped[str] = mapped_column(String(16), default="")
    urgency: Mapped[str] = mapped_column(String(16), default="High")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_by_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id"), nullable=True
    )

    department: Mapped[Department] = relationship(lazy="joined")

    def keyword_list(self) -> list[str]:
        try:
            value = json.loads(self.keywords or "[]")
        except (TypeError, ValueError):
            return []
        return [str(item).strip().lower() for item in value if str(item).strip()]


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    notification_type: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(256))
    message: Mapped[str] = mapped_column(Text, default="")
    priority: Mapped[str] = mapped_column(String(16), default="Medium", index=True)
    document_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True
    )
    action_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("actions.id", ondelete="CASCADE"), nullable=True
    )
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    read_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    __table_args__ = (Index("ix_notifications_user_read", "user_id", "is_read"),)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    actor_email: Mapped[str] = mapped_column(String(256), default="", index=True)
    actor_role: Mapped[str] = mapped_column(String(32), default="")
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(32), default="", index=True)
    entity_id: Mapped[str] = mapped_column(String(UUID_LEN), default="", index=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    ip_address: Mapped[str] = mapped_column(String(64), default="")
    user_agent: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    user: Mapped[User | None] = relationship()


class SearchHistory(Base):
    __tablename__ = "search_history"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    query: Mapped[str] = mapped_column(String(1024))
    mode: Mapped[str] = mapped_column(String(24), default="keyword", index=True)
    filters: Mapped[str] = mapped_column(Text, default="{}")
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
